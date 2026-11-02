#!/usr/bin/env python3
"""Durable AI operation coordinator. One host flock, GCS generation CAS, fail-closed recovery."""
import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from manifest import canonical, digest, match, validate


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
        name = stream.name
    os.replace(name, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Cloud:
    def __init__(self, bucket):
        match(bucket, r"[a-z0-9][a-z0-9.-]{2,62}", "bucket")
        self.bucket = bucket
        self.token = None

    def request(self, url, data=None, allow_missing=False):
        if self.token is None:
            result = subprocess.run(["gcloud", "auth", "print-access-token"],
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if result.returncode:
                raise RuntimeError("ADC token unavailable")
            self.token = result.stdout.decode().strip()
        request = urllib.request.Request(url, data=data, headers={"Authorization": "Bearer " + self.token,
                                          "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if allow_missing and error.code == 404:
                return None
            raise RuntimeError("GCS API operation failed") from None

    def read(self, key):
        encoded = urllib.parse.quote(key, safe="")
        url = f"https://storage.googleapis.com/storage/v1/b/{self.bucket}/o/{encoded}"
        metadata = self.request(url, allow_missing=True)
        if metadata is None:
            return None, 0
        generation = int(metadata["generation"])
        return self.request(f"{url}?alt=media&generation={generation}"), generation

    def write(self, key, value, generation):
        query = urllib.parse.urlencode({"uploadType": "media", "name": key, "ifGenerationMatch": generation})
        self.request(f"https://storage.googleapis.com/upload/storage/v1/b/{self.bucket}/o?{query}", canonical(value))


class Coordinator:
    def __init__(self, root, cloud, driver, host=None, revision=None):
        self.root, self.cloud, self.driver = Path(root), cloud, driver
        self.host, self.revision = host, revision

    def journal(self, op, record):
        atomic_json(self.root / "operations" / f"{op}.json", record)

    def run(self, request, manifest):
        match(request.get("operationId"), r"[a-z0-9][a-z0-9-]{0,62}", "operationId")
        if set(request) != {"operationId", "operation", "releaseId", "manifestSha256", "expectedRelease"}:
            raise ValueError("invalid operation request")
        if request["operation"] not in {"deploy", "rollback"}:
            raise ValueError("invalid operation")
        if request["expectedRelease"] is not None:
            match(request["expectedRelease"], r"[a-z0-9][a-z0-9-]{0,62}", "expected release")
        validate(manifest, self.host, self.revision)
        if request["releaseId"] != manifest["releaseId"] or request["manifestSha256"] != digest(manifest):
            raise ValueError("manifest integrity mismatch")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / "deployment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.recover()
            op = request["operationId"]
            path = self.root / "operations" / f"{op}.json"
            previous = json.loads(path.read_bytes()) if path.exists() else None
            if previous and previous["requestHash"] != digest(request):
                raise ValueError("operationId reused with different request")
            if previous and previous["phase"] == "succeeded":
                observed, _ = self.cloud.read("state.json")
                if observed == previous["after"]:
                    self.driver.health(previous["target"])
                    self.driver.drain(False)
                return previous
            state, generation = self.cloud.read("state.json")
            state = state or {"current": None, "previous": None}
            if state["current"] != request["expectedRelease"]:
                raise ValueError("stale expected release")
            if state["current"] == request["releaseId"]:
                raise ValueError("target already current under another operation")
            if request["operation"] == "rollback" and state["previous"] != request["releaseId"]:
                raise ValueError("rollback target is not previous release")
            record = {"operationId": op, "requestHash": digest(request), "phase": "preparing",
                      "before": state, "generation": generation, "target": request["releaseId"],
                      "after": {"current": request["releaseId"], "previous": state["current"]}}
            self.journal(op, record)
            try:
                self.driver.prepare(manifest)
            except Exception:
                record["phase"] = "preparation-failed"
                self.journal(op, record)
                raise
            record["phase"] = "prepared"
            self.journal(op, record)
            try:
                self.driver.drain(True)
                record["phase"] = "activating"
                self.journal(op, record)
                self.driver.activate(request["releaseId"])
                self.driver.health(request["releaseId"])
                record["phase"] = "committing"
                self.journal(op, record)
                try:
                    self.cloud.write("state.json", record["after"], generation)
                except Exception:
                    observed, _ = self.cloud.read("state.json")
                    if observed != record["after"]:
                        raise
                record["phase"] = "succeeded"
                self.journal(op, record)
                self.driver.drain(False)
                return record
            except Exception:
                # Unknown cloud state is never followed by blind rollback or reopening traffic.
                self.recover_record(record)
                raise RuntimeError("deployment failed; inspect sanitized operation journal") from None

    def recover_record(self, record, reopen=True):
        self.driver.drain(True)
        observed, _ = self.cloud.read("state.json")
        observed = observed or {"current": None, "previous": None}
        if observed == record["after"]:
            self.driver.activate(record["target"])
            self.driver.health(record["target"])
            record["phase"] = "succeeded"
        elif observed == record["before"] and record["before"]["current"]:
            self.driver.activate(record["before"]["current"])
            self.driver.health(record["before"]["current"])
            record["phase"] = "reverted"
        elif observed == record["before"]:
            self.driver.stop()
            record["phase"] = "failed-no-baseline"
            self.journal(record["operationId"], record)
            return
        else:
            record["phase"] = "reconcile-required"
            self.journal(record["operationId"], record)
            raise RuntimeError("unexpected GCS state; maintenance retained")
        self.journal(record["operationId"], record)
        if reopen:
            self.driver.drain(False)

    def recover(self, reopen=True):
        for path in sorted((self.root / "operations").glob("*.json")):
            record = json.loads(path.read_bytes())
            if record["phase"] not in {"succeeded", "reverted", "failed-no-baseline", "preparation-failed"}:
                self.recover_record(record, reopen=reopen)

    def boot(self):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with (self.root / "deployment.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self.driver.drain(True)
            try:
                self.recover(reopen=False)
                state, _ = self.cloud.read("state.json")
                if state and state["current"]:
                    self.driver.activate(state["current"])
                    self.driver.health(state["current"])
                    self.driver.drain(False)
            except Exception:
                self.driver.drain(True)
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "boot", "validate"])
    parser.add_argument("--operation-id")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("/etc/capstone-ai/host.json"))
    args = parser.parse_args()
    if args.command == "validate":
        validate(json.loads(args.manifest.read_bytes()))
        print("manifest valid (host/resource bindings checked on host)")
        return
    if os.geteuid() != 0:
        raise RuntimeError("host coordinator requires root")
    from host_driver import HostDriver
    host = json.loads(args.config.read_bytes())
    revision = Path("/opt/capstone-ai/REVISION").read_text().strip()
    cloud = Cloud(host["releaseBucket"])
    coordinator = Coordinator("/var/lib/capstone-ai", cloud, HostDriver(host), host, revision)
    if args.command == "boot":
        coordinator.boot()
    else:
        match(args.operation_id, r"[a-z0-9][a-z0-9-]{0,62}", "operationId")
        request, _ = cloud.read(f"operations/{args.operation_id}.json")
        if not request or request.get("operationId") != args.operation_id:
            raise ValueError("missing or mismatched operation")
        match(request.get("releaseId"), r"[a-z0-9][a-z0-9-]{0,62}", "releaseId")
        manifest, _ = cloud.read(f"releases/{request['releaseId']}.json")
        coordinator.run(request, manifest)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never print subprocess/SDK exception messages or secret-bearing data.
        raise SystemExit("AI deployment failed; inspect operation phase and authenticated readiness") from None
