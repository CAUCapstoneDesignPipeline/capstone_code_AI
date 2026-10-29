"""Approved workflow submission: immutable request/manifest, detached host operation."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from manifest import digest, match, validate
from release import Cloud


def validate_identity(env, tool_sha):
    expected = {"GITHUB_REPOSITORY": "CAUCapstoneDesignPipeline/capstone_code_AI",
                "GITHUB_REPOSITORY_ID": "1397258740", "GITHUB_REPOSITORY_OWNER_ID": "324880339",
                "GITHUB_REF": "refs/heads/develop", "GITHUB_EVENT_NAME": "workflow_dispatch",
                "GITHUB_WORKFLOW_REF": "CAUCapstoneDesignPipeline/capstone_code_AI/.github/workflows/gcp-deploy.yml@refs/heads/develop"}
    for key, value in expected.items():
        if env.get(key) != value:
            raise ValueError("unexpected deployment identity")
    match(tool_sha, r"[0-9a-f]{40}", "tool SHA")
    if tool_sha != env.get("GCP_AI_DEPLOY_TOOL_SHA"):
        raise ValueError("unapproved tool SHA")
    match(env.get("GCP_PROJECT_ID"), r"[a-z][a-z0-9-]{4,28}[a-z0-9]", "project")
    match(env.get("GCP_ZONE"), r"asia-northeast3-[a-z]", "zone")
    match(env.get("GCP_VM_NAME"), r"[a-z][a-z0-9-]{0,62}", "VM")
    match(env.get("RELEASE_BUCKET"), r"[a-z0-9][a-z0-9.-]{2,62}", "release bucket")


def immutable(cloud, key, value):
    existing, _ = cloud.read(key)
    if existing is not None:
        if existing != value:
            raise ValueError("immutable object conflict")
        return
    try:
        cloud.write(key, value, 0)
    except Exception:
        existing, _ = cloud.read(key)
        if existing != value:
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--operation", choices=["deploy", "rollback"], required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--expected-release", required=True)
    parser.add_argument("--tool-sha", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    validate_identity(os.environ, args.tool_sha)
    match(args.operation_id, r"[a-z0-9][a-z0-9-]{0,62}", "operation ID")
    match(args.manifest, r"deploy/gcp/releases/[a-z0-9][a-z0-9-]{0,62}\.json", "manifest path")
    file = Path(args.manifest)
    if not file.resolve().is_relative_to(Path.cwd().resolve()):
        raise ValueError("manifest path escapes checkout")
    manifest = validate(json.loads(file.read_bytes()))
    if file.stem != manifest["releaseId"] or manifest["bootstrapRevision"] != args.tool_sha:
        raise ValueError("manifest tool/release mismatch")
    expected = None if args.expected_release == "none" else args.expected_release
    if expected is not None:
        match(expected, r"[a-z0-9][a-z0-9-]{0,62}", "expected release")
    if args.preflight_only:
        print("AI deployment preflight passed; no cloud writes")
        return
    cloud = Cloud(os.environ["RELEASE_BUCKET"])
    request = {"operationId": args.operation_id, "operation": args.operation, "releaseId": manifest["releaseId"],
               "expectedRelease": expected, "manifestSha256": digest(manifest)}
    immutable(cloud, f"releases/{manifest['releaseId']}.json", manifest)
    immutable(cloud, f"operations/{args.operation_id}.json", request)
    base = ["gcloud", "compute", "ssh", os.environ["GCP_VM_NAME"], "--project", os.environ["GCP_PROJECT_ID"],
            "--zone", os.environ["GCP_ZONE"], "--tunnel-through-iap", "--quiet", "--command"]
    unit = "capstone-ai-" + args.operation_id
    # Detached systemd unit persists when Actions/IAP session is cancelled.
    command = f"sudo systemd-run --unit={unit} --collect --property=Type=exec --property=UMask=0077 --property=TimeoutStartSec=1800 /usr/bin/python3 /opt/capstone-ai/release.py run --operation-id {args.operation_id}"
    result = subprocess.run(base + [command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Existing unit is normal for a retry. Require a journal rather than declaring success.
    for _ in range(90):
        journal = subprocess.run(base + [f"sudo cat /var/lib/capstone-ai/operations/{args.operation_id}.json"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if journal.returncode == 0:
            state = json.loads(journal.stdout)
            if state.get("requestHash") != digest(request):
                raise ValueError("host operation mismatch")
            if state["phase"] == "succeeded":
                print(json.dumps({"operationId": args.operation_id, "releaseId": manifest["releaseId"], "phase": "succeeded"}))
                return
            if state["phase"] in {"reverted", "failed-no-baseline", "reconcile-required", "preparation-failed"}:
                raise RuntimeError("host failed; maintenance/recovery state must be inspected")
        elif result.returncode != 0:
            raise RuntimeError("host submission failed and no matching journal exists")
        time.sleep(20)
    raise RuntimeError("poll timeout; host operation may still be running; retry same operation ID")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("AI submission failed; inspect sanitized operation journal; do not generate a new ID blindly") from None
