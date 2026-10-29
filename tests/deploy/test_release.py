import copy
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/gcp"))
from manifest import MODEL_REVISION, digest, validate, validate_host
from release import Coordinator, atomic_json
from submit import immutable, validate_identity


def manifest(release="r2"):
    return {"schemaVersion": 1, "kind": "ai", "releaseId": release, "sourceSha": "b"*40,
        "image": "asia-northeast3-docker.pkg.dev/capsnote/capstone-ai/ai@sha256:" + "c"*64,
        "bootstrapRevision": "a"*40, "configRevision": "c"*64,
        "model": {"profile": "embeddings-bge-m3-1", "name": "BAAI/bge-m3", "revision": MODEL_REVISION},
        "profiles": ["extract-det-3", "embeddings-bge-m3-1"],
        "idf": {"bucket": "capsnote-ai-artifacts", "object": "idf/synthetic.json", "generation": "123", "sha256": "d"*64, "sourceSha": "b"*40},
        "provider": {"name": "anthropic", "models": {role:"synthetic-model-1" for role in ["relation", "facts", "explain", "verify", "sound"]},
                     "efforts": {role:"medium" for role in ["relation", "facts", "explain", "verify", "sound"]}, "timeoutSeconds": 90},
        "flags": {"relation": False, "explain": False, "verify": False},
        "secrets": {key:f"projects/capsnote/secrets/{key}/versions/1" for key in ["serviceToken", "providerKey", "tlsBundle"]},
        "tls": {"serverName": "capsnote-prod-ai.asia-northeast3-a.c.capsnote.internal", "caSha256": "e"*64},
        "cache": {"responses": "disabled", "modelScope": "pinned-revision"}}


def request(m, expected="r1", op="op-1", action="deploy"):
    return {"operationId":op, "operation":action, "releaseId":m["releaseId"], "expectedRelease":expected, "manifestSha256":digest(m)}


class Store:
    def __init__(self):
        self.value = {"current":"r1", "previous":"r0"}
        self.generation = 1
        self.fail = None
        self.objects = {}
    def read(self, key):
        if key != "state.json":
            return self.objects.get(key), 1 if key in self.objects else 0
        return copy.deepcopy(self.value), self.generation
    def write(self, key, value, generation):
        if key != "state.json":
            if key in self.objects:
                raise RuntimeError("conflict")
            self.objects[key] = copy.deepcopy(value)
            return
        if self.fail == "before" or generation != self.generation:
            raise RuntimeError("CAS failed")
        self.value, self.generation = copy.deepcopy(value), self.generation+1
        if self.fail == "after":
            raise RuntimeError("response lost")


class Driver:
    def __init__(self):
        self.active = "r1"
        self.draining = False
        self.fail = None
        self.activations = []
    def prepare(self, m):
        if self.fail == "prepare":
            raise ValueError("bad secret/IDF")
    def drain(self, enabled):
        self.draining = enabled
    def activate(self, release):
        self.active = release
        self.activations.append(release)
    def health(self, release):
        if self.fail == release:
            raise ValueError("bad readiness")
    def stop(self):
        self.active = None


@pytest.fixture
def setup(tmp_path):
    store, driver = Store(), Driver()
    return Coordinator(tmp_path, store, driver), store, driver


def test_success_and_same_operation_retry_no_redeploy(setup):
    c, store, driver = setup
    m = manifest()
    assert c.run(request(m), m)["phase"] == "succeeded"
    assert c.run(request(m), m)["phase"] == "succeeded"
    assert store.value == {"current":"r2", "previous":"r1"}
    assert driver.activations == ["r2"] and not driver.draining


def test_operation_id_payload_conflict(setup):
    c, _, _ = setup
    m = manifest()
    c.run(request(m), m)
    n = manifest("r3")
    with pytest.raises(ValueError, match="operationId"):
        c.run(request(n, expected="r2"), n)


@pytest.mark.parametrize("mode", ["before", "after"])
def test_cas_failure_reconciles_cloud_and_host(setup, mode):
    c, store, driver = setup
    store.fail = mode
    m = manifest()
    if mode == "before":
        with pytest.raises(RuntimeError):
            c.run(request(m), m)
        assert driver.active == "r1" and store.value["current"] == "r1"
    else:
        assert c.run(request(m), m)["phase"] == "succeeded"
        assert driver.active == "r2" and store.value["current"] == "r2"
    assert not driver.draining


def test_bad_readiness_restores_entire_previous_release(setup):
    c, store, driver = setup
    driver.fail = "r2"
    with pytest.raises(RuntimeError):
        c.run(request(manifest()), manifest())
    assert driver.active == "r1" and store.value["current"] == "r1"
    assert json.loads((c.root/"operations/op-1.json").read_bytes())["phase"] == "reverted"


def test_bad_material_does_not_stop_current(setup):
    c, store, driver = setup
    driver.fail = "prepare"
    with pytest.raises(ValueError):
        c.run(request(manifest()), manifest())
    assert driver.active == "r1" and not driver.draining and not driver.activations


def test_no_baseline_failure_keeps_drain(setup):
    c, store, driver = setup
    store.value = {"current":None, "previous":None}
    driver.fail = "r2"
    with pytest.raises(RuntimeError):
        c.run(request(manifest(), expected=None), manifest())
    assert driver.active is None and driver.draining


def test_runner_cancel_reboot_recovers_incomplete_journal(setup):
    c, store, driver = setup
    m = manifest()
    record = {"operationId":"interrupted", "requestHash":digest(request(m)), "phase":"activating",
        "before":store.value, "after":{"current":"r2", "previous":"r1"}, "target":"r2", "generation":1}
    c.journal("interrupted", record)
    driver.active = "r2"
    c.boot()
    assert driver.active == "r1" and not driver.draining


def test_unknown_cloud_state_never_reopens_traffic(setup):
    c, store, driver = setup
    record = {"operationId":"interrupted", "phase":"committing", "before":{"current":"r1", "previous":"r0"},
              "after":{"current":"r2", "previous":"r1"}, "target":"r2", "generation":1}
    c.journal("interrupted", record)
    store.value = {"current":"outsider", "previous":"r1"}
    with pytest.raises(RuntimeError):
        c.boot()
    assert driver.draining


def test_concurrent_operations_have_one_winner_without_sleep(setup):
    c, store, driver = setup
    barrier = threading.Barrier(3)
    results = []
    def run(target):
        barrier.wait()
        try:
            m = manifest(target)
            c.run(request(m, op=target), m)
            results.append("ok")
        except ValueError:
            results.append("stale")
    threads = [threading.Thread(target=run, args=(x,)) for x in ["r2", "r3"]]
    for t in threads: t.start()
    barrier.wait()
    for t in threads: t.join(5)
    assert sorted(results) == ["ok", "stale"]
    assert len(driver.activations) == 1


def test_rollback_only_previous(setup):
    c, store, driver = setup
    m = manifest("r0")
    assert c.run(request(m, action="rollback"), m)["phase"] == "succeeded"
    assert store.value == {"current":"r0", "previous":"r1"}


@pytest.mark.parametrize("change", [
    lambda m:m.update(image="image:latest"), lambda m:m["secrets"].update(serviceToken="projects/p/secrets/x/versions/latest"),
    lambda m:m["idf"].update(generation="latest"), lambda m:m["idf"].update(sourceSha="f"*40),
    lambda m:m["model"].update(revision="main"), lambda m:m.update(profiles=["embeddings-hash-dev-1"]),
    lambda m:m["tls"].update(serverName="localhost"), lambda m:m["provider"].update(name=""),
    lambda m:m["provider"].update(timeoutSeconds=150), lambda m:m["cache"].update(responses="enabled"),
    lambda m:m.update(unexpected="field"), lambda m:m["flags"].update(verify="false")])
def test_manifest_rejects_mutable_unapproved_or_unsafe_configuration(change):
    m = manifest()
    change(m)
    with pytest.raises(ValueError): validate(m)


def test_immutable_cloud_upload_retry_and_conflict():
    store = Store()
    immutable(store, "releases/r1.json", {"one":1})
    immutable(store, "releases/r1.json", {"one":1})
    with pytest.raises(ValueError): immutable(store, "releases/r1.json", {"one":2})


def deployment_identity():
    return {"GITHUB_REPOSITORY":"CAUCapstoneDesignPipeline/capstone_code_AI", "GITHUB_REPOSITORY_ID":"1397258740",
        "GITHUB_REPOSITORY_OWNER_ID":"324880339", "GITHUB_REF":"refs/heads/develop", "GITHUB_EVENT_NAME":"workflow_dispatch",
        "GITHUB_WORKFLOW_REF":"CAUCapstoneDesignPipeline/capstone_code_AI/.github/workflows/gcp-deploy.yml@refs/heads/develop",
        "GCP_AI_DEPLOY_TOOL_SHA":"a"*40, "GCP_PROJECT_ID":"capsnote", "GCP_ZONE":"asia-northeast3-a",
        "GCP_VM_NAME":"capsnote-prod-ai", "RELEASE_BUCKET":"capsnote-ai-releases"}


def test_deployment_identity_accepts_exact_caller():
    validate_identity(deployment_identity(), "a"*40)


@pytest.mark.parametrize("key,value", [("GITHUB_REPOSITORY","fork/repo"),("GITHUB_REPOSITORY_ID","1"),
    ("GITHUB_REPOSITORY_OWNER_ID","1"),("GITHUB_REF","refs/heads/main"),("GITHUB_EVENT_NAME","pull_request"),
    ("GITHUB_WORKFLOW_REF","CAUCapstoneDesignPipeline/capstone_code_AI/.github/workflows/other.yml@refs/heads/develop"),
    ("GCP_AI_DEPLOY_TOOL_SHA","b"*40),("GCP_VM_NAME","vm; command"),("GCP_ZONE","us-central1-a")])
def test_deployment_identity_rejects_wrong_claims_or_unapproved_tool(key,value):
    env=deployment_identity()
    env[key]=value
    with pytest.raises(ValueError): validate_identity(env,"a"*40)


def test_retry_after_commit_before_undrain_reopens_only_matching_state(setup):
    c,store,driver=setup
    m=manifest()
    c.run(request(m),m)
    driver.draining=True
    c.run(request(m),m)
    assert not driver.draining


@pytest.mark.parametrize("status",[401,403,429,500])
def test_cloud_permission_or_provider_error_is_never_empty_state(monkeypatch,status):
    import urllib.error
    from release import Cloud
    cloud=Cloud("capsnote-ai-releases")
    cloud.token="synthetic-token"
    def fail(*args,**kwargs):
        raise urllib.error.HTTPError("https://storage.googleapis.com",status,"failure",{},None)
    monkeypatch.setattr("urllib.request.urlopen",fail)
    with pytest.raises(RuntimeError): cloud.read("state.json")


def test_cloud_404_is_only_absent_state(monkeypatch):
    import urllib.error
    from release import Cloud
    cloud=Cloud("capsnote-ai-releases")
    cloud.token="synthetic-token"
    def fail(*args,**kwargs):
        raise urllib.error.HTTPError("https://storage.googleapis.com",404,"missing",{},None)
    monkeypatch.setattr("urllib.request.urlopen",fail)
    assert cloud.read("state.json")==(None,0)
