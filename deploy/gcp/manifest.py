"""AI release contract v1: pinned artifacts/config, never credentials or note content."""
import hashlib
import json
import re

MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
ROLES = {"relation", "facts", "explain", "verify", "sound"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def match(value, pattern, field):
    if not isinstance(value, str) or not re.fullmatch(pattern, value):
        raise ValueError(f"invalid {field}")


def keys(value, expected, field):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid {field} fields")


def validate(m, host=None, revision=None):
    keys(m, {"schemaVersion", "kind", "releaseId", "sourceSha", "image", "bootstrapRevision", "configRevision",
             "model", "profiles", "idf", "provider", "flags", "secrets", "tls", "cache"}, "manifest")
    if type(m["schemaVersion"]) is not int or m["schemaVersion"] != 1 or m["kind"] != "ai":
        raise ValueError("unsupported manifest")
    match(m["releaseId"], r"[a-z0-9][a-z0-9-]{0,62}", "releaseId")
    for key in ["sourceSha", "bootstrapRevision"]:
        match(m[key], r"[0-9a-f]{40}", key)
    match(m["configRevision"], r"[0-9a-f]{64}", "configRevision")
    match(m["image"], r"asia-northeast3-docker\.pkg\.dev/[a-z][a-z0-9-]+/capstone-ai/ai@sha256:[0-9a-f]{64}", "image")
    if m["model"] != {"profile": "embeddings-bge-m3-1", "name": "BAAI/bge-m3", "revision": MODEL_REVISION}:
        raise ValueError("unsupported model/profile/revision")
    if not isinstance(m["profiles"], list) or not m["profiles"] or len(m["profiles"]) != len(set(m["profiles"])):
        raise ValueError("invalid profiles")
    for profile in m["profiles"]:
        match(profile, r"[a-z][a-z0-9-]{0,79}", "profile")
        if "dev" in profile:
            raise ValueError("development profile forbidden")
    if m["model"]["profile"] not in m["profiles"]:
        raise ValueError("embedding readiness required")
    keys(m["idf"], {"bucket", "object", "generation", "sha256", "sourceSha"}, "idf")
    match(m["idf"]["bucket"], r"[a-z0-9][a-z0-9.-]{2,62}", "idf bucket")
    match(m["idf"]["object"], r"idf/[a-zA-Z0-9/_-]+\.json", "idf object")
    match(m["idf"]["generation"], r"[1-9][0-9]*", "idf generation")
    match(m["idf"]["sha256"], r"[0-9a-f]{64}", "idf sha256")
    if m["idf"]["sourceSha"] != m["sourceSha"]:
        raise ValueError("IDF source revision mismatch")
    keys(m["provider"], {"name", "models", "efforts", "timeoutSeconds"}, "provider")
    if m["provider"]["name"] not in {"anthropic", "openai", "google"}:
        raise ValueError("provider must be selected")
    keys(m["provider"]["models"], ROLES, "models")
    keys(m["provider"]["efforts"], ROLES, "efforts")
    for role in ROLES:
        match(m["provider"]["models"][role], r"[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,127}", "provider model")
        if m["provider"]["efforts"][role] not in {"low", "medium", "high"}:
            raise ValueError("invalid effort")
    if type(m["provider"]["timeoutSeconds"]) is not int or not 1 <= m["provider"]["timeoutSeconds"] <= 90:
        raise ValueError("provider timeout exceeds runtime budget")
    keys(m["flags"], {"relation", "explain", "verify"}, "flags")
    if any(type(v) is not bool for v in m["flags"].values()):
        raise ValueError("flags must be booleans")
    keys(m["secrets"], {"serviceToken", "providerKey", "tlsBundle"}, "secrets")
    for value in m["secrets"].values():
        match(value, r"projects/[a-z0-9-]+/secrets/[a-zA-Z0-9_-]+/versions/[1-9][0-9]*", "secret version")
    keys(m["tls"], {"serverName", "caSha256"}, "tls")
    match(m["tls"]["serverName"], r"[a-z0-9-]+\.[a-z0-9-]+\.c\.[a-z0-9-]+\.internal", "TLS SAN")
    match(m["tls"]["caSha256"], r"[0-9a-f]{64}", "CA hash")
    if m["cache"] != {"responses": "disabled", "modelScope": "pinned-revision"}:
        raise ValueError("response caches must be disabled until user scope/retention contract is approved")
    if revision and m["bootstrapRevision"] != revision:
        raise ValueError("installed deployment tool revision mismatch")
    if host:
        validate_host(host)
        if m["configRevision"] != digest(host):
            raise ValueError("host configuration mismatch")
        if not m["image"].startswith(host["repository"] + "@sha256:"):
            raise ValueError("wrong image repository")
        if m["idf"]["bucket"] != host["artifactBucket"] or m["tls"]["serverName"] != host["serverName"]:
            raise ValueError("wrong artifact/TLS destination")
        for key, resource in host["secretResources"].items():
            if m["secrets"][key].rsplit("/versions/", 1)[0] != resource:
                raise ValueError("wrong secret resource")
    return m


def validate_host(host):
    import ipaddress
    keys(host, {"projectId", "projectNumber", "region", "privateIp", "appIp", "serverName", "releaseBucket", "artifactBucket", "repository", "secretResources"}, "host")
    match(host["projectId"], r"[a-z][a-z0-9-]{4,28}[a-z0-9]", "project")
    match(host["projectNumber"], r"[0-9]+", "project number")
    if host["region"] != "asia-northeast3":
        raise ValueError("unexpected region")
    for key in ["privateIp", "appIp"]:
        address = ipaddress.ip_address(host[key])
        if address.version != 4 or not address.is_private or address.is_loopback:
            raise ValueError("private IPv4 required")
    for key in ["releaseBucket", "artifactBucket"]:
        match(host[key], r"[a-z0-9][a-z0-9.-]{2,62}", key)
    if host["releaseBucket"] == host["artifactBucket"]:
        raise ValueError("release and artifact buckets must differ")
    expected = "asia-northeast3-docker.pkg.dev/" + host["projectId"] + "/capstone-ai/ai"
    if host["repository"] != expected:
        raise ValueError("wrong AI AR repository")
    match(host["serverName"], r"[a-z0-9-]+\.asia-northeast3-[a-z]\.c\." + re.escape(host["projectId"]) + r"\.internal", "serverName")
    keys(host["secretResources"], {"serviceToken", "providerKey", "tlsBundle"}, "secretResources")
    for resource in host["secretResources"].values():
        match(resource, "projects/" + re.escape(host["projectId"]) + r"/secrets/[a-z0-9-]+", "secret resource")
    return host
