"""Ubuntu host adapter. Secrets stay in root-only release files, never command arguments/logs."""
import hashlib
import json
import os
import socket
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from manifest import canonical, digest, match, validate
from release import atomic_json


class HostDriver:
    def __init__(self, host, root=Path("/var/lib/capstone-ai"), revision=None, control=Path("/run/capstone-ai-control")):
        self.host, self.root = host, Path(root)
        self.revision = revision or Path("/opt/capstone-ai/REVISION").read_text().strip()
        self.releases = self.root / "releases"
        self.control = Path(control)
        self.control.mkdir(parents=True, exist_ok=True)
        os.chmod(self.control, 0o755)  # container can stat drain flags, never secret files

    def command(self, argv, data=None, timeout=120):
        result = subprocess.run(argv, input=data, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
        if result.returncode:
            raise RuntimeError("host command failed")
        return result.stdout

    def write_private(self, file, data):
        fd = os.open(file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    def prepare(self, manifest):
        validate(manifest, self.host, self.revision)
        target = self.releases / manifest["releaseId"]
        if target.exists():
            if json.loads((target / "manifest.json").read_bytes()) != manifest:
                raise ValueError("immutable local release conflict")
            return
        self.releases.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(dir=self.releases) as staging:
            stage = Path(staging)
            idf = manifest["idf"]
            raw = self.command(["gcloud", "storage", "cat", f"gs://{idf['bucket']}/{idf['object']}#{idf['generation']}"])
            if len(raw) > 32 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != idf["sha256"]:
                raise ValueError("IDF hash or size mismatch")
            idf_document = json.loads(raw)
            if set(idf_document) != {"df", "n_docs", "version", "config_version"}:
                raise ValueError("IDF must include pinned extractor configuration")
            match(idf_document["config_version"], r"[0-9a-f]{12}", "IDF config version")
            match(idf_document["version"], r"[0-9a-f]{12}", "IDF corpus version")
            count = idf_document["n_docs"]
            if type(count) is not int or count < 1 or not isinstance(idf_document["df"], dict):
                raise ValueError("invalid IDF corpus")
            if any(not isinstance(k, str) or type(v) is not int or not 0 <= v <= count for k, v in idf_document["df"].items()):
                raise ValueError("invalid IDF document frequencies")
            self.write_private(stage / "idf.json", raw)
            os.chown(stage / "idf.json", 10001, 10001)
            secrets = {}
            for key, ref in manifest["secrets"].items():
                project, secret, version = ref.split("/")[1::2]
                raw = self.command(["gcloud", "secrets", "versions", "access", version, "--secret", secret, "--project", project])
                if not raw or len(raw) > 128 * 1024:
                    raise ValueError("invalid secret payload size")
                secrets[key] = raw.decode()
            for key in ["serviceToken", "providerKey"]:
                if any(c in secrets[key] for c in "\r\n\x00") or len(secrets[key]) < 16:
                    raise ValueError("invalid credential payload")
            tls = json.loads(secrets.pop("tlsBundle"))
            if set(tls) != {"certificate", "privateKey", "caCertificate"}:
                raise ValueError("TLS bundle must contain leaf key/cert and public CA only")
            if hashlib.sha256(tls["caCertificate"].encode()).hexdigest() != manifest["tls"]["caSha256"]:
                raise ValueError("public CA hash mismatch")
            for key, filename in [("certificate", "leaf.pem"), ("privateKey", "leaf.key"), ("caCertificate", "ca.pem")]:
                self.write_private(stage / filename, tls[key].encode())
            self.command(["openssl", "verify", "-CAfile", str(stage / "ca.pem"), "-verify_hostname",
                          manifest["tls"]["serverName"], str(stage / "leaf.pem")])
            self.command(["openssl", "x509", "-in", str(stage / "leaf.pem"), "-checkend", "1209600", "-noout"])
            details = self.command(["openssl", "x509", "-in", str(stage / "leaf.pem"), "-text", "-noout"])
            if b"CA:TRUE" in details:
                raise ValueError("CA key must never be deployed")
            public_cert = self.command(["openssl", "x509", "-in", str(stage / "leaf.pem"), "-pubkey", "-noout"])
            public_key = self.command(["openssl", "pkey", "-in", str(stage / "leaf.key"), "-pubout"])
            if public_cert != public_key:
                raise ValueError("TLS leaf key mismatch")
            token = self.command(["gcloud", "auth", "print-access-token"])
            registry = manifest["image"].split("/")[0]
            with tempfile.TemporaryDirectory() as auth:
                self.command(["docker", "--config", auth, "login", "-u", "oauth2accesstoken", "--password-stdin", registry], token)
                self.command(["docker", "--config", auth, "pull", manifest["image"]], timeout=1200)
            image = json.loads(self.command(["docker", "image", "inspect", manifest["image"]]))[0]
            if image["Architecture"] != "amd64" or image["Config"]["User"] != "10001:10001":
                raise ValueError("wrong architecture or image user")
            if image["Config"]["Labels"].get("org.opencontainers.image.revision") != manifest["sourceSha"]:
                raise ValueError("image source label mismatch")
            config_version = self.command(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python",
                manifest["image"], "-c", "from app.services.concepts.config import ExtractConfig; print(ExtractConfig().version())"]).decode().strip()
            if idf_document["config_version"] != config_version:
                raise ValueError("IDF extractor configuration does not match image")
            env = {"AI_SERVICE_TOKEN": secrets["serviceToken"], "AI_MAX_INFLIGHT": "2", "AI_DEV_UI": "0",
                   "AI_EMBEDDING_PROFILES": "embeddings-bge-m3-1", "AI_CONCEPT_IDF": "/run/ai/idf.json",
                   "HF_HOME": "/var/cache/ai/huggingface", "AI_DRAIN_FILE": "/run/ai/control/runtime-drain",
                   "AI_LLM_TIMEOUT_SECONDS": str(manifest["provider"]["timeoutSeconds"])}
            provider = manifest["provider"]["name"]
            env[{"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}[provider]] = secrets["providerKey"]
            for role, model in manifest["provider"]["models"].items():
                env[f"AI_{role.upper()}_LLM_PROVIDER"] = provider
                env[f"AI_{role.upper()}_LLM_MODEL"] = model
                env[f"AI_{role.upper()}_LLM_EFFORT"] = manifest["provider"]["efforts"][role]
            for key, enabled in manifest["flags"].items():
                env[f"AI_{key.upper()}_LLM"] = "1" if enabled else "0"
            compose = {"name": "capstone-ai", "services": {"ai": {
                "image": manifest["image"], "platform": "linux/amd64", "user": "10001:10001",
                "read_only": True, "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"],
                "environment": {k: v.replace("$", "$$") for k, v in env.items()},
                "ports": ["127.0.0.1:8000:8000"], "cpus": 3, "mem_limit": "10g", "pids_limit": 512,
                "stop_grace_period": "180s", "restart": "no", "tmpfs": ["/tmp:size=512m,mode=1777"],
                "volumes": [f"{target}/idf.json:/run/ai/idf.json:ro", f"{self.control}:/run/ai/control:ro",
                            f"{self.root}/model-cache:/var/cache/ai"],
                "logging": {"driver": "json-file", "options": {"max-size": "10m", "max-file": "3"}}}}}
            atomic_json(stage / "compose.json", compose)
            atomic_json(stage / "manifest.json", manifest)
            self.write_private(stage / "service-token", secrets["serviceToken"].encode())
            nginx = f'''server {{
    listen {self.host['privateIp']}:8443 ssl;
    server_name {manifest['tls']['serverName']};
    ssl_certificate {target}/leaf.pem;
    ssl_certificate_key {target}/leaf.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    access_log off;
    error_log /var/log/nginx/capstone-ai-error.log crit;
    allow {self.host['appIp']};
    allow {self.host['privateIp']};
    deny all;
    client_max_body_size 2m;
    if (-f {self.control}/drain) {{ return 503; }}
    location /internal/v1/ {{
        proxy_pass http://127.0.0.1:8000;
        proxy_connect_timeout 5s;
        proxy_read_timeout 160s;
        proxy_send_timeout 160s;
        proxy_next_upstream off;
    }}
    location / {{ return 404; }}
}}
'''
            self.write_private(stage / "nginx.conf", nginx.encode())
            os.rename(stage, target)

    def drain(self, enabled):
        for name in ["drain", "runtime-drain"]:
            flag = self.control / name
            if enabled:
                flag.touch(mode=0o644)
            else:
                flag.unlink(missing_ok=True)

    def activate(self, release):
        target = self.releases / release
        manifest = json.loads((target / "manifest.json").read_bytes())
        validate(manifest, self.host, self.revision)
        cache = self.root / "model-cache"
        cache.mkdir(mode=0o700, exist_ok=True)
        os.chown(cache, 10001, 10001)
        # Stop old process gracefully while both gates are closed. No automatic restart on boot.
        self.command(["docker", "compose", "-p", "capstone-ai", "-f", str(target / "compose.json"), "stop", "--timeout", "180"], timeout=200)
        (self.control / "runtime-drain").unlink(missing_ok=True)
        self.command(["docker", "compose", "-p", "capstone-ai", "-f", str(target / "compose.json"), "up", "-d", "--force-recreate", "--no-build", "--pull", "never"], timeout=180)
        conf = Path("/etc/nginx/conf.d/capstone-ai.conf")
        temporary = conf.with_suffix(".pending")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target / "nginx.conf")
        os.replace(temporary, conf)
        self.command(["nginx", "-t"])
        self.command(["systemctl", "reload", "nginx"])

    def health(self, release):
        target = self.releases / release
        manifest = json.loads((target / "manifest.json").read_bytes())
        token = (target / "service-token").read_text()
        started = time.monotonic()
        while time.monotonic() - started < 900:
            try:
                req = urllib.request.Request("http://127.0.0.1:8000/internal/v1/health/ready", headers={
                    "Authorization": "Bearer " + token, "X-Request-Id": str(uuid.uuid4())})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.load(response)
                if data.get("status") != "ready" or data.get("schemaVersion") != "1.0" or not set(manifest["profiles"]) <= set(data.get("profileVersions", [])):
                    raise ValueError("required profile missing")
                context = ssl.create_default_context(cafile=str(target / "ca.pem"))
                with socket.create_connection((self.host["privateIp"], 8443), timeout=5) as raw:
                    with context.wrap_socket(raw, server_hostname=manifest["tls"]["serverName"]):
                        pass
                bad = urllib.request.Request("http://127.0.0.1:8000/internal/v1/health/live", headers={
                    "Authorization": "Bearer invalid-token", "X-Request-Id": str(uuid.uuid4())})
                try:
                    urllib.request.urlopen(bad, timeout=5)
                except urllib.error.HTTPError as error:
                    if error.code != 401:
                        raise ValueError("bad token not rejected") from None
                else:
                    raise ValueError("bad token accepted")
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2)
        raise RuntimeError("readiness timeout")

    def stop(self):
        # Only this tool's project, never other Docker resources.
        names = self.command(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=capstone-ai"]).decode().split()
        if names:
            self.command(["docker", "stop", "--time", "180", *names], timeout=200)
