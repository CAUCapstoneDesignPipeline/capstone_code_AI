import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "deploy/gcp"))
from host_driver import HostDriver
from manifest import digest
from test_release import manifest


@pytest.fixture(scope="module")
def tls(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic-ca")
    def run(*args):
        subprocess.run(["openssl", *args], cwd=root, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem", "-days", "365", "-subj", "/CN=synthetic-test-ca")
    run("req", "-new", "-newkey", "rsa:2048", "-nodes", "-keyout", "leaf.key", "-out", "leaf.csr", "-subj", "/CN=synthetic-ai")
    (root/"extensions.cnf").write_text("subjectAltName=DNS:capsnote-prod-ai.asia-northeast3-a.c.capsnote.internal\nbasicConstraints=CA:FALSE\n")
    run("x509", "-req", "-in", "leaf.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial", "-out", "leaf.pem", "-days", "365", "-extfile", "extensions.cnf")
    return {"certificate":(root/"leaf.pem").read_text(), "privateKey":(root/"leaf.key").read_text(), "caCertificate":(root/"ca.pem").read_text()}


def material_driver(tmp_path, monkeypatch, tls):
    host = {"projectId":"capsnote", "projectNumber":"169516793433", "region":"asia-northeast3",
            "privateIp":"10.60.2.10", "appIp":"10.60.1.10",
            "serverName":"capsnote-prod-ai.asia-northeast3-a.c.capsnote.internal",
            "releaseBucket":"capsnote-ai-releases", "artifactBucket":"capsnote-ai-artifacts",
            "repository":"asia-northeast3-docker.pkg.dev/capsnote/capstone-ai/ai",
            "secretResources":{k:"projects/capsnote/secrets/"+name for k,name in [("serviceToken","ai-token"),("providerKey","ai-provider"),("tlsBundle","ai-tls")]}}
    m = manifest()
    m["configRevision"] = digest(host)
    m["secrets"] = {k:v+"/versions/1" for k,v in host["secretResources"].items()}
    m["tls"]["caSha256"] = hashlib.sha256(tls["caCertificate"].encode()).hexdigest()
    raw = json.dumps({'df':{'synthetic':1},'n_docs':1,'version':'b'*12,'config_version':'a'*12}).encode()
    m["idf"]["sha256"] = hashlib.sha256(raw).hexdigest()
    driver = HostDriver(host, tmp_path, "a"*40, control=tmp_path/"public-control")
    actual = driver.command
    def command(argv, data=None, timeout=120):
        if argv[:3] == ["gcloud","storage","cat"]: return raw
        if argv[:4] == ["gcloud","secrets","versions","access"]:
            name=argv[argv.index("--secret")+1]
            return json.dumps(tls).encode() if name=="ai-tls" else b"synthetic-not-a-real-secret-value"
        if argv[:2] == ["docker","run"]: return b"aaaaaaaaaaaa\n"
        if argv[:3] == ["docker","image","inspect"]:
            return json.dumps([{"Architecture":"amd64","Config":{"User":"10001:10001","Labels":{"org.opencontainers.image.revision":m["sourceSha"]}}}]).encode()
        if argv[0] in {"gcloud","docker"}: return b"synthetic-token"
        return actual(argv,data,timeout)
    monkeypatch.setattr(driver,"command",command)
    monkeypatch.setattr("os.chown",lambda *args:None)
    return driver,m


def test_material_pins_tls_idf_and_secret_files(tmp_path,monkeypatch,tls):
    driver,m=material_driver(tmp_path,monkeypatch,tls)
    driver.prepare(m)
    target=tmp_path/"releases/r2"
    assert (target/"service-token").stat().st_mode & 0o777 == 0o600
    assert json.loads((target/"manifest.json").read_bytes()) == m
    compose=json.loads((target/"compose.json").read_bytes())
    assert compose["services"]["ai"]["ports"] == ["127.0.0.1:8000:8000"]
    assert compose["services"]["ai"]["restart"] == "no"
    assert compose["services"]["ai"]["cpus"] == 1.5
    assert compose["services"]["ai"]["mem_limit"] == "10g"
    assert compose["services"]["ai"]["environment"]["AI_MAX_INFLIGHT"] == "1"
    assert "ca.key" not in {p.name for p in target.iterdir()}
    driver.prepare(m)  # local immutable material reuse for rollback


@pytest.mark.parametrize("field",["idf-hash","ca-hash","wrong-san","wrong-profile","mutable-secret"])
def test_bad_material_rejected_before_runtime_mutation(tmp_path,monkeypatch,tls,field):
    driver,m=material_driver(tmp_path,monkeypatch,tls)
    if field=="idf-hash": m["idf"]["sha256"]="0"*64
    elif field=="ca-hash": m["tls"]["caSha256"]="0"*64
    elif field=="wrong-san": m["tls"]["serverName"]="other.asia-northeast3-a.c.capsnote.internal"
    elif field=="wrong-profile": m["model"]["revision"]="0"*40
    else: m["secrets"]["providerKey"]=m["secrets"]["providerKey"].rsplit("/",1)[0]+"/latest"
    with pytest.raises((ValueError,RuntimeError)): driver.prepare(m)
    assert not (tmp_path/"releases/r2").exists()


def test_idf_configuration_is_checked_in_exact_image_before_activation(tmp_path,monkeypatch,tls):
    driver,m=material_driver(tmp_path,monkeypatch,tls)
    command=driver.command
    def changed_image(argv,data=None,timeout=120):
        if argv[:2]==["docker","run"]: return b"bbbbbbbbbbbb\n"
        return command(argv,data,timeout)
    monkeypatch.setattr(driver,"command",changed_image)
    with pytest.raises(ValueError,match="extractor configuration"):
        driver.prepare(m)
    assert not (tmp_path/"releases/r2").exists()


INSTALL = Path(__file__).parents[2] / "deploy/gcp/install.sh"
# Google add-google-cloud-ops-agent-repo.sh (install.sh에 고정한 hash)의 옵션 파싱을 최소로 재현한다.
# getopts -- '-:'로 읽고 version=* 한 인자 형태만 받는다.
OPS_AGENT_PARSER = r"""
while getopts -- '-:' OPTCHAR; do
  case "${OPTCHAR}" in
    -)
      case "${OPTARG}" in
        also-install) ;;
        version=*) AGENT_VERSION="${OPTARG#*=}" ;;
        *) echo "Unknown option '${OPTARG}'." >&2; exit 1 ;;
      esac
  esac
done
[[ "${AGENT_VERSION:-}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || exit 1
printf '%s\n' "$AGENT_VERSION"
"""


def install_line(pattern):
    return next(line for line in INSTALL.read_text().splitlines() if re.search(pattern, line))


def test_install_passes_ops_agent_version_as_single_argument(tmp_path):
    (tmp_path / "ops-agent.sh").write_text(OPS_AGENT_PARSER)
    call = install_line(r'ops-agent\.sh" --')
    assert re.search(r'--version="\$OPS_AGENT_VERSION"', call) and not re.search(r"--version\s", call)
    run = subprocess.run(["bash", "-c", "set -euo pipefail\n" + call],
                         env={"PATH": "/usr/bin:/bin", "staging": str(tmp_path), "OPS_AGENT_VERSION": "2.73.0"},
                         capture_output=True, text=True)
    assert run.returncode == 0 and run.stdout == "2.73.0\n", run.stderr
    split = subprocess.run(["bash", str(tmp_path / "ops-agent.sh"), "--also-install", "--version", "2.73.0"],
                           capture_output=True, text=True)
    assert split.returncode == 1 and "Unknown option 'version'" in split.stderr


@pytest.mark.parametrize("version,ok", [("2.73.0", True), ("2.73.0~ubuntu24.04", False), ("latest", False),
                                        ("2.*.*", False), ("2.73", False), ("", False)])
def test_install_rejects_bad_ops_agent_version_before_packages(version, ok):
    check = install_line(r'"\$OPS_AGENT_VERSION" =~')
    text = INSTALL.read_text()
    assert text.index(check) < text.index("apt-get update")
    run = subprocess.run(["bash", "-c", check], env={"PATH": "/usr/bin:/bin", "OPS_AGENT_VERSION": version},
                         capture_output=True, text=True)
    assert (run.returncode == 0) is ok
