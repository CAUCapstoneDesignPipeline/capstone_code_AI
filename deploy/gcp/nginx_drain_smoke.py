"""Local Docker regression for the nonroot Nginx drain gate; no cloud/provider calls."""
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

IMAGE = "nginx:stable-alpine@sha256:0e52bd054126cdc040b1276c0af347aeb02a41b9cfe305d6094767ac30e97949"


def main():
    name = "capstone-ai-drain-" + uuid.uuid4().hex[:12]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        root.chmod(0o755)
        control = root / "control"
        control.mkdir(mode=0o755)
        flag = control / "drain"
        flag.touch(mode=0o644)
        config = root / "nginx.conf"
        config.write_text('''user nginx;
events {}
http {
    access_log off;
    server {
        listen 8080;
        if (-f /run/capstone-ai-control/drain) { return 503; }
        location / { return 200 "ready"; }
    }
}
''')
        config.chmod(0o644)
        try:
            subprocess.run(["docker", "run", "-d", "--name", name, "-p", "127.0.0.1::8080",
                "-v", f"{config}:/etc/nginx/nginx.conf:ro", "-v", f"{control}:/run/capstone-ai-control:ro",
                "--entrypoint", "sh", IMAGE, "-c", 'mkdir -p /var/lib/capstone-ai; chmod 0700 /var/lib/capstone-ai; exec nginx -g "daemon off;"'],
                check=True, stdout=subprocess.DEVNULL)
            address = subprocess.check_output(["docker", "port", name, "8080/tcp"], text=True).strip()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def status():
                try:
                    return opener.open("http://"+address+"/", timeout=2).status
                except urllib.error.HTTPError as e:
                    return e.code
            deadline = time.monotonic()+20
            while True:
                try:
                    assert status()==503
                    break
                except urllib.error.URLError:
                    if time.monotonic()>deadline: raise
                    time.sleep(0.1)
            # Docker Desktop shared-file propagation is asynchronous. Observe the marker
            # from inside the container before checking Nginx, rather than sleeping blindly.
            flag.unlink()
            deadline = time.monotonic()+10
            while subprocess.run(["docker", "exec", name, "test", "-e", "/run/capstone-ai-control/drain"]).returncode == 0:
                if time.monotonic()>deadline:
                    raise AssertionError("drain marker deletion did not reach container")
            actual = status()
            assert actual==200, f"expected reopened 200, received {actual}"
            flag.touch(mode=0o644)
            deadline = time.monotonic()+10
            while subprocess.run(["docker", "exec", name, "test", "-f", "/run/capstone-ai-control/drain"]).returncode != 0:
                if time.monotonic()>deadline:
                    raise AssertionError("drain marker creation did not reach container")
            assert status()==503
            workers = subprocess.check_output(["docker", "exec", name, "ps", "-o", "user,args"],text=True)
            assert any(line.startswith("nginx ") and "nginx: worker" in line for line in workers.splitlines())
            permission = subprocess.check_output(["docker", "exec", name, "stat", "-c", "%a", "/var/lib/capstone-ai"],text=True).strip()
            assert permission=="700"
            print(json.dumps({"worker":"nginx", "secretRootMode":permission, "drainStatuses":[503,200,503]}))
        finally:
            subprocess.run(["docker", "rm", "-f", name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


if __name__=="__main__":
    main()
