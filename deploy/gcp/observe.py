#!/usr/bin/env python3
"""Emit only bounded readiness/TLS metadata; never provider requests or SDK diagnostics."""
import datetime
import json
import os
import socket
import ssl
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path


def observe(config=Path("/etc/capstone-ai/host.json"), nginx=Path("/etc/nginx/conf.d/capstone-ai.conf"),
            control=Path("/run/capstone-ai-control")):
    started = time.monotonic()
    result = {"component":"capstone-ai", "state":"degraded", "tlsExpiring":None}
    try:
        if (control/"drain").exists():
            result["state"]="maintenance"
            return result
        target=nginx.resolve(strict=True).parent
        target.relative_to(Path("/var/lib/capstone-ai/releases"))
        manifest=json.loads((target/"manifest.json").read_bytes())
        host=json.loads(config.read_bytes())
        check=subprocess.run(["openssl","x509","-in",str(target/"leaf.pem"),"-checkend","1209600","-noout"],
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
        result["tlsExpiring"]=check.returncode != 0
        request=urllib.request.Request("http://127.0.0.1:8000/internal/v1/health/ready",headers={
            "Authorization":"Bearer "+(target/"service-token").read_text(),"X-Request-Id":str(uuid.uuid4())})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request,timeout=5) as response:
            ready=json.load(response)
        if ready.get("status")!="ready" or not set(manifest["profiles"])<=set(ready.get("profileVersions",[])):
            return result
        context=ssl.create_default_context(cafile=str(target/"ca.pem"))
        with socket.create_connection((host["privateIp"],8443),timeout=5) as raw:
            with context.wrap_socket(raw,server_hostname=manifest["tls"]["serverName"]):
                pass
        if not result["tlsExpiring"]:
            result["state"]="ready"
        return result
    except Exception:
        return result
    finally:
        result["latencyMs"]=round((time.monotonic()-started)*1000)
        result["timestamp"]=datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    result=observe()
    path=Path("/var/log/capstone-ai/health.jsonl")
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    with os.fdopen(fd,"w") as output:
        output.write(json.dumps(result)+"\n")
    print(json.dumps(result))
    return 0 if result["state"] in {"ready","maintenance"} else 1


if __name__=="__main__":
    raise SystemExit(main())
