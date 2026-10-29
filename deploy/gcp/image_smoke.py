"""CPU-only production image smoke. Synthetic input; no paid provider calls."""
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from app.core.config import EMBEDDING_PROFILES, EXTRACT_PROFILES, EXTRACT_PROFILE
from app.services.concepts import ConceptExtractor
from app.services.concepts.extract import build_idf


def main():
    assert platform.machine() in {"x86_64", "AMD64"}, "amd64 production smoke required"
    assert os.getuid() == 10001, "nonroot image required"
    profile = EMBEDDING_PROFILES["embeddings-bge-m3-1"]
    assert profile.revision == "5617a9f61b028005a4858fdac845db406aefb181"
    assert importlib.util.find_spec("app.dev") is None
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        extractor = ConceptExtractor(EXTRACT_PROFILES[EXTRACT_PROFILE].concepts)
        idf = build_idf(extractor, {"synthetic-a": "# 큐\n큐는 먼저 넣은 값을 먼저 꺼내는 자료구조다.",
                                    "synthetic-b": "# FIFO\nFIFO는 입력 순서대로 값을 꺼낸다."})
        idf.save(root / "idf.json", extractor.config_version)
        del extractor
        token = uuid.uuid4().hex + uuid.uuid4().hex
        env = {**os.environ, "AI_CONCEPT_IDF": str(root / "idf.json"), "AI_SERVICE_TOKEN": token,
               "AI_EMBEDDING_PROFILES": profile.id, "AI_RELATION_LLM": "0", "AI_VERIFY_LLM": "0",
               "AI_EXPLAIN_LLM": "0", "AI_DEV_UI": "0", "AI_MAX_INFLIGHT": "2"}
        # Suppress process output: SDK diagnostics may contain provider configuration.
        process = subprocess.Popen(["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
                                    "--workers", "1", "--no-access-log"], env=env,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        def call(path, payload=None, bearer=token):
            raw = json.dumps(payload).encode() if payload else None
            headers = {"Authorization": f"Bearer {bearer}", "X-Request-Id": str(uuid.uuid4())}
            if raw:
                headers.update({"Content-Type": "application/json", "X-Attempt-Id": str(uuid.uuid4()),
                                "Idempotency-Key": payload["context"]["stageExecutionId"],
                                "X-Input-SHA256": hashlib.sha256(raw).hexdigest()})
            try:
                with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000" + path,
                                             raw, headers), timeout=35) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, {}
        try:
            started = time.monotonic()
            while time.monotonic() - started < 900:
                assert process.poll() is None, "AI process exited during model load"
                try:
                    status, ready = call("/internal/v1/health/ready")
                    if status == 200:
                        break
                except urllib.error.URLError:
                    pass
                time.sleep(2)
            else:
                raise AssertionError("model/readiness timeout")
            assert profile.id in ready["profileVersions"]
            assert call("/internal/v1/health/live", bearer="wrong-token")[0] == 401
            assert call("/dev/")[0] == 404
            context = {k: str(uuid.uuid4()) for k in ["analysisJobId", "stageExecutionId", "userId", "noteId"]}
            context.update(noteVersion=1, inputRevision=1, graphRevision=1, pipelineVersion="pipeline-smoke-1")
            status, result = call("/internal/v1/embeddings", {"context": context,
                "profileVersion": profile.id, "expectedDimensions": 1024,
                "items": [{"itemId": "synthetic-queue", "text": "큐는 FIFO 자료구조다."}]})
            assert status == 200
            vector = result["items"][0]["vector"]
            assert len(vector) == 1024 and all(math.isfinite(x) for x in vector)
            print(json.dumps({"architecture": platform.machine(), "uid": os.getuid(),
                              "profile": profile.id, "modelRevision": profile.revision,
                              "readiness": True, "embeddingDimensions": len(vector),
                              "providerCalled": False, "coldReadySeconds": round(time.monotonic()-started, 2)}))
        finally:
            process.terminate()
            try:
                process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
