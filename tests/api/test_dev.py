"""개발용 playground 엔드포인트 (`ai serve --dev`)."""

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig

LOCAL = ("127.0.0.1", 50000)


def dev_app():
    s = Settings(service_token="t", concept_idf=Path("does-not-exist.json"), dev_ui=True)
    return create_app(s, extractor=ConceptExtractor(ExtractConfig()))


def test_dev_routes_absent_without_flag():
    s = Settings(service_token="t", concept_idf=Path("does-not-exist.json"))
    with TestClient(create_app(s), client=LOCAL) as c:
        assert c.get("/dev/").status_code == 404 and c.get("/dev/config").status_code == 404


def test_pick_llm_and_clear():
    with TestClient(dev_app(), client=LOCAL) as c:
        assert c.get("/dev/config").json()["llm"] is None
        r = c.post("/dev/llm", json={"provider": "google", "apiKey": "secret-key-123"})
        assert r.status_code == 200 and r.json()["llm"] == {"provider": "google", "model": "gemini-3.5-flash"}
        cfg = c.get("/dev/config")
        assert cfg.json()["llm"]["provider"] == "google"
        assert "secret-key-123" not in cfg.text and "secret-key-123" not in r.text      # 키는 돌려주지 않는다
        assert c.delete("/dev/llm").json() == {"llm": None}
        assert c.get("/dev/llm").json() == {"llm": None}


def test_pick_llm_validation():
    with TestClient(dev_app(), client=LOCAL) as c:
        assert c.post("/dev/llm", json={"provider": "openai", "apiKey": "k"}).status_code == 400   # 모델 필수
        assert c.post("/dev/llm", json={"provider": "mistral", "apiKey": "k"}).status_code == 400
        r = c.post("/dev/llm", json={"provider": "openai", "model": "some-model", "apiKey": "k"})
        assert r.status_code == 200 and r.json()["llm"]["model"] == "some-model"


def test_pick_llm_only_from_loopback():
    with TestClient(dev_app(), client=("10.0.0.5", 50000)) as c:
        assert c.post("/dev/llm", json={"provider": "google", "apiKey": "k"}).status_code == 403
        assert c.delete("/dev/llm").status_code == 403
        assert c.get("/dev/config").json()["token"] is None      # 로컬이 아니면 서비스 토큰도 채워 주지 않는다
