from pathlib import Path

from fastapi.testclient import TestClient


def test_production_runtime_disables_private_response_caches_and_dev(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_SERVICE_TOKEN", "synthetic-runtime-token")
    monkeypatch.setenv("AI_CONCEPT_IDF", str(tmp_path / "missing-idf.json"))
    monkeypatch.setenv("AI_RELATION_LLM", "0")
    monkeypatch.setenv("AI_EXPLAIN_LLM", "0")
    monkeypatch.setenv("AI_VERIFY_LLM", "0")
    monkeypatch.setenv("AI_DEV_UI", "1")
    monkeypatch.setenv("AI_EMBEDDING_PROFILES", "")
    monkeypatch.setenv("AI_DRAIN_FILE", str(tmp_path / "drain"))
    from app.gcp_runtime import production_app
    app = production_app()
    settings = app.state.settings
    assert settings.concept_cache is settings.relation_cache is settings.explain_cache is settings.verify_cache is None
    assert settings.dev_ui is False
    with TestClient(app) as client:
        assert client.get("/dev/").status_code == 404
        headers={"Authorization":"Bearer synthetic-runtime-token", "X-Request-Id":"7dbe037d-3358-45cb-9f0f-61d24ea10311"}
        assert client.get("/internal/v1/health/live", headers=headers).status_code == 200
        assert client.get("/internal/v1/health/live", headers={**headers,"Authorization":"Bearer wrong-token"}).status_code == 401
        (tmp_path / "drain").touch()
        assert client.post("/internal/v1/facts", json={}).status_code == 503
        assert client.get("/internal/v1/health/ready").status_code == 503
        (tmp_path / "drain").unlink()
        # Still unavailable without IDF, never false-success after drain.
        assert client.get("/internal/v1/health/ready", headers={"Authorization":"Bearer synthetic-runtime-token",
            "X-Request-Id":"7dbe037d-3358-45cb-9f0f-61d24ea10311"}).status_code == 503
