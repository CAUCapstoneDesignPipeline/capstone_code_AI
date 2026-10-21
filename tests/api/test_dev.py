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


def test_links_with_pasted_text(tmp_path, monkeypatch):
    from app.dev import router as dev
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "db-wal.md").write_text("# WAL\n\n**WAL**은 쓰기 전에 로그를 먼저 기록한다. 체크포인트가 있다.\n", encoding="utf-8")
    (notes / "ml.md").write_text("# 회귀\n\n선형 회귀는 오차를 줄인다.\n", encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", notes)
    with TestClient(dev_app(), client=LOCAL) as c:
        assert c.get("/dev/index").json()["notes"] == 2
        r = c.post("/dev/links", json={"text": "저널링은 DB 의 **WAL** 처럼 먼저 기록한다. 체크포인트도 쓴다."}).json()
        assert r["note"] == dev.FREE_ID and r["links"][0]["note"] == "db-wal"
        assert any(b["labelA"] == "WAL" and b["confidenceA"] is not None for b in r["links"][0]["bridges"])
        strict = c.post("/dev/links", json={"text": "저널링은 DB 의 **WAL** 처럼 먼저 기록한다.", "minConfidence": 1.0}).json()
        assert strict["links"] == []                                  # 신뢰도 1 이상인 개념은 없다
        assert c.get("/dev/links/db-wal").status_code == 200
        assert c.post("/dev/links", json={"noteId": "nope"}).status_code == 404



NET_NOTES = {"TCP": '# TCP\n\n**TCP**는 **전송 프로토콜**의 일종이다. TCP는 연결 지향 방식으로 데이터를 보낸다.\n\n## 혼잡 제어\n\nTCP는 **혼잡 제어**를 한다. 혼잡 제어는 패킷 손실을 줄인다. TCP는 재전송으로 신뢰성을 보장한다.\n**패킷**은 헤더와 페이로드로 이루어진다. 포트 번호는 패킷 헤더에 있다.\n',
             "전송 계층": '# 전송 계층\n\n**전송 프로토콜**은 **네트워크 프로토콜**의 일종이다. 전송 프로토콜은 포트 번호로 프로세스를 구분한다.\n\n## UDP\n\n**UDP**는 전송 프로토콜의 일종이다. UDP는 혼잡 제어를 하지 않는다. UDP는 패킷 손실을 허용한다.\nTCP와 UDP는 모두 패킷 헤더에 포트 번호를 담는다. 재전송은 TCP에만 있다.\n'}


def test_suggest_new_knowledge_end_to_end(tmp_path, monkeypatch):
    """처음 보는 노트 두 개: TCP is_a 전송 프로토콜 (노트 1) + 전송 프로토콜 is_a 네트워크 프로토콜 (노트 2)
    → 'TCP 는 네트워크 프로토콜의 한 종류' 제안. 설명은 LLM(여기선 가짜)이 쓰고, 내부 API 를 BE 처럼 차례로 불러
    모든 검증을 통과해야 한다."""
    from app.dev import router as dev
    for name, body in NET_NOTES.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    import re

    from app.services.llm import LLMResult
    from app.services.reasoning.explain_llm import ExplainLLM

    class StepCitingLLM:                    # 단계별 근거 표지를 프롬프트에서 읽어 인용하는 가짜 LLM
        provider, model = "anthropic", "claude-opus-5-5"

        def complete_json(self, system, user, schema, name):
            first, second = user.split("단계 2:")
            e1, e2 = re.findall(r"\[(E\d+)\]", first)[0], re.findall(r"\[(E\d+)\]", second)[0]
            return LLMResult({"summary": "TCP는 네트워크 프로토콜의 한 종류입니다.", "claims": [
                {"kind": "premise", "text": "TCP는 전송 프로토콜의 한 종류입니다.", "evidence": [e1]},
                {"kind": "premise", "text": "전송 프로토콜은 네트워크 프로토콜의 한 종류입니다.", "evidence": [e2]},
                {"kind": "condition", "text": "두 노트의 전송 프로토콜이 같은 뜻일 때 성립합니다.", "evidence": [e1, e2]},
                {"kind": "conclusion", "text": "TCP는 네트워크 프로토콜의 한 종류입니다.", "evidence": [e1, e2]}]},
                self.provider, self.model)

    s = Settings(service_token="t", concept_idf=Path("does-not-exist.json"), dev_ui=True)
    app = create_app(s, extractor=ConceptExtractor(ExtractConfig()), explainer=ExplainLLM(StepCitingLLM(), None))
    with TestClient(dev_app(), client=LOCAL) as c:                         # LLM 이 없으면 지식 제안을 하지 않는다
        assert c.post("/dev/suggest", json={"noteId": "TCP"}).status_code == 503
    with TestClient(app, client=LOCAL) as c:
        r = c.post("/dev/suggest", json={"noteId": "TCP"})
        assert r.status_code == 200, r.text
        d = r.json()
        assert [l["note"] for l in d["links"]] == ["전송 계층"]
        s = next(x for x in d["suggestions"] if (x["source"], x["target"]) == ("TCP", "네트워크 프로토콜"))
        assert s["proposedTypeCode"] == "is_a" and s["status"] == "candidate", s["statusReasons"]
        assert s["bridgeVerdict"]["decision"] == "same"
        assert [v["outcome"] for v in s["edgeVerdicts"]] == ["entailed", "entailed"]
        assert s["explanationVerdict"]["overall"] == "pass"
        assert s["knowledge"] == "TCP는 네트워크 프로토콜의 한 종류입니다."
        assert s["explainedBy"] == {"profile": "explain-llm-6", "provider": "anthropic", "model": "claude-opus-5-5"}


def test_graph_shows_notes_and_their_connections(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in {**NET_NOTES, "UDP": "# UDP\n\n**UDP**는 **전송 프로토콜**의 일종이다. UDP는 혼잡 제어를 하지 않는다.\n"
                                           "UDP는 패킷 손실을 허용한다. 포트 번호로 프로세스를 구분한다.\n",
                       "회귀": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n"}.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/graph", json={"noteId": "TCP"}).json()
        ids = [n["id"] for n in g["nodes"]]
        assert ids[0] == "TCP" and "회귀" not in ids and {"전송 계층", "UDP"} <= set(ids)
        pairs = {frozenset((e["a"], e["b"])) for e in g["edges"]}
        assert frozenset(("전송 계층", "UDP")) in pairs                  # 가운데가 아닌 노트끼리의 선도 있다
        e = next(e for e in g["edges"] if {e["a"], e["b"]} == {"TCP", "전송 계층"})
        assert e["bridges"] and all(b["quoteA"] and b["quoteB"] for b in e["bridges"])
        chain = next(p for p in e["paths"] if p["rule"] == "is_a")      # TCP → 전송 프로토콜 → 네트워크 프로토콜
        assert [s["type"] for s in chain["steps"]] == ["is_a", "is_a"] and all(s["quote"] for s in chain["steps"])


def test_whole_collection_graph_and_edge_detail(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in {**NET_NOTES, "회귀": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n"}.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/graph/all", json={"k": 3}).json()
        assert {n["id"] for n in g["nodes"]} == {"TCP", "전송 계층", "회귀"}           # 이어지지 않은 노트도 점으로
        assert [(e["a"], e["b"]) for e in g["edges"]] == [("TCP", "전송 계층")]
        d = c.post("/dev/graph/edge", json={"a": "TCP", "b": "전송 계층"}).json()
        assert d["bridges"][0]["quoteA"] and any(p["rule"] == "is_a" for p in d["paths"])
        assert c.post("/dev/graph/edge", json={"a": "TCP", "b": "회귀"}).status_code == 404
        assert c.post("/dev/graph", json={"noteId": "TCP"}).status_code == 200      # 기존 그래프도 그대로
