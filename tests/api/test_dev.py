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
        assert s["explainedBy"] == {"profile": "explain-llm-8", "provider": "anthropic", "model": "claude-opus-5-5"}


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


def test_keywords_group_concepts_across_pages(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in {**NET_NOTES, "회귀": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n"}.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        every = c.post("/dev/keywords", json={}).json()
        assert every["notes"] == 3 and every["total"] == every["all"] >= 1
        shared = c.post("/dev/keywords", json={"minPages": 2}).json()
        k = next(k for k in shared["keywords"] if k["name"] == "전송 프로토콜")      # 두 노트가 같은 키워드에 개념을 넣었다
        assert k["pages"] == 2 and {x["note"] for x in k["concepts"]} == {"TCP", "전송 계층"}
        assert all(x["origin"] == "extracted" and x["explanation"] and x["conceptId"] for x in k["concepts"])
        assert all(len(x["concepts"]) >= 1 and x["pages"] >= 2 for x in shared["keywords"])
        mine = c.post("/dev/keywords", json={"noteId": "회귀"}).json()               # 이 노트에 개념을 넣은 키워드만
        assert mine["keywords"] and all(any(x["note"] == "회귀" for x in k["concepts"]) for k in mine["keywords"])
        assert c.post("/dev/keywords", json={"q": "전송 프로"}).json()["keywords"][0]["name"] == "전송 프로토콜"
        assert c.post("/dev/keywords", json={"q": "없는말"}).json()["keywords"] == []
        assert c.post("/dev/keywords", json={"noteId": "nope"}).status_code == 404
        assert c.post("/dev/keywords", json={"text": "  "}).status_code == 400


def test_keywords_include_pasted_text(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in NET_NOTES.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        r = c.post("/dev/keywords", json={"text": "**전송 프로토콜**은 신뢰성을 다루는 규약이다. 포트로 프로세스를 구분한다.",
                                          "noteId": None, "minPages": 2}).json()
        assert r["note"] == dev.FREE_ID
        k = next(k for k in r["keywords"] if k["name"] == "전송 프로토콜")
        assert dev.FREE_ID in {x["note"] for x in k["concepts"]} and k["pages"] == 3


def test_keyword_groups_never_merge_unrelated_concepts_of_one_note():
    from app.services.links.keywords import _Groups
    g = _Groups()
    a, b, c = ("n1", "dp"), ("n1", "decision"), ("n2", "dp")
    for x in (a, b, c):
        g.add(x)
    ok = lambda x, y: False                                   # n1 의 두 개념은 서로 별칭이 아니다
    assert g.union(a, c, ok) and g.find(a) == g.find(c)
    assert not g.union(b, c, ok) and g.find(b) != g.find(c)   # c 는 이미 a 와 한 묶음이라 b 와는 합칠 수 없다


def test_extract_keywords_with_descriptions(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in NET_NOTES.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        r = c.post("/dev/extract", json={"noteId": "TCP"}).json()
        by = {k["label"]: k for k in r["keywords"]}
        assert "혼잡 제어" in by and all(0 <= k["confidence"] <= 1 and k["mentions"] for k in r["keywords"])
        body = (tmp_path / "TCP.md").read_text(encoding="utf-8")
        d = by["혼잡 제어"]["description"]
        assert d["text"] and body[d["startOffset"]:d["endOffset"]] == d["text"]          # 설명은 원문 구간이다
        assert "혼잡 제어" in d["text"]
        pasted = c.post("/dev/extract", json={"text": "**WAL**은 쓰기 전에 로그를 먼저 기록하는 방식이다. 체크포인트가 있다."}).json()
        assert pasted["note"] == dev.FREE_ID
        assert next(k for k in pasted["keywords"] if k["label"] == "WAL")["description"]["text"].startswith("**WAL**은")
        assert c.post("/dev/extract", json={"noteId": "nope"}).status_code == 404
        assert c.post("/dev/extract", json={"text": " "}).status_code == 400


def test_keyword_graph_links_keywords_named_in_one_sentence(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in {**NET_NOTES, "회귀": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n"}.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/keyword-graph", json={"noteId": "TCP", "scope": "note"}).json()
        titles = {n["id"]: n["title"] for n in g["nodes"]}
        assert g["mine"] >= 1 and g["mineLinked"] >= 1 and g["edges"]
        for e in g["edges"]:
            assert e["a"] in titles and e["b"] in titles and e["counts"] and e["score"] > 0
            assert all(v["text"] and v["level"] in ("sentence", "heading") and v["note"] in ("TCP", "전송 계층") for v in e["evidence"])
        assert all(n["concepts"] and n["title"] for n in g["nodes"])
        assert any(n["inNote"] for n in g["nodes"])
        pair = {tuple(sorted((titles[e["a"]], titles[e["b"]]))): e for e in g["edges"]}
        tcp_class = pair[("TCP", "전송 프로토콜")]                       # "TCP는 전송 프로토콜의 일종이다." 한 문장에 둘만
        assert "TCP" in tcp_class["evidence"][0]["text"] and "전송 프로토콜" in tcp_class["evidence"][0]["text"]
        assert ("TCP", "혼잡 제어") in pair and ("UDP", "전송 프로토콜") in pair
        crowded = pair[("TCP", "포트 번호")]                            # "TCP와 UDP는 … 포트 번호를 담는다." 키워드 셋이 한 문장 → 선이 약하다
        assert crowded["score"] < tcp_class["score"]
        every = c.post("/dev/keyword-graph", json={"noteId": "TCP", "scope": "all"}).json()
        assert every["scope"] == "all" and all(n["pages"] >= 2 for n in every["nodes"])
        assert c.post("/dev/keyword-graph", json={"scope": "note"}).status_code == 400
        assert c.post("/dev/keyword-graph", json={"noteId": "nope"}).status_code == 404
        assert c.post("/dev/keyword-graph", json={"noteId": "TCP", "scope": "x"}).status_code == 400


def _net(tmp_path, monkeypatch):
    from app.dev import router as dev
    for name, body in NET_NOTES.items():
        (tmp_path / f"{name}.md").write_text(body, encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)


def test_keyword_graph_evidence_levels(tmp_path, monkeypatch):
    _net(tmp_path, monkeypatch)
    with TestClient(dev_app(), client=LOCAL) as c:
        def edges(levels):
            g = c.post("/dev/keyword-graph", json={"noteId": "TCP", "scope": "note", "limit": 30, "levels": levels}).json()
            t = {n["id"]: n["title"] for n in g["nodes"]}
            return {tuple(sorted((t[e["a"]], t[e["b"]]))): e for e in g["edges"]}
        sent, head, para = edges(["sentence"]), edges(["heading"]), edges(["paragraph"])
        assert ("TCP", "전송 프로토콜") in sent and ("TCP", "전송 프로토콜") not in head
        h = head[("패킷", "혼잡 제어")]                                  # "## 혼잡 제어" 절 본문에 패킷이 나온다
        assert h["counts"] == {"heading": 1} and h["evidence"][0]["level"] == "heading" and h["evidence"][0]["text"].startswith("혼잡 제어\n")
        p = para[("패킷", "혼잡 제어")]                                  # 같은 문단, 다른 문장
        assert p["evidence"][0]["level"] == "paragraph" and p["score"] < h["score"] < sent[("TCP", "전송 프로토콜")]["score"]
        assert not set(sent) & set(para) - {("TCP", "전송 프로토콜")}    # 같은 문장으로 이어진 쌍은 문단 증거를 더하지 않는다 (다른 노트의 문단은 예외)
        both = edges(["sentence", "heading", "paragraph"])
        assert set(both) >= set(sent) | set(head) | set(para)
        assert c.post("/dev/keyword-graph", json={"noteId": "TCP", "levels": ["x"]}).status_code == 400
        assert c.post("/dev/keyword-graph", json={"noteId": "TCP", "levels": []}).status_code == 400


def test_keyword_connect_direct_and_paths(tmp_path, monkeypatch):
    _net(tmp_path, monkeypatch)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/keyword-graph", json={"noteId": "TCP", "scope": "note", "limit": 30}).json()
        ids = {n["title"]: n["id"] for n in g["nodes"]}
        r = c.post("/dev/keyword-connect", json={"a": ids["TCP"], "b": ids["전송 프로토콜"]}).json()
        assert r["direct"] and r["direct"]["evidence"][0]["level"] == "sentence" and r["a"]["name"] == "TCP"
        r = c.post("/dev/keyword-connect", json={"a": ids["TCP"], "b": ids["네트워크 프로토콜"]}).json()
        assert r["direct"] is None                                      # 한 문장에 같이 나온 적이 없다
        two = [[n["name"] for n in p["nodes"]] for p in r["paths"] if p["hops"] == 2]
        assert ["TCP", "전송 프로토콜", "네트워크 프로토콜"] in two          # TCP → 전송 프로토콜 → 네트워크 프로토콜
        path = next(p for p in r["paths"] if p["hops"] == 2)
        assert len(path["edges"]) == 2 and all(e["evidence"] for e in path["edges"])
        assert c.post("/dev/keyword-connect", json={"a": ids["TCP"], "b": ids["TCP"]}).status_code == 404
        assert c.post("/dev/keyword-connect", json={"a": ids["TCP"], "b": "nope"}).status_code == 404


def test_keyword_relation_endpoint(tmp_path, monkeypatch):
    from app.services.links.relation_llm import KeywordRelationJudge
    from app.services.llm import LLMResult

    class Fake:
        provider, model = "anthropic", "fake-model"
        def __init__(self, data): self.data, self.calls = data, 0
        def complete_json(self, system, user, schema, name):
            self.calls += 1
            return LLMResult(self.data, self.provider, self.model)

    body = {"a": {"name": "TCP", "concepts": ["TCP는 전송 프로토콜의 일종이다."]}, "b": {"name": "전송 프로토콜", "concepts": []},
            "evidence": [{"level": "sentence", "note": "TCP", "text": "TCP는 전송 프로토콜의 일종이다."},
                         {"level": "path", "label": "TCP — UDP", "note": "전송 계층", "text": "TCP와 UDP는 모두 패킷 헤더에 포트 번호를 담는다."}]}
    app = dev_app()
    fake = Fake({"verdict": "related", "type": "is_a", "direction": "a_to_b", "reason": "TCP 는 전송 프로토콜의 한 종류다.", "evidence": ["E1"]})
    app.state.kwrel = KeywordRelationJudge(fake, tmp_path / "cache")
    with TestClient(app, client=LOCAL) as c:
        r = c.post("/dev/keyword-relation", json=body).json()
        assert (r["verdict"], r["type"], r["direction"], r["evidence"], r["cached"]) == ("related", "is_a", "a_to_b", ["E1"], False)
        assert r["model"] == "fake-model"
        assert c.post("/dev/keyword-relation", json=body).json()["cached"] is True and fake.calls == 1
        assert c.post("/dev/keyword-relation", json={**body, "evidence": []}).status_code == 400
        bad = {**body, "evidence": [{"level": "x", "note": "n", "text": "t"}]}
        assert c.post("/dev/keyword-relation", json=bad).status_code == 400
    with TestClient(app, client=("10.0.0.5", 50000)) as c:
        assert c.post("/dev/keyword-relation", json=body).status_code == 403          # 노트 내용이 밖으로 나가므로 로컬만
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for k in ("AI_LLM_PROVIDER", "AI_KEYWORD_RELATION_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        assert c.post("/dev/keyword-relation", json=body).status_code == 503


def _no_embeddings(monkeypatch):
    from app.services.links.pairing import EmbeddingStore
    monkeypatch.setattr(EmbeddingStore, "vectors", lambda self, texts: None)        # 모델 파일이 없어도 돌아야 한다


def test_proposal_candidates_pair_elements_of_different_notes(tmp_path, monkeypatch):
    from app.dev import router as dev
    (tmp_path / "DB-WAL.md").write_text("# WAL\n\n**WAL**은 **트랜잭션**이 커밋되기 전에 변경 내용을 로그로 디스크에 먼저 기록하는 방식이다. "
                                        "**체크포인트**는 장애가 났을 때 복구 시간을 줄이려고 쓰는 기록 지점이다.\n", encoding="utf-8")
    (tmp_path / "ACID.md").write_text("# ACID\n\n**원자성**은 **트랜잭션**이 전부 반영되거나 전혀 반영되지 않는 성질이라는 뜻이다. "
                                      "**복구**는 장애 뒤에 **트랜잭션** 로그를 다시 읽어 데이터를 되돌리는 일이다.\n", encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    _no_embeddings(monkeypatch)
    with TestClient(dev_app(), client=LOCAL) as c:
        r = c.post("/dev/proposals/candidates", json={"noteId": "DB-WAL", "limit": 20}).json()
        assert r["note"] == "DB-WAL" and r["anchors"] >= 1 and r["embeddings"] is False and r["cosineFloor"] is None
        assert r["candidates"], r["channels"]
        for x in r["candidates"]:
            assert x["a"]["note"] == "DB-WAL" and x["b"]["note"] == "ACID" and x["a"]["keywordId"] != x["b"]["keywordId"]
            assert x["channels"] == ["pivot"] and x["via"] == ["트랜잭션"] and x["a"]["text"] and x["b"]["conceptId"] and x["id"]
        assert len({x["id"] for x in r["candidates"]}) == len(r["candidates"])
        assert len({(x["a"]["text"], x["b"]["text"]) for x in r["candidates"]}) == len(r["candidates"])
        assert c.post("/dev/proposals/candidates", json={"noteId": "nope"}).status_code == 404
        assert c.post("/dev/proposals/candidates", json={}).status_code == 400
        t = c.post("/dev/proposals/candidates", json={"text": "**TCP**는 **트랜잭션** 없이 신뢰성을 위해 재전송을 하는 프로토콜이다."}).json()
        assert t["note"] == "(직접 입력)"


def test_proposal_screen_and_generate_endpoints(tmp_path, monkeypatch):
    from app.services.links.generate import KnowledgeGenerator
    from app.services.links.pair_screen import PairScreen
    from app.services.llm import LLMResult
    from app.services.llm.anthropic_llm import WebResult
    N1, N2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"

    class Fake:
        provider, model = "anthropic", "fake"
        def complete_json(self, system, user, schema, name):
            ids = [l[4:] for l in user.splitlines() if l.startswith("id: ")]
            return LLMResult({"results": [{"id": i, "label": "useful", "conclusion": "결론"} for i in ids]}, self.provider, self.model)
        def complete_web(self, system, user, schema, name, max_searches=3):
            return WebResult({"status": "ok", "newKeyword": "새 키워드", "fromA": "로그를 먼저", "fromB": "전부 반영",
                              "conclusion": "로그를 먼저 쓰기 때문에 전부 반영되거나 전혀 반영되지 않는 성질을 지킬 수 있어요.",
                              "external": [{"url": "https://example.com/p", "title": "T", "quote": "q", "supports": "s"}], "reason": "r"},
                             "anchored", "fake", [{"url": "https://example.com/p", "title": "T"}], 1)

    side = lambda k, note, nid, cid, text: {"keyword": k, "note": note, "noteId": nid, "conceptId": cid, "text": text, "start": 0, "end": len(text), "quote": text}
    a = side("WAL", "DB-WAL", N1, "33333333-3333-4333-8333-333333333333", "WAL은 로그를 먼저 디스크에 쓴다.")
    b = side("Atomicity", "ACID", N2, "44444444-4444-4444-8444-444444444444", "트랜잭션은 전부 반영되거나 전혀 반영되지 않는다.")
    app = dev_app()
    app.state.pair_screen = PairScreen(Fake(), None)
    app.state.knowledge_generator = KnowledgeGenerator(Fake(), None)
    with TestClient(app, client=LOCAL) as c:
        s = c.post("/dev/proposals/screen", json={"pairs": [{"id": "p1", "a": {"keyword": "WAL", "text": a["text"]}, "b": {"keyword": "Atomicity", "text": b["text"]}}]}).json()
        assert s["results"]["p1"]["label"] == "useful" and s["promptVersion"] == "pairscreen-llm-1"
        g = c.post("/dev/proposals/generate", json={"a": a, "b": b}).json()
        assert g["status"] == "candidate" and g["concept"]["origin"] == "generated" and g["external"][0]["url"] == "https://example.com/p"
        assert c.post("/dev/proposals/generate", json={"a": a, "b": {**b, "note": "DB-WAL"}}).status_code == 400
        assert c.post("/dev/proposals/screen", json={"pairs": []}).status_code == 400
    with TestClient(app, client=("10.0.0.5", 50000)) as c:                                 # 노트 내용이 밖으로 나가므로 로컬만
        assert c.post("/dev/proposals/generate", json={"a": a, "b": b}).status_code == 403
        assert c.post("/dev/proposals/screen", json={"pairs": [{"id": "p", "a": {"keyword": "x", "text": "y"}, "b": {"keyword": "x", "text": "y"}}]}).status_code == 403
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for k in ("AI_LLM_PROVIDER", "AI_SUGGEST_LLM_PROVIDER", "AI_PAIR_SCREEN_LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        assert c.post("/dev/proposals/generate", json={"a": a, "b": b}).status_code == 503


def test_keyword_connect_returns_the_element_pair_to_make_new_knowledge_from(tmp_path, monkeypatch):
    from app.dev import router as dev
    (tmp_path / "DB-WAL.md").write_text("# WAL\n\n**WAL**은 **트랜잭션**이 커밋되기 전에 변경 내용을 로그로 디스크에 먼저 기록하는 방식이다. "
                                        "**체크포인트**는 장애가 났을 때 복구 시간을 줄이려고 쓰는 기록 지점이다.\n", encoding="utf-8")
    (tmp_path / "ACID.md").write_text("# ACID\n\n**원자성**은 **트랜잭션**이 전부 반영되거나 전혀 반영되지 않는 성질이라는 뜻이다. "
                                      "**복구**는 장애 뒤에 **트랜잭션** 로그를 다시 읽어 데이터를 되돌리는 일이다.\n", encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/keyword-graph", json={"noteId": "DB-WAL", "scope": "note", "limit": 30, "levels": ["sentence", "heading", "paragraph"]}).json()
        ids = {n["title"]: n["id"] for n in g["nodes"]}
        r = c.post("/dev/keyword-connect", json={"a": ids["WAL"], "b": ids["복구"], "levels": ["sentence", "heading", "paragraph"]}).json()
        p = r["pair"]
        assert p["a"]["keyword"] == "WAL" and p["b"]["keyword"] == "복구" and p["a"]["note"] == "DB-WAL" and p["b"]["note"] == "ACID"
        assert p["a"]["conceptId"] and p["b"]["noteId"] and p["a"]["text"] and p["b"]["end"] > p["b"]["start"]


def _sse(text: str) -> list[dict]:
    import json
    out = []
    for block in text.strip().split("\n\n"):
        data = [l[6:] for l in block.split("\n") if l.startswith("data: ")]
        if data:
            out.append(json.loads(data[0]))
    return out


def test_proposal_generate_stream_reports_progress_then_the_result(tmp_path):
    from app.services.links.generate import KnowledgeGenerator
    from app.services.llm.anthropic_llm import WebResult
    N1, N2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"

    class Fake:
        provider, model = "anthropic", "fake"
        def complete_web(self, system, user, schema, name, max_searches=3, tool=None, on_event=None):
            for ev in ({"type": "thinking"}, {"type": "search_start"}, {"type": "search", "query": "write ahead log os page cache"},
                       {"type": "search_results", "results": [{"url": "https://example.com/p", "title": "T"}]}, {"type": "writing", "chars": 0}):
                on_event(ev)
            return WebResult({"status": "ok", "newKeyword": "새 키워드", "fromA": "로그를 먼저", "fromB": "전부 반영",
                              "conclusion": "로그를 먼저 쓰기 때문에 전부 반영되거나 전혀 반영되지 않는 성질을 지킬 수 있어요.",
                              "external": [{"url": "https://example.com/p", "title": "T", "quote": "q", "supports": "s"}], "reason": "r"},
                             "anthropic", "fake", [{"url": "https://example.com/p", "title": "T"}], 1)

    side = lambda k, note, nid, cid, text: {"keyword": k, "note": note, "noteId": nid, "conceptId": cid, "text": text, "start": 0, "end": len(text), "quote": text}
    a = side("WAL", "DB-WAL", N1, "33333333-3333-4333-8333-333333333333", "WAL은 로그를 먼저 디스크에 쓴다.")
    b = side("Atomicity", "ACID", N2, "44444444-4444-4444-8444-444444444444", "트랜잭션은 전부 반영되거나 전혀 반영되지 않는다.")
    app = dev_app()
    app.state.knowledge_generator = KnowledgeGenerator(Fake(), tmp_path)
    with TestClient(app, client=LOCAL) as c:
        r = c.post("/dev/proposals/generate/stream", json={"a": a, "b": b})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream") and r.headers["cache-control"] == "no-cache"
        evs = _sse(r.text)
        assert [e["type"] for e in evs] == ["start", "thinking", "search_start", "search", "search_results", "writing", "validating", "result"]
        assert evs[3]["query"] == "write ahead log os page cache" and evs[4]["results"][0]["url"] == "https://example.com/p"
        assert all("t" in e for e in evs) and evs[0]["model"] == "fake"
        assert evs[-1]["result"]["status"] == "candidate" and evs[-1]["result"]["concept"]["origin"] == "generated"
        again = _sse(c.post("/dev/proposals/generate/stream", json={"a": a, "b": b}).text)      # 저장된 결과는 바로 돌려준다
        assert [e["type"] for e in again] == ["cached", "result"] and again[-1]["result"]["cached"] is True
        assert c.post("/dev/proposals/generate/stream", json={"a": a, "b": {**b, "note": "DB-WAL"}}).status_code == 400
    with TestClient(app, client=("10.0.0.5", 50000)) as c:
        assert c.post("/dev/proposals/generate/stream", json={"a": a, "b": b}).status_code == 403


def test_proposal_generate_stream_turns_model_errors_into_an_error_event():
    from app.services.links.generate import KnowledgeGenerator
    from app.services.llm import LLMError
    N1, N2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"

    class Boom:
        provider, model = "anthropic", "fake"
        def complete_web(self, *a, **k):
            raise LLMError("rate_limit", "x")

    s = lambda note, nid, cid: {"keyword": "k", "note": note, "noteId": nid, "conceptId": cid, "text": "문장 하나 둘 셋", "start": 0, "end": 5, "quote": "문장"}
    app = dev_app()
    app.state.knowledge_generator = KnowledgeGenerator(Boom(), None)
    with TestClient(app, client=LOCAL) as c:
        evs = _sse(c.post("/dev/proposals/generate/stream", json={"a": s("A", N1, "c1"), "b": s("B", N2, "c2")}).text)
        assert [e["type"] for e in evs] == ["start", "error"] and "rate_limit" in evs[-1]["message"] and evs[-1]["status"] == 502


def _chain_notes(tmp_path, monkeypatch):
    from app.dev import router as dev
    (tmp_path / "DB-WAL.md").write_text("# WAL\n\n**WAL**은 **트랜잭션** 로그를 디스크에 먼저 기록해서 장애에 대비하는 방식이다.\n", encoding="utf-8")
    (tmp_path / "ACID.md").write_text("# ACID\n\n**트랜잭션**은 **원자성**을 지켜야 하는 작업 단위라는 뜻이다.\n", encoding="utf-8")
    (tmp_path / "ISO.md").write_text("# 격리\n\n**원자성**은 **격리성**과 함께 ACID 를 이루는 성질이라고 부른다.\n", encoding="utf-8")
    monkeypatch.setattr(dev, "NOTES_DIR", tmp_path)
    monkeypatch.delenv("AI_NOTES_DIR", raising=False)
    monkeypatch.setattr(dev, "INFER_MIN_SCORE", -99.0)               # 작은 시험 그래프라 문턱을 없앤다
    _no_embeddings(monkeypatch)


def test_inferred_links_join_keywords_that_never_share_a_sentence_or_paragraph(tmp_path, monkeypatch):
    _chain_notes(tmp_path, monkeypatch)
    with TestClient(dev_app(), client=LOCAL) as c:
        base = c.post("/dev/keyword-graph", json={"noteId": "DB-WAL", "scope": "note", "limit": 30, "levels": ["sentence"]}).json()
        t = {n["id"]: n["title"] for n in base["nodes"]}
        assert not any("inferred" in e["counts"] for e in base["edges"]) and base["inferredEdges"] == 0 and base["semantic"] is None
        g = c.post("/dev/keyword-graph", json={"noteId": "DB-WAL", "scope": "note", "limit": 30, "levels": ["sentence", "inferred"]}).json()
        t = {n["id"]: n["title"] for n in g["nodes"]}
        inf = [e for e in g["edges"] if "inferred" in e["counts"]]
        assert g["inferredEdges"] == len(inf) >= 1 and g["semantic"] is False
        e = next(e for e in inf if sorted((t[e["a"]], t[e["b"]])) == ["WAL", "원자성"])      # 같은 문장·문단에 나온 적 없다
        assert e["via"] == ["트랜잭션"] and e["evidence"][0]["level"] == "inferred" and "공통 이웃 1개: 트랜잭션" in e["evidence"][0]["text"]
        assert 0.1 <= e["score"] <= 0.5 and e["pages"] == 0
        far = next(n for n in g["nodes"] if n["title"] == "격리성")                           # 3걸음 거리라 증거 이웃으로는 안 올라오고 추정으로만 올라온다
        assert far["inferred"] is True and far["inNote"] is False and not next(n for n in g["nodes"] if n["title"] == "원자성")["inferred"]
        assert any(sorted((t[x["a"]], t[x["b"]])) == ["WAL", "격리성"] and "inferred" in x["counts"] for x in g["edges"])
        assert not any(sorted((t[x["a"]], t[x["b"]])) == ["WAL", "트랜잭션"] and "inferred" in x["counts"] for x in g["edges"])   # 이미 이어진 쌍은 아니다
        assert c.post("/dev/keyword-graph", json={"noteId": "DB-WAL", "levels": ["inferred"]}).status_code == 400   # 증거 수준이 하나는 있어야 한다


def test_connect_shows_the_inferred_score_only_for_pairs_without_evidence(tmp_path, monkeypatch):
    _chain_notes(tmp_path, monkeypatch)
    with TestClient(dev_app(), client=LOCAL) as c:
        g = c.post("/dev/keyword-graph", json={"noteId": "DB-WAL", "scope": "note", "limit": 30, "levels": ["sentence", "inferred"]}).json()
        ids = {n["title"]: n["id"] for n in g["nodes"]}
        r = c.post("/dev/keyword-connect", json={"a": ids["WAL"], "b": ids["원자성"], "levels": ["sentence", "inferred"]}).json()
        assert r["direct"] is None and [v["name"] for v in r["inferred"]["via"]] == ["트랜잭션"] and r["inferred"]["semantic"] is False
        assert any([n["name"] for n in p["nodes"]] == ["WAL", "트랜잭션", "원자성"] for p in r["paths"])
        linked = c.post("/dev/keyword-connect", json={"a": ids["WAL"], "b": ids["트랜잭션"], "levels": ["sentence", "inferred"]}).json()
        assert linked["direct"] and linked["inferred"] is None                               # 증거로 이어진 쌍은 추정하지 않는다
        off = c.post("/dev/keyword-connect", json={"a": ids["WAL"], "b": ids["원자성"], "levels": ["sentence"]}).json()
        assert off["inferred"] is None
