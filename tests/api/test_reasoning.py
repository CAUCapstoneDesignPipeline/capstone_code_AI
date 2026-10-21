"""병합·검증·설명·노트 연결 operation. 요청은 계약 문서(ai-api.md)의 예시를 그대로 쓴다."""

import copy
import hashlib
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig

TOKEN = "t"
B1 = "# 지식 🧠\n**재시도**는 중복을 만든다.\n"
B2 = "# DB\n중복은 오류를 만든다.\n"
N1, N2 = "10f77e94-6b1f-5e82-b8cd-f6847b0e5479", "4a14ccc1-bfd4-554b-9500-08998d709a36"
E1, E2 = "6cee49e9-0505-5853-9b30-e9254dc26f48", "fead3cae-7388-5d6d-88ea-f68ab0a01a86"
CA, CB, CC = ("42ed80a7-9357-5444-8951-ecf67cc142f9", "2562b589-6615-5349-881c-b52ee1c3ae29",
              "c1c738ee-7d49-52a9-ba0d-6b05a78918fc")
T1, T2 = "27d8c4db-f3fb-5b56-b20f-188c42f8288d", "83c6e69f-a2f1-5540-8071-610f9fbc37a6"
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()

CONTEXT = {"analysisJobId": "d551b3d2-8825-5ae9-8e0e-96c4f63b5021", "stageExecutionId": str(uuid.uuid4()),
           "userId": "c49bdcd3-9f13-5186-93c2-28eccf386140", "noteId": N1, "noteVersion": 3, "inputRevision": 4,
           "graphRevision": 7, "pipelineVersion": "pipeline-fixture-1"}
DOCS = [{"noteId": N1, "noteVersion": 3, "inputRevision": 4, "title": "note-1", "body": B1, "bodySha256": sha(B1),
         "topic": {"id": T1, "name": "네트워크"}},
        {"noteId": N2, "noteVersion": 3, "inputRevision": 4, "title": "note-2", "body": B2, "bodySha256": sha(B2),
         "topic": {"id": T2, "name": "DB"}}]
EVIDENCE = [{"evidenceId": E1, "noteId": N1, "noteVersion": 3, "bodySha256": sha(B1), "startOffset": 8,
             "endOffset": 25, "quote": "**재시도**는 중복을 만든다."},
            {"evidenceId": E2, "noteId": N2, "noteVersion": 3, "bodySha256": sha(B2), "startOffset": 5,
             "endOffset": 17, "quote": "중복은 오류를 만든다."}]
POLICY = {"version": "relations-fixture-1",
          "relationTypes": [{"code": "cause", "label": "원인", "description": "앞 개념이 뒤 개념을 일으킴"},
                            {"code": "indirect_cause", "label": "간접 원인", "description": "두 원인 경로의 합성"}],
          "compositionRules": [{"firstType": "cause", "secondType": "cause", "resultType": "indirect_cause"}]}
PATH = {"pathId": "d89fe9d1-bf21-58da-ad0e-ad52816b7f49",
        "concepts": [{"conceptId": CA, "label": "재시도"}, {"conceptId": CB, "label": "중복"},
                     {"conceptId": CC, "label": "오류"}],
        "sourceConceptId": CA, "bridgeConceptId": CB, "targetConceptId": CC,
        "steps": [{"stepOrder": 1, "relationId": "6819f6d8-0404-5712-939d-52f1bec1ef94", "fromConceptId": CA,
                   "toConceptId": CB, "typeCode": "cause", "evidenceIds": [E1]},
                  {"stepOrder": 2, "relationId": "d91e6535-c622-564f-a95f-68b548cc749b", "fromConceptId": CB,
                   "toConceptId": CC, "typeCode": "cause", "evidenceIds": [E2]}],
        "proposedTypeCode": "indirect_cause"}
PAIR = {"pairId": "pair-1",
        "left": {"conceptId": CB, "label": "중복", "topicId": T1, "evidenceId": E1, "surface": "중복"},
        "right": {"conceptId": "b2f3c213-c677-5e0b-92bb-935a56be07d0", "label": "중복", "topicId": T2,
                  "evidenceId": E2, "surface": "중복"},
        "similarity": 0.92}
STAMP = {"profileVersion": "extract-fixture-1", "provider": "fixture", "model": "extract-model",
         "modelRevision": "fixture-revision-1", "promptVersion": "extract-prompt-1", "schemaVersion": "1.0",
         "preprocessingVersion": "markdown-source-map-1"}
EXPLANATION = {"summary": "이 예시의 두 전제가 성립하면 재시도는 중복을 통해 오류로 이어질 수 있다.",
               "claims": [{"claimRef": "claim-1", "kind": "premise", "text": "재시도는 중복을 만든다.", "evidenceIds": [E1]},
                          {"claimRef": "claim-2", "kind": "premise", "text": "중복은 오류를 만든다.", "evidenceIds": [E2]},
                          {"claimRef": "claim-3", "kind": "conclusion",
                           "text": "예시 합성 규칙 아래 재시도는 오류의 간접 원인이다.", "evidenceIds": [E1, E2]}]}


def base(profile, **extra):
    ctx = dict(CONTEXT, stageExecutionId=str(uuid.uuid4()))
    return {"context": ctx, "profileVersion": profile, "documents": copy.deepcopy(DOCS),
            "evidence": copy.deepcopy(EVIDENCE), **copy.deepcopy(extra)}


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")),
                     extractor=ConceptExtractor(ExtractConfig()))
    with TestClient(app) as c:
        yield c


def post(client, path, body):
    raw = json.dumps(body, ensure_ascii=False).encode()
    return client.post("/internal/v1" + path, content=raw, headers={
        "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4()),
        "X-Attempt-Id": str(uuid.uuid4()), "Idempotency-Key": body["context"]["stageExecutionId"],
        "X-Input-SHA256": hashlib.sha256(raw).hexdigest()})


def code(r):
    return r.json()["error"]["code"]


def test_merge_decisions(client):
    r = post(client, "/merge-decisions", base("merge-det-1", pairs=[PAIR]))
    assert r.status_code == 200, r.text
    d = r.json()
    assert [x["pairId"] for x in d["decisions"]] == ["pair-1"]
    assert d["decisions"][0]["decision"] in ("same", "different", "uncertain") and d["decisions"][0]["reason"]
    assert d["meta"]["profileVersion"] == "merge-det-1" and d["context"]["noteId"] == N1


def test_bridges_need_independent_profile(client):
    ok = post(client, "/verifications/bridges", base("bridges-det-1", pairs=[PAIR], generationProvenance=[STAMP]))
    assert ok.status_code == 200, ok.text
    same = dict(STAMP, model="rules-identity", promptVersion="bridges-rules-1")     # 검증 profile 과 같은 모델+prompt
    r = post(client, "/verifications/bridges", base("bridges-det-1", pairs=[PAIR], generationProvenance=[same]))
    assert r.status_code == 422 and code(r) == "UNSUPPORTED_PROFILE"


def test_verify_edges(client):
    edge = {"edgeId": "6819f6d8-0404-5712-939d-52f1bec1ef94", "source": {"conceptId": CA, "label": "재시도"},
            "target": {"conceptId": CB, "label": "중복"}, "typeCode": "cause", "evidenceIds": [E1]}
    wrong = dict(edge, edgeId=str(uuid.uuid4()), target={"conceptId": CC, "label": "오류"})
    r = post(client, "/verifications/edges", base("edges-det-1", policy=POLICY, edges=[edge, wrong],
                                                  generationProvenance=[STAMP]))
    assert r.status_code == 200, r.text
    out = {v["edgeId"]: v["outcome"] for v in r.json()["verdicts"]}
    assert out[edge["edgeId"]] == "entailed"
    assert out[wrong["edgeId"]] == "unknown"           # 근거에 '오류' 가 없다


def test_verify_contract_example_and_tampered(client):
    ok = post(client, "/verifications/explanations", base("explanation-det-2", path=PATH, policy=POLICY,
                                                           explanation=EXPLANATION, generationProvenance=[STAMP]))
    assert ok.status_code == 200 and ok.json()["overall"] == "pass", ok.text
    bad = copy.deepcopy(EXPLANATION)
    bad["claims"][2]["evidenceIds"] = [E1]                                  # 결론이 한 전제의 근거만 인용
    r = post(client, "/verifications/explanations", base("explanation-det-2", path=PATH, policy=POLICY,
                                                          explanation=bad, generationProvenance=[STAMP]))
    assert r.json()["overall"] == "fail"
    assert next(v for v in r.json()["claimVerdicts"] if v["claimRef"] == "claim-3")["outcome"] == "fail"


def test_no_rule_is_hypothesis(client):
    path = dict(PATH, proposedTypeCode=None)
    policy = dict(POLICY, compositionRules=[])
    v = post(client, "/verifications/explanations", base("explanation-det-2", path=path, policy=policy,
                                                          explanation=EXPLANATION, generationProvenance=[STAMP]))
    assert v.json()["overall"] == "uncertain" and any(g["stepOrder"] is None for g in v.json()["gaps"])
    assert all(x["reason"].endswith("니다.") for x in v.json()["claimVerdicts"])   # 판정 이유는 ~습니다체


def test_template_generation_is_gone(client):
    r = post(client, "/explanations", base("explain-template-1", path=PATH, policy=POLICY))
    assert r.status_code == 422 and code(r) == "UNSUPPORTED_PROFILE"          # 지식 제안은 LLM 으로만


def test_input_errors(client):
    ev = copy.deepcopy(EVIDENCE)
    ev[0]["quote"] = "재시도는 중복을 만든다."                                  # 마크다운을 걷어 낸 인용
    r = post(client, "/explanations", dict(base("explain-llm-6", path=PATH, policy=POLICY), evidence=ev))
    assert r.status_code == 422 and code(r) == "INVALID_SPAN"

    r = post(client, "/explanations", base("explain-llm-6", path=dict(PATH, proposedTypeCode="cause"),
                                           policy=POLICY))
    assert r.status_code == 422 and code(r) == "INVALID_REFERENCE"           # 합성 규칙 결과와 다르다

    same_note = copy.deepcopy(PATH)
    same_note["steps"][1]["evidenceIds"] = [E1]
    r = post(client, "/explanations", base("explain-llm-6", path=same_note, policy=POLICY))
    assert r.status_code == 422                                              # 한 노트 안 경로

    ex = copy.deepcopy(EXPLANATION)
    ex["claims"][0]["evidenceIds"] = []                                      # 근거 없는 전제는 스키마 위반
    r = post(client, "/verifications/explanations", base("explanation-det-2", path=PATH, policy=POLICY,
                                                          explanation=ex, generationProvenance=[STAMP]))
    assert r.status_code == 400

    r = post(client, "/merge-decisions", base("edges-det-1", pairs=[PAIR]))
    assert r.status_code == 422 and code(r) == "UNSUPPORTED_PROFILE"         # 다른 operation 의 profile


NOTES = {
    "a": "# WAL\n\n**WAL**은 Write Ahead Logging이다.\n\n## 로그 버퍼\n\n"
         "**로그 버퍼**는 WAL의 구성 요소다. 로그 버퍼는 메모리에 있다.\n",
    "b": "# 저널링\n\n저널링은 파일 시스템의 변경을 먼저 기록한다. DB의 **WAL**처럼, 저널을 쓴다.\n"
         "WAL은 복구를 위해 쓰인다.\n",
    "c": "# 회귀\n\n선형 회귀는 오차를 줄인다.\n",
}


def test_note_links(client):
    ids = {k: str(uuid.uuid5(uuid.NAMESPACE_URL, k)) for k in NOTES}
    snap = lambda k: {"noteId": ids[k], "noteVersion": 1, "inputRevision": 1, "title": k, "body": NOTES[k],
                      "bodySha256": sha(NOTES[k]), "topic": {"id": None, "name": "미분류"}}
    policy = {"version": "p1", "relationTypes": [{"code": c, "label": c, "description": c} for c in
                                                 ("is_a", "part_of", "used_for", "analogous_to")],
              "compositionRules": [{"firstType": "part_of", "secondType": "used_for", "resultType": "used_for"}]}
    ctx = dict(CONTEXT, stageExecutionId=str(uuid.uuid4()), noteId=ids["a"], noteVersion=1, inputRevision=1)
    body = {"context": ctx, "profileVersion": "links-det-1", "note": snap("a"),
            "candidates": [snap("b"), snap("c")], "policy": policy, "k": 5, "minConfidence": 0.0}
    r = post(client, "/note-links", body)
    assert r.status_code == 200, r.text
    links = r.json()["links"]
    assert [l["noteId"] for l in links] == [ids["b"]]
    assert any(b["label"] == "WAL" for b in links[0]["bridges"])
    text = {ids[k]: v for k, v in NOTES.items()}
    for b in links[0]["bridges"]:
        for e in (b["evidence"], b["candidateEvidence"]):                    # 모든 위치가 원문 UTF-16 slice 와 같다
            u = text[e["noteId"]].encode("utf-16-le")
            assert u[2 * e["startOffset"]:2 * e["endOffset"]].decode("utf-16-le") == e["quote"]
    # 로그 버퍼 —part_of→ WAL (노트 a), WAL —used_for→ 복구 (노트 b), 규칙 part_of + used_for → used_for
    path = next(p for p in links[0]["paths"] if p["source"] == "로그 버퍼")
    assert (path["bridge"], path["target"], path["proposedTypeCode"]) == ("WAL", "복구", "used_for")
    s1, s2 = path["steps"]
    assert (s1["noteId"], s1["typeCode"], s2["noteId"], s2["typeCode"]) == (ids["a"], "part_of", ids["b"], "used_for")
    assert "로그 버퍼는 메모리에 있다." in [e["quote"] for e in s1["context"]]      # 같은 절에서 단계 개념을 말하는 문장
    for e in s1["evidence"] + s2["evidence"] + s1["context"] + s2["context"]:
        u = text[e["noteId"]].encode("utf-16-le")
        assert u[2 * e["startOffset"]:2 * e["endOffset"]].decode("utf-16-le") == e["quote"]


class FakeLLM:
    provider, model = "anthropic", "claude-opus-5-5"

    def __init__(self, out):
        self.out, self.calls = out, []

    def complete_json(self, system, user, schema, name):
        from app.services.llm import LLMResult
        self.calls.append((system, user, schema))
        return LLMResult(self.out, self.provider, self.model)


GOOD = {"summary": "재시도는 중복을 거쳐 오류의 간접 원인이 됩니다.",
        "claims": [{"kind": "premise", "text": "재시도는 중복을 만듭니다.", "evidence": ["E1"]},
                   {"kind": "premise", "text": "중복은 오류를 만듭니다.", "evidence": ["E2"]},
                   {"kind": "condition", "text": "두 노트의 중복이 같은 뜻일 때 성립합니다.", "evidence": ["E1", "E2"]},
                   {"kind": "conclusion", "text": "재시도는 중복을 거쳐 오류의 간접 원인이 됩니다.", "evidence": ["E1", "E2"]}]}


def llm_client(out):
    from app.services.reasoning.explain_llm import ExplainLLM
    fake = FakeLLM(out)
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")),
                     extractor=ConceptExtractor(ExtractConfig()), explainer=ExplainLLM(fake, None))
    return TestClient(app), fake


def test_llm_explanation_is_grounded_and_verified():
    c, fake = llm_client(GOOD)
    with c:
        r = post(c, "/explanations", base("explain-llm-6", path=PATH, policy=POLICY))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["meta"]["provider"] == "anthropic" and body["meta"]["model"] == "claude-opus-5-5"
        concl = next(x for x in body["explanation"]["claims"] if x["kind"] == "conclusion")
        assert set(concl["evidenceIds"]) == {E1, E2}                         # E1/E2 표지가 요청의 evidenceId 로 돌아온다
        system, user, schema = fake.calls[0]
        assert "**재시도**는 중복을 만든다." in user and "indirect_cause" in user  # 근거 원문·합성 결과를 준다
        assert "말투" in system and "보고서·번역투" in system                # 노트 말투를 따르게 한다
        assert schema["properties"]["claims"]["items"]["properties"]["evidence"]["items"]["enum"] == ["E1", "E2"]
        v = post(c, "/verifications/explanations", base("explanation-det-2", path=PATH, policy=POLICY,
                                                         explanation=body["explanation"],
                                                         generationProvenance=[dict(STAMP, **{k: body["meta"][k] for k in (
                                                             "profileVersion", "provider", "model", "modelRevision",
                                                             "promptVersion", "preprocessingVersion")})]))
        assert v.json()["overall"] == "pass", v.text


def test_llm_explanation_breaking_contract_is_502():
    bad = copy.deepcopy(GOOD)
    bad["claims"][3]["evidence"] = ["E1"]                                    # 결론이 단계 2 근거를 인용하지 않음
    c, _ = llm_client(bad)
    with c:
        r = post(c, "/explanations", base("explain-llm-6", path=PATH, policy=POLICY))
        assert r.status_code == 502 and code(r) == "INVALID_MODEL_OUTPUT"


def test_llm_profile_unavailable_without_llm(client):
    r = post(client, "/explanations", base("explain-llm-6", path=PATH, policy=POLICY))
    assert r.status_code == 503 and code(r) == "SERVICE_UNAVAILABLE"


def test_links_hybrid_uses_relation_llm():
    """links-hybrid-1: 규칙이 못 정한 문장 속 쌍을 관계 LLM 이 정하면 그 관계로도 경로를 만든다."""
    from app.services.relations.typer import Typed

    class FakeTyper:
        provider, model = "anthropic", "claude-opus-5-5"

        def __init__(self):
            self.asked = 0

        def type_pairs(self, items, policy_types):
            self.asked += len(items)
            return {it.item_id: Typed("used_for", True, 0.8) for it in items if {it.a, it.b} == {"WAL", "복구"}}

    notes = {"a": NOTES["a"],                                                 # 로그 버퍼 —part_of→ WAL
             "b": "# 복구\n\n**WAL**과 **복구**를 함께 본다. WAL 덕분에 장애 뒤 복구를 한다. 복구는 중요하다.\n"}
    ids = {k: str(uuid.uuid5(uuid.NAMESPACE_URL, "hy" + k)) for k in notes}
    snap = lambda k: {"noteId": ids[k], "noteVersion": 1, "inputRevision": 1, "title": k, "body": notes[k],
                      "bodySha256": sha(notes[k]), "topic": {"id": None, "name": "미분류"}}
    policy = {"version": "p1", "relationTypes": [{"code": c, "label": c, "description": c} for c in ("part_of", "used_for")],
              "compositionRules": [{"firstType": "part_of", "secondType": "used_for", "resultType": "used_for"}]}
    typer = FakeTyper()
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")),
                     extractor=ConceptExtractor(ExtractConfig()), typer=typer)
    with TestClient(app) as c:
        body = lambda prof: {"context": dict(CONTEXT, stageExecutionId=str(uuid.uuid4()), noteId=ids["a"], noteVersion=1,
                                             inputRevision=1), "profileVersion": prof, "note": snap("a"),
                             "candidates": [snap("b")], "policy": policy}
        det = post(c, "/note-links", body("links-det-1")).json()
        assert not any(l["paths"] for l in det["links"])                    # 규칙만으로는 'WAL → 복구' 를 못 찾는다
        hy = post(c, "/note-links", body("links-hybrid-1"))
        assert hy.status_code == 200, hy.text
        assert hy.json()["meta"]["provider"] == "local+anthropic" and typer.asked > 0
        path = hy.json()["links"][0]["paths"][0]
        assert (path["source"], path["bridge"], path["target"], path["proposedTypeCode"]) == ("로그 버퍼", "WAL", "복구", "used_for")
    with TestClient(create_app(Settings(service_token=TOKEN, concept_idf=Path("x")),
                               extractor=ConceptExtractor(ExtractConfig()))) as c:
        r = post(c, "/note-links", body("links-hybrid-1"))
        assert r.status_code == 503                                          # 관계 LLM 이 없으면 준비되지 않음


def edge_client(verdicts):
    from app.services.reasoning.entailment_llm import EdgeJudge
    fake = FakeLLM({"verdicts": verdicts})
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")),
                     extractor=ConceptExtractor(ExtractConfig()), edge_judge=EdgeJudge(fake, None))
    return TestClient(app), fake


EDGE = {"edgeId": "6819f6d8-0404-5712-939d-52f1bec1ef94", "source": {"conceptId": CA, "label": "재시도"},
        "target": {"conceptId": CB, "label": "중복"}, "typeCode": "cause", "evidenceIds": [E1]}


def test_llm_edge_check():
    c, fake = edge_client([{"edge": "X1", "outcome": "entailed", "confidence": 0.9,
                            "reason": "근거가 재시도가 중복을 만든다고 말합니다."}])
    with c:
        r = post(c, "/verifications/edges", base("edges-llm-1", policy=POLICY, edges=[EDGE],
                                                 generationProvenance=[STAMP]))
        assert r.status_code == 200, r.text
        v = r.json()["verdicts"][0]
        assert (v["edgeId"], v["outcome"]) == (EDGE["edgeId"], "entailed") and r.json()["meta"]["model"] == "claude-opus-5-5"
        assert "**재시도**는 중복을 만든다." in fake.calls[0][1] and "습니다" in fake.calls[0][0]
        same = dict(STAMP, model="claude-opus-5-5", promptVersion="edges-llm-1")   # 같은 모델+같은 prompt 로 만든 간선
        r = post(c, "/verifications/edges", base("edges-llm-1", policy=POLICY, edges=[EDGE], generationProvenance=[same]))
        assert r.status_code == 422 and code(r) == "UNSUPPORTED_PROFILE"


def test_llm_edge_check_missing_verdict_is_502():
    c, _ = edge_client([])
    with c:
        r = post(c, "/verifications/edges", base("edges-llm-1", policy=POLICY, edges=[EDGE],
                                                 generationProvenance=[STAMP]))
        assert r.status_code == 502 and code(r) == "INVALID_MODEL_OUTPUT"
