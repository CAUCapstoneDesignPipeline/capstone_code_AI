"""/facts · /derivations operation 과 참조 클라이언트 흐름 (derive_insights)."""

import asyncio
import copy
import uuid
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.client import AiClient, derive_insights, snapshot
from app.core.config import Settings
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.derive.cases import load_note, manual_facts
from app.services.derive.extract_llm import FactExtractor
from app.services.derive.sound_llm import SoundnessJudge
from app.services.llm import LLMResult

from test_reasoning import TOKEN, code, post

USER = "c49bdcd3-9f13-5186-93c2-28eccf386140"


def doc(name):
    n = load_note(name)
    return snapshot(n.note_id, n.title, n.body)


def fact_json(f):
    e = f.evidence
    return {"factId": f.fact_id, "kind": f.kind, "subject": f.subject, "subjectKey": f.subject_key, "role": f.role,
            "op": f.op, "expr": f.expr, "case": f.case, "negated": f.negated, "condition": f.condition,
            "other": f.other, "otherKey": f.other_key, "origin": f.origin,
            "evidence": {"noteId": e.note_id, "noteVersion": e.note_version, "bodySha256": e.body_sha256,
                         "startOffset": e.start, "endOffset": e.end, "quote": e.quote}}


def ctx(note):
    return {"analysisJobId": str(uuid.uuid4()), "stageExecutionId": str(uuid.uuid4()), "userId": USER,
            "noteId": note["noteId"], "noteVersion": note["noteVersion"], "inputRevision": note["inputRevision"],
            "graphRevision": 1, "pipelineVersion": "test-1"}


def derive_body(names, profile="derive-rules-1", **extra):
    docs = [doc(n) for n in names]
    facts = [fact_json(f) for n in names for f in manual_facts(load_note(n))]
    return {"context": ctx(docs[0]), "profileVersion": profile, "documents": docs, "facts": facts, **extra}


class FakeLLM:
    provider, model = "anthropic", "claude-opus-5-5"

    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def complete_json(self, system, user, schema, name):
        self.calls += 1
        return LLMResult(self.fn(user), self.provider, self.model)


def manual_as_llm(user):
    """노트 원문을 보고 그 노트의 사람이 적은 사실을 LLM 출력 모양으로 돌려준다."""
    for name in ("dijkstra", "indexed-heap", "bfs", "queue-linked-list", "stack"):
        n = load_note(name)
        if n.body in user:
            return {"facts": [{"kind": f.kind, "subject": f.subject, "subjectKey": f.subject_key, "role": f.role or "none",
                               "op": f.op or "none", "expr": f.expr or "", "case": f.case or "none", "negated": f.negated,
                               "condition": f.condition or "", "other": f.other or "", "otherKey": f.other_key or "",
                               "quote": f.evidence.quote} for f in manual_facts(n)]}
    return {"facts": []}


def all_sound(user):
    refs = sorted({line.split("]")[0][1:] for line in user.splitlines() if line.startswith("[D")})
    return {"judgments": [{"derivation": r, "unfaithful": [], "verdict": "sound", "reason": "맞습니다."} for r in refs]}


def make_client(extractor=True, judge=True):
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")),
                     extractor=ConceptExtractor(ExtractConfig()),
                     fact_extractor=FactExtractor(FakeLLM(manual_as_llm)) if extractor else None,
                     soundness_judge=SoundnessJudge(FakeLLM(all_sound)) if judge else None)
    return app


@pytest.fixture(scope="module")
def client():
    with TestClient(make_client()) as c:
        yield c


def test_derivations_rules(client):
    r = post(client, "/derivations", derive_body(["dijkstra", "indexed-heap"]))
    assert r.status_code == 200, r.text
    d = r.json()
    r2 = next(x for x in d["derivations"] if x["rule"] == "R2")
    assert r2["status"] == "conclusion" and r2["result"] == "O((V + E) log V)" and r2["automation"] == "manual"
    assert r2["dependsOn"] == next(x for x in d["derivations"] if x["rule"] == "R1")["derivationId"]
    assert len(r2["noteIds"]) == 2 and r2["soundness"] is None and d["meta"]["profileVersion"] == "derive-rules-1"
    assert sorted(d["noveltyScope"]["noteIds"]) == sorted(x["noteId"] for x in derive_body(["dijkstra", "indexed-heap"])["documents"])


def test_derivations_reject_stale_or_moved_evidence(client):
    body = derive_body(["dijkstra", "indexed-heap"])
    stale = copy.deepcopy(body)
    stale["facts"][0]["evidence"]["noteVersion"] = 0                      # 옛 리비전에서 뽑은 사실
    r = post(client, "/derivations", stale)
    assert r.status_code == 422 and code(r) == "INVALID_REFERENCE"
    moved = copy.deepcopy(body)
    moved["facts"][0]["evidence"]["startOffset"] += 1                     # 위치와 quote 가 맞지 않음
    r = post(client, "/derivations", moved)
    assert r.status_code == 422 and code(r) == "INVALID_SPAN"
    dup = copy.deepcopy(body)
    dup["facts"][1]["factId"] = dup["facts"][0]["factId"]
    assert post(client, "/derivations", dup).status_code == 422


def test_derivations_sound_profile_and_independence(client):
    r = post(client, "/derivations", derive_body(["dijkstra", "indexed-heap"], "derive-sound-1"))
    assert r.status_code == 200, r.text
    concl = [x for x in r.json()["derivations"] if x["status"] == "conclusion"]
    assert concl and all(x["soundness"]["verdict"] == "sound" for x in concl)
    same = {"profileVersion": "facts-x", "provider": "anthropic", "model": "claude-opus-5-5", "modelRevision": "r",
            "promptVersion": "derive-sound-1", "schemaVersion": "1.0", "preprocessingVersion": "p"}
    r = post(client, "/derivations", derive_body(["dijkstra", "indexed-heap"], "derive-sound-1",
                                                  generationProvenance=[same]))
    assert r.status_code == 422 and code(r) == "UNSUPPORTED_PROFILE"


def test_sound_profile_unavailable_without_judge():
    with TestClient(make_client(judge=False)) as c:
        r = post(c, "/derivations", derive_body(["dijkstra", "indexed-heap"], "derive-sound-1"))
        assert r.status_code == 503
        assert post(c, "/derivations", derive_body(["dijkstra", "indexed-heap"])).status_code == 200


def test_facts(client):
    note = doc("indexed-heap")
    r = post(client, "/facts", {"context": ctx(note), "profileVersion": "facts-llm-1", "note": note,
                                "knownSubjects": ["dijkstra"]})
    assert r.status_code == 200, r.text
    fs = r.json()["facts"]
    assert {f["op"] for f in fs if f["kind"] == "op_cost"} == {"insert", "extract_min", "decrease_key"}
    for f in fs:
        e = f["evidence"]
        assert note["body"].encode("utf-16-le")[2 * e["startOffset"]:2 * e["endOffset"]].decode("utf-16-le") == e["quote"]
    other = dict(ctx(note), noteId=str(uuid.uuid4()))
    r = post(client, "/facts", {"context": other, "profileVersion": "facts-llm-1", "note": note})
    assert r.status_code == 422


def test_derive_insights_end_to_end():
    app = make_client()

    async def go():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ai") as http:
                c = AiClient("http://ai", TOKEN, http)
                notes = [doc(n) for n in ("dijkstra", "indexed-heap", "bfs", "queue-linked-list", "stack")]
                cache = {}
                res = await derive_insights(c, notes[0], notes[1:], cache)
                again = await derive_insights(c, notes[0], notes[1:], cache)
                return res, again, app.state.fact_extractor.llm.calls
    res, again, calls = asyncio.run(go())
    assert [i["rule"] for i in res["insights"]] == ["R2"]                # 다익스트라가 낸 결론만, R2 카드 하나
    assert res["insights"][0]["result"] == "O((V + E) log V)"
    assert res["notes"][0] == doc("dijkstra")["noteId"] and doc("indexed-heap")["noteId"] in res["notes"]
    assert again["insights"] == res["insights"] and calls == 5           # 사실은 노트마다 한 번만 뽑는다
