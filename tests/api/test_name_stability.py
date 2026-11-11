"""#18: 같은 입력을 다시 분석하면 개념 이름과 경로 이름이 같다 (결정적 profile).

BE 는 추천 카드 id 를 "사용자 + 근거 노트 쌍 + 경로 개념 이름" 으로 만든다. 그래서 이름이 흔들리면 같은 추천이 새 카드로 보인다.
여기서는 BE 가 실제로 겪는 경우만 본다: 다시 요청(요청마다 stage id 가 다르다), 노트 version 만 오름, 후보를 보내는 순서가 다름,
제목·주제를 바꿈 (BE 는 이것을 분석 입력 변경으로 보지 않는다). LLM 을 쓰는 profile 은 여기서 다루지 않는다 (docs/be-integration.md 참고).
"""

import copy
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from test_reasoning import CONTEXT, NOTES, TOKEN, post, sha

POLICY = {"version": "p1", "relationTypes": [{"code": c, "label": c, "description": c} for c in ("is_a", "part_of", "used_for", "analogous_to")],
          "compositionRules": [{"firstType": "part_of", "secondType": "used_for", "resultType": "used_for"}]}
IDS = {k: str(uuid.uuid5(uuid.NAMESPACE_URL, k)) for k in NOTES}


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("does-not-exist.json")), extractor=ConceptExtractor(ExtractConfig()))
    with TestClient(app) as c:
        yield c


def snap(k, version=1, title=None, topic="미분류"):
    return {"noteId": IDS[k], "noteVersion": version, "inputRevision": version, "title": title or k, "body": NOTES[k], "bodySha256": sha(NOTES[k]),
            "topic": {"id": None, "name": topic}}


def ctx(k, version=1):
    return dict(CONTEXT, stageExecutionId=str(uuid.uuid4()), noteId=IDS[k], noteVersion=version, inputRevision=version)


def extract(client, k, **kw):
    r = post(client, "/extract", {"context": ctx(k, kw.get("version", 1)), "profileVersion": "extract-det-3", "note": snap(k, **kw), "policy": POLICY})
    assert r.status_code == 200, r.text
    d = r.json()
    by = {x["ref"]: x["label"] for x in d["concepts"]}
    return sorted(by.values()), sorted((by[x["sourceRef"]], x["typeCode"], by[x["targetRef"]]) for x in d["relations"])


def links(client, trigger, candidates, **kw):
    r = post(client, "/note-links", {"context": ctx(trigger, kw.get("version", 1)), "profileVersion": "links-det-1", "note": snap(trigger, **kw),
                                     "candidates": [snap(c) for c in candidates], "policy": POLICY, "k": 5, "minConfidence": 0.0})
    assert r.status_code == 200, r.text
    return [(l["noteId"], tuple(b["label"] for b in l["bridges"]),
             tuple((p["source"], p["bridge"], p["target"], p["proposedTypeCode"]) for p in l["paths"])) for l in r.json()["links"]]


def test_concept_and_relation_names_do_not_change_when_the_same_note_is_analysed_again(client):
    for k in ("a", "b"):
        first = extract(client, k)
        assert first[0], "예시 노트에서 개념이 나와야 비교가 의미 있다"
        assert extract(client, k) == first                                        # 다시 요청 (stage id 가 다르다)
        assert extract(client, k, version=2) == first                             # 내용은 그대로, 노트 version·inputRevision 만 오름
        assert extract(client, k, title="이름을 바꾼 노트", topic="다른 주제") == first   # 제목·주제 변경은 분석 결과에 영향이 없다


def test_path_names_do_not_depend_on_request_order_version_title_or_topic(client):
    base = links(client, "a", ["b", "c"])
    paths = [p for _, _, ps in base for p in ps]
    assert ("로그 버퍼", "WAL", "복구", "used_for") in paths, base              # 비어 있지 않다: 이름이 정해진 경로가 실제로 있다
    assert links(client, "a", ["b", "c"]) == base                                 # 다시 요청
    assert links(client, "a", ["c", "b"]) == base                                 # 후보를 보내는 순서
    assert links(client, "a", ["b", "c"], version=7) == base                      # 노트 version 만 오름
    assert links(client, "a", ["b", "c"], title="다른 제목", topic="다른 주제") == base   # 제목·주제 (BE 는 분석 입력 변경으로 보지 않는다)
