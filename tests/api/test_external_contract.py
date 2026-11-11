"""외부 지식의 실제 응답이 내부 계약(docs 저장소 api/ai-openapi.yaml 의 /external-knowledge)과 같은가. 계약 파일이 없으면 건너뛴다.

계약 파일 위치: 환경 변수 AI_CONTRACT_YAML, 없으면 이 저장소 옆의 docs/ · capstone_docs/. 스키마 검사기는 챗봇 계약 테스트의 것을 쓴다.
"""

import copy

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("yaml")

from test_chat_contract import SPEC, conforms, ref
from test_external_api import ITEM, ITEM2, PROFILE, FakeFinder, body, err, make, post

from app.services.links.external import Found, safe_url
from app.services.llm import LLMError

pytestmark = pytest.mark.skipif(not SPEC or "/external-knowledge" not in SPEC["paths"], reason="계약 파일(api/ai-openapi.yaml 의 /external-knowledge)이 없다")

OP = SPEC["paths"]["/external-knowledge"]["post"] if SPEC and "/external-knowledge" in SPEC["paths"] else None
REQ_EX = OP["requestBody"]["content"]["application/json"]["examples"]["normal"]["value"] if OP else None
RESP = OP["responses"]["200"]["content"]["application/json"] if OP else None


def found(*items, n=1):
    return Found("found" if items else "none", [dict(i) for i in items], n, 4300, 1100, "anthropic", "claude-sonnet-5-5")


def own(req=None):
    return {**copy.deepcopy(req or REQ_EX), "profileVersion": PROFILE}        # 계약 예시의 profile 은 fixture 이름이다


def test_operation_id_and_headers_match_the_contract():
    from pathlib import Path
    from app.core.config import Settings
    from app.main import create_app
    ai = create_app(Settings(service_token="t", concept_idf=Path("none.json"))).openapi()["paths"]["/internal/v1/external-knowledge"]["post"]
    assert ai["operationId"] == OP["operationId"] == "findExternalKnowledge"
    names = {ref(p["$ref"])["name"] for p in OP["parameters"]}
    assert {"X-Request-Id", "X-Attempt-Id", "Idempotency-Key", "X-Input-SHA256"} <= names


def test_the_contracts_own_examples_match_the_schema_and_the_request_is_accepted():
    conforms(OP["requestBody"]["content"]["application/json"]["schema"], REQ_EX)
    for name, ex in RESP["examples"].items():
        conforms(RESP["schema"], ex["value"], name)
    with TestClient(make(FakeFinder(found(ITEM)))) as c:
        assert post(c, own()).status_code == 200


@pytest.mark.parametrize("finder, max_sources", [(found(ITEM), 1), (found(ITEM, ITEM2), 3), (found(n=1), 1)])
def test_real_responses_conform_to_the_response_schema_and_the_semantic_rules(finder, max_sources):
    with TestClient(make(FakeFinder(finder))) as c:
        r = post(c, {**own(), "maxSources": max_sources})
    assert r.status_code == 200
    d = r.json()
    conforms(RESP["schema"], d)
    assert d["context"] == REQ_EX["context"]
    items, urls = d["externalKnowledge"], [i["sourceUrl"] for i in d["externalKnowledge"]]
    assert (d["status"] == "found") == bool(items) and len(items) <= max_sources and len(urls) == len(set(urls))
    assert all(safe_url(u) for u in urls)
    ranks = [0 if i["support"] == "yes" else 1 for i in items]
    assert ranks == sorted(ranks)
    assert set(d["meta"]) == set(ref("#/components/schemas/ResponseMeta")["properties"])


@pytest.mark.parametrize("change", [lambda r: r.update(extra=1), lambda r: r["claim"].update(conclusion=""), lambda r: r["claim"]["evidence"][0].update(label="C"),
                                    lambda r: r.update(maxSources=4), lambda r: r.update(maxSources=0), lambda r: r["claim"].update(evidence=[])])
def test_requests_the_contract_forbids_are_rejected_with_400(change):
    bad = own()
    change(bad)
    conforms_bad = False
    try:
        conforms(OP["requestBody"]["content"]["application/json"]["schema"], bad)
    except AssertionError:
        conforms_bad = True
    assert conforms_bad                                                        # 스키마도 이 요청을 거부한다
    with TestClient(make(FakeFinder())) as c:
        r = post(c, bad)
    assert r.status_code == 400 and err(r) == "VALIDATION_FAILED"


def test_every_status_the_endpoint_can_return_is_declared_by_the_contract():
    declared = set(OP["responses"])
    seen = {"200"}
    for finder, make_kw in ((FakeFinder(found(ITEM)), {}), (FakeFinder(error=LLMError("timeout", "x")), {}), (FakeFinder(error=LLMError("provider", "x")), {}),
                            (FakeFinder(error=LLMError("rate_limit", "x")), {}), (FakeFinder(error=LLMError("auth", "x")), {}), (None, {})):
        with TestClient(make(finder, **make_kw)) as c:
            seen.add(str(post(c, own()).status_code))
            seen.add(str(post(c, {**own(), "profileVersion": "nope"}).status_code))
            seen.add(str(post(c, {**own(), "extra": 1}).status_code))
            seen.add(str(post(c, own(), **{"X-Input-SHA256": "0" * 64}).status_code))
            seen.add(str(post(c, own(), Authorization=None).status_code))
    assert seen <= declared, f"계약에 없는 상태 {seen - declared}"
