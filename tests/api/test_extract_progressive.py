"""참조 클라이언트의 두 단계 개념 추출: 빠른 결과가 먼저 오고, LLM 결과가 나중에 오며, LLM 이 실패해도 빠른 결과는 남는다."""

import asyncio
import time
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient  # noqa: F401  (앱 수명 주기를 쓰기 위해 같은 환경을 쓴다)

from app.client import AiCallError, AiClient, extract_progressive, snapshot, swap_concepts
from app.main import create_app
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.llm_concepts import ConceptLLM
from app.services.llm import LLMError
from test_api import TOKEN, settings

BODY = "# 데이터베이스 로그\n\n**WAL**은 로그를 먼저 쓰는 규칙이다. WAL 덕분에 복구할 수 있다. 체크포인트는 복구 시간을 줄인다. **체크포인트** 이후의 로그만 적용한다.\n"
POLICY = {"version": "relations-fixture-1", "relationTypes": [{"code": "cause", "label": "원인", "description": "예시"}], "compositionRules": []}


class SlowLLM:
    provider, model = "anthropic", "fake"

    def __init__(self, data=None, error=None, delay=0.4):
        self.data, self.error, self.delay = data, error, delay

    def complete_json(self, system, user, schema, name):
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return SimpleNamespace(data=self.data)


async def run(llm):
    app = create_app(settings(), extractor=ConceptExtractor(ExtractConfig(), Idf({}, 10, "t")), concept_llm=ConceptLLM(llm))
    out = []
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ai", timeout=30) as http:
            client = AiClient("http://ai", TOKEN, http)
            t0 = time.monotonic()
            async for stage, value in extract_progressive(client, snapshot("11111111-1111-4111-8111-111111111111", "log", BODY), POLICY):
                out.append((stage, value, time.monotonic() - t0))
    return out


def test_the_fast_result_comes_first_and_the_llm_result_replaces_it_later():
    out = asyncio.run(run(SlowLLM({"concepts": [{"surface": "WAL", "aliases": []}, {"surface": "체크포인트", "aliases": []}]})))
    assert [s for s, _, _ in out] == ["fast", "refined"]
    fast, refined = out[0][1], out[1][1]
    assert fast["meta"]["profileVersion"] == "extract-student-1" and refined["meta"]["profileVersion"] == "extract-llm-1"
    assert out[0][2] < out[1][2]                                              # 빠른 결과가 먼저 (LLM 은 0.4초 이상 걸렸다)
    swap = swap_concepts(fast, refined)
    assert set(swap["kept"]) | set(swap["added"]) == {c["label"] for c in refined["concepts"]}
    assert set(swap["kept"]) | set(swap["removed"]) == {c["label"] for c in fast["concepts"]}


def test_when_the_llm_fails_the_caller_still_has_the_fast_result():
    out = asyncio.run(run(SlowLLM(error=LLMError("refusal", "x"), delay=0.0)))
    assert [s for s, _, _ in out] == ["fast", "refine_failed"]
    assert isinstance(out[1][1], AiCallError) and out[1][1].status == 502 and out[1][1].error["code"] == "INVALID_MODEL_OUTPUT"


def test_swap_concepts_ignores_case_spaces_and_hyphens():
    fast = {"concepts": [{"label": "Write-Ahead Log"}, {"label": "체크포인트"}, {"label": "잡음"}]}
    refined = {"concepts": [{"label": "write ahead log"}, {"label": "체크포인트"}, {"label": "복구"}]}
    assert swap_concepts(fast, refined) == {"kept": ["write ahead log", "체크포인트"], "added": ["복구"], "removed": ["잡음"]}
