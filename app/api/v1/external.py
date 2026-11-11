"""외부 지식 operation (계약 ai-openapi.yaml, 설명은 docs/external-knowledge.md).

  POST /external-knowledge   findExternalKnowledge   추천 결론을 뒷받침하는 외부 자료(한 문장 요약 · 출처 URL) 찾기

웹 검색을 한 번 하고 호출당 15~20초가 걸리므로 다른 LLM operation 과 같은 한 자리(inflight)를 쓴다. 기본은 꺼져 있다 (AI_EXTERNAL_LLM=1 로 켠다).
뒷받침하는 출처가 없으면 오류가 아니라 status=none 이다. 응답은 저장하지 않는다.
"""

from fastapi import APIRouter, Request

from app.core.errors import AiError
from app.schemas.external import ExternalKnowledgeItem, ExternalKnowledgeRequest, ExternalKnowledgeResponse, ExternalKnowledgeUsage
from app.services.links.external import safe_url

from .reasoning import build, meta, prepare, run

router = APIRouter()


@router.post("/external-knowledge", response_model=ExternalKnowledgeResponse, operation_id="findExternalKnowledge")
async def external_knowledge(request: Request):
    t0, data, req, profile = await prepare(request, ExternalKnowledgeRequest, "external")
    claim = req.claim.model_dump()
    found = await run(request, profile, lambda: request.app.state.external_finder.find(claim, req.maxSources))
    items = [build(ExternalKnowledgeItem, **it) for it in found.items]
    # 자체 검증 (§3.10): found 이면 1~maxSources 개, none 이면 없음, URL 중복·안전하지 않은 URL 없음, yes 가 partly 보다 앞
    urls = [i.sourceUrl for i in items]
    ranks = [0 if i.support == "yes" else 1 for i in items]
    if (found.status == "found") != bool(items) or len(items) > req.maxSources or len(urls) != len(set(urls)) \
            or not all(safe_url(u) for u in urls) or ranks != sorted(ranks):
        raise AiError("INVALID_MODEL_OUTPUT")
    usage = build(ExternalKnowledgeUsage, searches=found.searches, inputTokens=found.input_tokens, outputTokens=found.output_tokens)
    return build(ExternalKnowledgeResponse, context=data["context"], meta=meta(profile, t0, found.provider, found.model),
                 status=found.status, externalKnowledge=items, usage=usage)
