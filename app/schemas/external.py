"""POST /external-knowledge (계약 ai-openapi.yaml 의 AiExternalKnowledge*). 필드는 계약 그대로다."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .common import Closed, Context, Version
from .extract import ResponseMeta

Text2000 = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ExternalEvidence(Closed):
    label: Literal["A", "B"]
    keyword: Annotated[str, StringConstraints(max_length=200)]
    text: Text2000


class ExternalClaim(Closed):
    conclusion: Text2000
    evidence: Annotated[list[ExternalEvidence], Field(min_length=1, max_length=4)]


class ExternalKnowledgeRequest(Closed):
    context: Context
    profileVersion: Version
    claim: ExternalClaim
    maxSources: Annotated[int, Field(ge=1, le=3)] = 1


class ExternalKnowledgeItem(Closed):
    summary: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    sourceUrl: Annotated[str, StringConstraints(max_length=2048, pattern=r"^https?://\S+$")]
    title: Annotated[str, StringConstraints(max_length=200)] | None
    retrievedAt: Annotated[str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")]
    support: Literal["yes", "partly"]


class ExternalKnowledgeUsage(Closed):
    searches: Annotated[int, Field(ge=0, le=3)]
    inputTokens: Annotated[int, Field(ge=0)]
    outputTokens: Annotated[int, Field(ge=0)]


class ExternalKnowledgeResponse(Closed):
    context: dict           # 요청 context 를 그대로 돌려준다 (§3.10)
    meta: ResponseMeta
    status: Literal["found", "none"]
    externalKnowledge: Annotated[list[ExternalKnowledgeItem], Field(max_length=3)]
    usage: ExternalKnowledgeUsage | None = None
