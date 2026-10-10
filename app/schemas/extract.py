"""extractKnowledge 요청·응답 (ai-openapi.yaml ExtractRequest / ExtractResponse)."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .common import (Closed, Confidence, Context, EndOffset, NoteVersion, Offset, Quote, Ref, Revision, Sha256,
                     Span, Uuid, Version)


class TopicSnapshot(Closed):
    id: Uuid | None
    name: Annotated[str, StringConstraints(min_length=1, max_length=50)]


class NoteSnapshot(Closed):
    noteId: Uuid
    noteVersion: NoteVersion
    inputRevision: Revision
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    body: Annotated[str, StringConstraints(min_length=0, max_length=50000)]
    bodySha256: Sha256
    topic: TopicSnapshot


class RelationType(Closed):
    code: Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]{0,39}$")]
    label: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    description: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class CompositionRule(Closed):
    firstType: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    secondType: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    resultType: Annotated[str, StringConstraints(min_length=1, max_length=40)]


class Policy(Closed):
    version: Version
    # 계약 스키마는 minItems 1 이지만 문서(§3.5)는 빈 relationTypes 를 422 INVALID_POLICY 로 정한다.
    # 그래서 빈 목록은 스키마(400)가 아니라 의미 검증(422)에서 거절한다.
    relationTypes: Annotated[list[RelationType], Field(max_length=32)]
    compositionRules: Annotated[list[CompositionRule], Field(max_length=128)]


class ExtractRequest(Closed):
    context: Context
    profileVersion: Version
    note: NoteSnapshot
    policy: Policy


class ExtractedConcept(Closed):
    ref: Ref
    label: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    mentions: Annotated[list[Span], Field(min_length=1, max_length=20)]
    confidence: Confidence


class ExtractedEvidence(Closed):
    ref: Ref
    startOffset: Offset
    endOffset: EndOffset
    quote: Quote


class ExtractedRelation(Closed):
    ref: Ref
    sourceRef: Ref
    targetRef: Ref
    typeCode: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    evidenceRefs: Annotated[list[Ref], Field(min_length=1, max_length=8)]
    confidence: Confidence


class ResponseMeta(Closed):
    profileVersion: Version
    provider: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    model: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    modelRevision: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    promptVersion: Version | None
    schemaVersion: Literal["1.0"]
    preprocessingVersion: Version
    durationMs: Annotated[int, Field(ge=0, le=120000)]


class ExtractResponse(Closed):
    context: dict           # 요청 context 를 그대로 돌려준다 (§3.10)
    meta: ResponseMeta
    concepts: Annotated[list[ExtractedConcept], Field(max_length=200)]
    evidence: Annotated[list[ExtractedEvidence], Field(max_length=400)]
    relations: Annotated[list[ExtractedRelation], Field(max_length=300)]
