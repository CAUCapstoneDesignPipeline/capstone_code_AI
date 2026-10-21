"""병합·설명·검증 operation 요청·응답 (ai-openapi.yaml MergeRequest … VerifyExplanationResponse).

필드 이름·제약은 계약 필드 사전(ai-api.md) 그대로다. 모든 객체는 additionalProperties: false.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .common import Closed, Confidence, Context, EndOffset, NoteVersion, Offset, Quote, Ref, Sha256, Uuid, Version
from .extract import NoteSnapshot, Policy, ResponseMeta

Id = Uuid                                                                   # BE 가 발급하거나 staging 에서 예약한 UUID
Label = Annotated[str, StringConstraints(min_length=1, max_length=200)]
Reason = Annotated[str, StringConstraints(min_length=1, max_length=1000)]
TypeCode = Annotated[str, StringConstraints(min_length=1, max_length=40)]
EvidenceIds = Annotated[list[Id], Field(min_length=1, max_length=8)]
Documents = Annotated[list[NoteSnapshot], Field(min_length=1, max_length=10)]


class GroundedSpan(Closed):
    evidenceId: Id
    noteId: Id
    noteVersion: NoteVersion
    bodySha256: Sha256
    startOffset: Offset
    endOffset: EndOffset
    quote: Quote


Evidence = Annotated[list[GroundedSpan], Field(min_length=1, max_length=40)]


class ModelStamp(Closed):
    profileVersion: Version
    provider: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    model: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    modelRevision: Annotated[str, StringConstraints(min_length=1, max_length=120)]
    promptVersion: Annotated[str, StringConstraints(max_length=80)] | None
    schemaVersion: Literal["1.0"]
    preprocessingVersion: Version


Provenance = Annotated[list[ModelStamp], Field(min_length=1, max_length=8)]


# ---- 개념 동일성 (병합 판단, 매개 개념 검증) ----

class Occurrence(Closed):
    conceptId: Id
    label: Label
    topicId: Id | None
    evidenceId: Id
    surface: Label


class ConceptPair(Closed):
    pairId: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    left: Occurrence
    right: Occurrence
    similarity: Annotated[float, Field(ge=-1, le=1)]


class MergeDecision(Closed):
    pairId: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    decision: Literal["same", "different", "uncertain"]
    confidence: Confidence
    reason: Reason


class MergeRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Documents
    evidence: Evidence
    pairs: Annotated[list[ConceptPair], Field(min_length=1, max_length=32)]


class MergeResponse(Closed):
    context: dict
    meta: ResponseMeta
    decisions: Annotated[list[MergeDecision], Field(min_length=1, max_length=32)]


class VerifyBridgesRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Documents
    evidence: Evidence
    pairs: Annotated[list[ConceptPair], Field(min_length=1, max_length=20)]
    generationProvenance: Provenance


class VerifyBridgesResponse(Closed):
    context: dict
    meta: ResponseMeta
    verdicts: Annotated[list[MergeDecision], Field(min_length=1, max_length=20)]


# ---- A-B-C 경로 ----

class ConceptRef(Closed):
    conceptId: Id
    label: Label


class PathStep(Closed):
    stepOrder: Annotated[int, Field(ge=1, le=2)]
    relationId: Id
    fromConceptId: Id
    toConceptId: Id
    typeCode: TypeCode
    evidenceIds: EvidenceIds


class Path(Closed):
    pathId: Id
    concepts: Annotated[list[ConceptRef], Field(min_length=3, max_length=3)]
    sourceConceptId: Id
    bridgeConceptId: Id
    targetConceptId: Id
    steps: Annotated[list[PathStep], Field(min_length=2, max_length=2)]
    proposedTypeCode: Annotated[str, StringConstraints(max_length=40)] | None


class Claim(Closed):
    claimRef: Ref
    kind: Literal["premise", "condition", "assumption", "conclusion"]
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    evidenceIds: Annotated[list[Id], Field(min_length=0, max_length=8)]

    @model_validator(mode="after")
    def _grounded(self):
        # 계약 Claim.allOf: premise·condition·conclusion 은 근거가 하나 이상. assumption 만 빈 배열을 허용한다
        if self.kind != "assumption" and not self.evidenceIds:
            raise ValueError("evidenceIds")
        return self


class Explanation(Closed):
    summary: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    claims: Annotated[list[Claim], Field(min_length=1, max_length=16)]


class ExplainRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Documents
    evidence: Evidence
    path: Path
    policy: Policy


class ExplainResponse(Closed):
    context: dict
    meta: ResponseMeta
    explanation: Explanation


# ---- 간선 함의 ----

class EdgeCheck(Closed):
    edgeId: Id
    source: ConceptRef
    target: ConceptRef
    typeCode: TypeCode
    evidenceIds: EvidenceIds


class EdgeVerdict(Closed):
    edgeId: Id
    outcome: Literal["entailed", "contradicted", "unknown"]
    confidence: Confidence
    reason: Reason


class VerifyEdgesRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Documents
    evidence: Evidence
    policy: Policy
    edges: Annotated[list[EdgeCheck], Field(min_length=1, max_length=20)]
    generationProvenance: Provenance


class VerifyEdgesResponse(Closed):
    context: dict
    meta: ResponseMeta
    verdicts: Annotated[list[EdgeVerdict], Field(min_length=1, max_length=20)]


# ---- 설명 검증 ----

class ClaimVerdict(Closed):
    claimRef: Ref
    outcome: Literal["pass", "fail", "uncertain"]
    reason: Reason
    evidenceIds: Annotated[list[Id], Field(min_length=0, max_length=8)]


class Gap(Closed):
    stepOrder: Annotated[int, Field(ge=1, le=2)] | None
    reason: Reason


class VerifyExplanationRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Documents
    evidence: Evidence
    path: Path
    policy: Policy
    explanation: Explanation
    generationProvenance: Provenance


class VerifyExplanationResponse(Closed):
    context: dict
    meta: ResponseMeta
    overall: Literal["pass", "fail", "uncertain"]
    claimVerdicts: Annotated[list[ClaimVerdict], Field(min_length=1, max_length=16)]
    gaps: Annotated[list[Gap], Field(min_length=0, max_length=16)]


# ---- 노트 사이 연결 (AI 쪽 제안 operation, 계약 v0.1.0 에 없음) ----

class LinksRequest(Closed):
    context: Context
    profileVersion: Version
    note: NoteSnapshot                                                       # 트리거 노트
    candidates: Annotated[list[NoteSnapshot], Field(min_length=1, max_length=200)]
    policy: Policy
    k: Annotated[int, Field(ge=1, le=30)] = 10
    minConfidence: Annotated[float, Field(ge=0, le=1)] = 0.0


class LinkSpan(Closed):
    noteId: Id
    startOffset: Offset
    endOffset: EndOffset
    quote: Quote


class LinkBridge(Closed):
    label: Label                       # 트리거 노트의 표기
    candidateLabel: Label              # 상대 노트의 표기 (별칭으로 이어졌으면 다르다)
    confidence: Confidence             # 트리거 노트에서의 개념 신뢰도 (extract 와 같은 값)
    candidateConfidence: Confidence
    contribution: Annotated[float, Field(ge=0)]
    evidence: LinkSpan | None          # 트리거 노트에서 그 개념이 나오는 문장
    candidateEvidence: LinkSpan | None


class LinkStep(Closed):
    stepOrder: Annotated[int, Field(ge=1, le=2)]
    noteId: Id
    fromLabel: Label
    toLabel: Label
    typeCode: TypeCode
    evidence: Annotated[list[LinkSpan], Field(min_length=1, max_length=8)]


class LinkPath(Closed):
    source: Label
    bridge: Label
    target: Label
    steps: Annotated[list[LinkStep], Field(min_length=2, max_length=2)]
    proposedTypeCode: TypeCode | None   # policy.compositionRules 결과. 규칙이 없으면 null (가설)


class NoteLink(Closed):
    noteId: Id
    score: Annotated[float, Field(ge=0)]
    bridges: Annotated[list[LinkBridge], Field(min_length=1, max_length=5)]
    paths: Annotated[list[LinkPath], Field(max_length=10)]


class LinksResponse(Closed):
    context: dict
    meta: ResponseMeta
    links: Annotated[list[NoteLink], Field(max_length=30)]
