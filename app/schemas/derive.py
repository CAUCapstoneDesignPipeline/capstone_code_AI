"""계산용 사실 추출(/facts)과 조건부 결론 도출(/derivations) 요청·응답. 계약 v0.1.0 에 없는 AI 쪽 제안 operation 이다.

필드 이름은 계약과 같이 camelCase, 모든 객체는 additionalProperties: false.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.services.derive.facts import CASES, KINDS, OPS, ROLES

from .common import Closed, Context, EndOffset, NoteVersion, Offset, Quote, Sha256, Uuid, Version
from .extract import NoteSnapshot, ResponseMeta
from .reasoning import Provenance

FactId = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9#._:-]{1,80}$")]
Key = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9_]{1,80}$")]
Name = Annotated[str, StringConstraints(min_length=1, max_length=200)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Text = Annotated[str, StringConstraints(min_length=1, max_length=4000)]

Kind = Literal[KINDS]
Role = Literal[tuple(ROLES)]
Op = Literal[tuple(OPS)]
Case = Literal[CASES]


class FactEvidence(Closed):
    noteId: Uuid
    noteVersion: NoteVersion
    bodySha256: Sha256
    startOffset: Offset
    endOffset: EndOffset
    quote: Quote


class Fact(Closed):
    factId: FactId
    kind: Kind
    subject: Name
    subjectKey: Key
    role: Role | None
    op: Op | None
    expr: ShortText | None
    case: Case | None
    negated: bool
    condition: ShortText | None
    other: Name | None
    otherKey: Key | None
    origin: Literal["extracted", "manual"]
    evidence: FactEvidence


class FactsRequest(Closed):
    context: Context
    profileVersion: Version
    note: NoteSnapshot
    knownSubjects: Annotated[list[Key], Field(max_length=500)] = []


class Dropped(Closed):
    reason: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    count: Annotated[int, Field(ge=1)]


class FactsResponse(Closed):
    context: dict
    meta: ResponseMeta
    facts: Annotated[list[Fact], Field(max_length=300)]
    dropped: list[Dropped]


class DeriveRequest(Closed):
    context: Context
    profileVersion: Version
    documents: Annotated[list[NoteSnapshot], Field(min_length=1, max_length=20)]
    facts: Annotated[list[Fact], Field(min_length=1, max_length=400)]
    generationProvenance: Provenance | None = None       # 사실을 뽑은 profile (사람이 적은 사실만이면 없음)


class Subject(Closed):
    key: Key
    label: Name


class Condition(Closed):
    text: Text
    factId: FactId | None


class DeriveStep(Closed):
    text: Text
    factIds: list[FactId]
    expr: ShortText | None


class Soundness(Closed):
    verdict: Literal["sound", "unsound", "uncertain"]
    reason: Text
    unfaithfulFactIds: list[FactId]


class Derivation(Closed):
    derivationId: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    rule: Literal["R1", "R2"]
    ruleVersion: Version
    status: Literal["conclusion", "held", "rejected", "known"]
    reasons: list[Text]
    algorithm: Subject
    structure: Subject
    role: Role
    conclusion: Text
    result: ShortText | None
    case: Case | None
    conditions: list[Condition]
    steps: list[DeriveStep]
    premiseFactIds: list[FactId]
    noteIds: list[Uuid]
    missing: list[ShortText]
    dependsOn: Annotated[str, StringConstraints(max_length=40)] | None
    duplicateOfFactIds: list[FactId]
    automation: Literal["extracted", "manual", "mixed"]
    explanation: Annotated[str, StringConstraints(min_length=1, max_length=8000)]
    soundness: Soundness | None


class UnusedFact(Closed):
    factId: FactId
    reason: Annotated[str, StringConstraints(min_length=1, max_length=40)]


class NoveltyScope(Closed):
    """새로움(이미 쓴 결론인가)을 찾아본 범위: 요청에 온 노트들. 이 밖의 노트는 찾아보지 않았다."""
    noteIds: list[Uuid]


class DeriveResponse(Closed):
    context: dict
    meta: ResponseMeta
    derivations: list[Derivation]
    unusedFacts: list[UnusedFact]
    noveltyScope: NoveltyScope
