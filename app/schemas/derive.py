"""계산용 사실 추출(/facts)과 조건부 결론 도출(/derivations) 요청·응답. 계약 v0.1.0 에 없는 AI 쪽 제안 operation 이다.

필드 이름은 계약과 같이 camelCase, 모든 객체는 additionalProperties: false.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.services.derive.facts import CASES, KEY_RE, KINDS

from .common import Closed, Context, EndOffset, NoteVersion, Offset, Quote, Sha256, Uuid, Version
from .extract import NoteSnapshot, ResponseMeta
from .reasoning import Provenance

FactId = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9#._:-]{1,80}$")]
Key = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[a-z0-9_]{1,80}$")]
Name = Annotated[str, StringConstraints(min_length=1, max_length=200)]
ShortText = Annotated[str, StringConstraints(min_length=1, max_length=300)]
Text = Annotated[str, StringConstraints(min_length=1, max_length=4000)]

Kind = Literal[KINDS]
Role = Annotated[str, StringConstraints(pattern=f"^{KEY_RE}$")]       # 열린 key (자주 쓰는 것은 facts.ROLES·OPS)
Op = Annotated[str, StringConstraints(pattern=f"^{KEY_RE}$")]
Label = Annotated[str, StringConstraints(min_length=1, max_length=80)]
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
    roleLabel: Label | None
    opLabel: Label | None
    expr: ShortText | None
    case: Case | None
    negated: bool
    condition: ShortText | None
    other: Name | None
    otherKey: Key | None
    origin: Literal["extracted", "manual"]
    evidence: FactEvidence
    contentHash: Sha256 | None = None      # 사실의 뜻 해시 (응답에는 항상). 요청에 있으면 AI 가 다시 계산해 맞는지 본다


class FactsRequest(Closed):
    context: Context
    profileVersion: Version
    note: NoteSnapshot
    knownSubjects: Annotated[list[Key], Field(max_length=500)] = []
    knownRoles: Annotated[list[Key], Field(max_length=200)] = []
    knownOps: Annotated[list[Key], Field(max_length=200)] = []


class Dropped(Closed):
    reason: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    count: Annotated[int, Field(ge=1)]


class FactsResponse(Closed):
    context: dict
    meta: ResponseMeta
    facts: Annotated[list[Fact], Field(max_length=300)]
    dropped: list[Dropped]
    factsHash: Sha256                      # 이 노트의 사실 전체의 뜻 해시. 노트를 고쳐도 같으면 사실이 같다 (결론은 그대로 유효)


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
    roleLabel: Label
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
    fingerprint: Sha256                    # 같은 추천인가 (알고리즘 × 자료 구조 × 역할). BE 가 사용자를 붙여 카드 id 를 이어 간다
    basisHash: Sha256                      # 같은 지문의 내용이 바뀌었는가 (기댄 사실의 뜻 + 규칙 + 경우)


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
