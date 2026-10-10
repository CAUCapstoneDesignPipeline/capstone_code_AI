"""계약(ai-openapi.yaml) 공통 타입. 필드 이름은 계약 그대로 camelCase 다.

모든 객체는 additionalProperties: false 이므로 extra="forbid", 타입 변환 없이 strict 로 검증한다.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Uuid = Annotated[str, StringConstraints(
    pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")]
Sha256 = Annotated[str, StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Ref = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Version = Annotated[str, StringConstraints(min_length=1, max_length=80)]
Revision = Annotated[int, Field(ge=0, le=9007199254740991)]
NoteVersion = Annotated[int, Field(ge=0, le=2147483647)]
Offset = Annotated[int, Field(ge=0, le=100000)]
EndOffset = Annotated[int, Field(ge=1, le=100000)]
Quote = Annotated[str, StringConstraints(min_length=1, max_length=50000)]
Confidence = Annotated[float, Field(ge=0, le=1)]


class Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Context(Closed):
    analysisJobId: Uuid
    stageExecutionId: Uuid
    userId: Uuid
    noteId: Uuid
    noteVersion: NoteVersion
    inputRevision: Revision
    graphRevision: Revision
    pipelineVersion: Version


class Span(Closed):
    startOffset: Offset
    endOffset: EndOffset
    quote: Quote
