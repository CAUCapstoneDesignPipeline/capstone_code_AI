"""createEmbeddings 요청·응답 (ai-openapi.yaml EmbeddingsRequest / EmbeddingsResponse)."""

from typing import Annotated

from pydantic import Field, StringConstraints

from .common import Closed, Context, Version
from .extract import ResponseMeta

ItemId = Annotated[str, StringConstraints(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")]


class EmbeddingItem(Closed):
    itemId: ItemId
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class EmbeddingsRequest(Closed):
    context: Context
    profileVersion: Version
    expectedDimensions: Annotated[int, Field(ge=1, le=4096)]
    items: Annotated[list[EmbeddingItem], Field(min_length=1, max_length=32)]


class EmbeddedItem(Closed):
    itemId: Annotated[str, StringConstraints(min_length=1, max_length=80)]
    vector: Annotated[list[float], Field(min_length=1, max_length=4096)]


class EmbeddingsResponse(Closed):
    context: dict           # 요청 context 를 그대로 돌려준다 (§3.10)
    meta: ResponseMeta
    dimensions: Annotated[int, Field(ge=1, le=4096)]
    items: Annotated[list[EmbeddedItem], Field(min_length=1, max_length=32)]
