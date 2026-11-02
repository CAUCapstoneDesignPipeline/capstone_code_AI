"""지식 그래프의 키워드(노드)와 개념 (회의록 2026-10-30 지식 그래프 설계, docs/knowledge-graph.md).

키워드 하나가 노드 하나이고, 키워드 안에 개념(설명 1개 + 소스)이 여러 개 들어 있다. 페이지는 노트(noteId)다.
회의에서 정하지 않은 것은 필드로 두지 않았다: 개념의 `etc`, 인터넷 URL 을 소스에 넣을지 (그래서 `external` 을 따로 둔다).
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .common import Closed, NoteVersion, Quote, Span, Uuid

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
# 설명은 임베딩에도 들어가므로 /embeddings 의 text 한도와 같게 둔다.
Explanation = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Url = Annotated[str, StringConstraints(pattern=r"^https?://\S+$", max_length=2048)]
Timestamp = Annotated[str, StringConstraints(
    pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")]


class PageSource(Closed):
    """개념이 나온 페이지(노트)와 그 안의 구간."""
    noteId: Uuid
    noteVersion: NoteVersion
    span: Span


class ExternalSource(Closed):
    """생성에 쓴 인터넷 자료. 결과 카드에 URL 을 보여 주려고 남긴다."""
    url: Url
    title: Name | None = None
    retrievedAt: Timestamp
    quote: Quote | None = None


class Concept(Closed):
    """키워드 안의 설명 1개.

    extracted: 노트에서 뽑은 개념. 소스는 그 페이지 하나이고 외부 자료는 없다.
    generated: 사용자가 모르는 지식으로 만든 개념. 소스에 도출에 쓴 개념들의 페이지(둘 이상, 서로 다른 노트)를 모두 넣는다.
    """
    conceptId: Uuid
    origin: Literal["extracted", "generated"]
    explanation: Explanation
    sources: Annotated[list[PageSource], Field(min_length=1, max_length=20)]
    external: Annotated[list[ExternalSource], Field(max_length=10)] = []

    @model_validator(mode="after")
    def _origin_rules(self):
        if self.origin == "extracted":
            if len(self.sources) != 1:
                raise ValueError("extracted 개념의 소스는 페이지 하나다")
            if self.external:
                raise ValueError("extracted 개념에는 외부 자료가 없다")
        elif len({s.noteId for s in self.sources}) < 2:
            raise ValueError("generated 개념은 서로 다른 두 노트 이상을 소스로 가져야 한다")
        return self


class Keyword(Closed):
    """그래프의 노드. 같은 뜻의 개념들을 묶는 단위 (뜻이 다르면 다른 키워드)."""
    keywordId: Uuid
    name: Name
    aliases: Annotated[list[Name], Field(max_length=50)] = []
    concepts: Annotated[list[Concept], Field(min_length=1, max_length=500)]

    @model_validator(mode="after")
    def _unique(self):
        ids = [c.conceptId for c in self.concepts]
        if len(set(ids)) != len(ids):
            raise ValueError("conceptId 가 겹친다")
        if self.name in self.aliases:
            raise ValueError("name 이 aliases 에 또 들어 있다")
        return self

    def note_ids(self) -> set[str]:
        """이 키워드에 개념을 넣은 페이지들. 사용자용 그래프는 같은 키워드를 공유하는 두 페이지를 잇는다."""
        return {s.noteId for c in self.concepts for s in c.sources}
