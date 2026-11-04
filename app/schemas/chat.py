"""POST /chat 요청 (챗봇, 여러 턴). 필드 이름은 계약처럼 camelCase 이고 모르는 필드는 거절한다.

AI 서버는 대화를 저장하지 않는다 (상태 없음). BE 가 대화 기록을 보관하고 매 요청에 지금까지의 messages 를 모두 보낸다.
knowledge 는 사용자가 보고 있는 새 지식 카드 (근거 A·B, 외부 지식, 결론) 이고, 모든 턴에서 챗봇의 배경 자료가 된다.
"""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .common import Closed, Uuid

Short = Annotated[str, StringConstraints(max_length=200)]
Passage = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Url = Annotated[str, StringConstraints(pattern=r"^https?://\S+$", max_length=2048)]

MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 4000


class Evidence(Closed):
    label: Literal["A", "B"]
    keyword: Short
    note: Short                 # 노트(페이지) 이름
    text: Passage               # 노트에서 뽑은 문장


class External(Closed):
    url: Url
    title: Short | None = None
    quote: Annotated[str, StringConstraints(max_length=600)] | None = None


class Knowledge(Closed):
    newKeyword: Short
    conclusion: Passage
    evidence: Annotated[list[Evidence], Field(min_length=1, max_length=4)]
    external: Annotated[list[External], Field(max_length=5)] = []


class ChatMessage(Closed):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(min_length=1, max_length=MAX_MESSAGE_CHARS)]


class ChatRequest(Closed):
    conversationId: Uuid        # 추적용. 서버는 이 값으로 대화를 찾지 않는다
    knowledge: Knowledge
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=MAX_MESSAGES)]
    stream: bool = True         # true: SSE 로 토큰마다 · false: 다 쓴 답 하나를 JSON 으로

    @model_validator(mode="after")
    def _turns(self):
        roles = [m.role for m in self.messages]
        if roles[0] != "user" or roles[-1] != "user":
            raise ValueError("messages 는 사용자 발화로 시작해 사용자 발화로 끝나야 한다")
        if any(a == b for a, b in zip(roles, roles[1:])):
            raise ValueError("user 와 assistant 가 번갈아 나와야 한다")
        return self
