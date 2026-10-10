"""추출 결과 형식. 모든 추출 방법(통계, 마크다운 구조, LLM)이 같은 형식을 낸다 (design §5).

ERD 의 concept_mention (surface, start_offset, end_offset) 과 맞춘다.
위치는 ERD E1 에 따라 NFC 마크다운 원문 기준 UTF-16 코드 단위다.
"""

from dataclasses import asdict, dataclass
from enum import Enum


class Method(Enum):
    TFIDF = "tfidf"
    MARKDOWN = "markdown"
    LLM = "llm"
    USER = "user"


@dataclass(frozen=True, order=True)
class Span:
    note_id: str
    note_version: int
    start: int  # UTF-16 코드 단위
    end: int


@dataclass(frozen=True)
class Mention:
    span: Span
    surface: str          # 원문 표기 그대로
    key: str              # 정규화 키 (같은 표기 → 같은 key). 별칭 병합은 다음 단계
    method: Method
    method_version: str   # 설정 version + IDF version
    score: float          # 이 key 의 노트 안 점수

    def to_dict(self):
        d = asdict(self)
        d["method"] = self.method.value
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(span=Span(**d["span"]), surface=d["surface"], key=d["key"],
                   method=Method(d["method"]), method_version=d["method_version"], score=d["score"])
