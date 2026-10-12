"""서비스 설정과 추론 profile 등록부.

환경 변수
  AI_SERVICE_TOKEN        BE 서비스 secret (Bearer). 없으면 모든 요청에 503
  AI_CONCEPT_IDF          기준 IDF 파일 (기본 data/private/concepts/idf.json). 없으면 추출 profile 이 준비되지 않음
  AI_MAX_INFLIGHT         동시에 받을 요청 수 (기본 4). 넘으면 429 AI_BUSY
  AI_RELATION_LLM         1 이면 rules+LLM 추출 profile (extract-hybrid-1) 을 켠다
  AI_LLM_PROVIDER         LLM 제공자: anthropic | openai | google
  AI_LLM_MODEL            모델 id (anthropic 기본 claude-opus-5-5, google 기본 gemini-3.5-flash, openai 는 필수)
  AI_RELATION_LLM_PROVIDER / AI_RELATION_LLM_MODEL   관계 판정에만 다른 제공자·모델을 쓸 때
  인증은 각 회사 SDK 표준 변수: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY (또는 GOOGLE_API_KEY)
  AI_RELATION_CACHE       LLM 판정 캐시 폴더 (기본 data/private/cache/relations)
  AI_EMBEDDING_PROFILES   켤 임베딩 profile id, 쉼표로 구분 (기본 없음)
  AI_DEV_UI               1 이면 개발용 playground (/dev/) 를 붙인다. 배포에서는 켜지 않는다

profile 은 불변이다. 동작을 바꾸면 새 id 를 만든다 (ai-api.md §2, ai-communication.md "pipelineVersion").
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from app.services.concepts import ExtractConfig
from app.services.relations import Limits

API_PREFIX = "/internal/v1"
SCHEMA_VERSION = "1.0"
MAX_BODY_BYTES = 2 * 1024 * 1024        # 요청 본문 2 MiB
REQUEST_TIMEOUT_S = 115                 # AI 전체 처리 (BE 호출 제한 130s 안쪽)
EMBEDDING_TIMEOUT_S = 30


@dataclass(frozen=True)
class ExtractProfile:
    """개념 선택 규칙과 관계 방식. 개념·언급 수 상한은 잘라내기가 아니라 이 profile 의 선택 규칙이다
    (점수 순 → key 순으로 max_concepts 개, 개념마다 위치 순으로 max_mentions 개)."""
    id: str
    concepts: ExtractConfig = field(default_factory=ExtractConfig)
    max_concepts: int = 200             # ExtractResponse.concepts maxItems
    max_mentions: int = 20              # ExtractedConcept.mentions maxItems
    relations: str = "none"             # "none" | "rules" | "rules+llm"
    limits: Limits = field(default_factory=Limits)


EXTRACT_PROFILES = {p.id: p for p in (
    ExtractProfile("extract-concepts-det-1"),                     # 개념만 (첫 버전, 그대로 유지)
    ExtractProfile("extract-det-2", relations="rules"),           # 개념 + 규칙 관계·근거
    ExtractProfile("extract-hybrid-1", relations="rules+llm"),    # + 규칙이 못 정한 쌍은 LLM (캐시)
)}
EXTRACT_PROFILE = "extract-det-2"       # 권장 기본값


@dataclass(frozen=True)
class EmbeddingProfile:
    id: str
    backend: str                        # "hash" | "sentence-transformers"
    dims: int
    model: str = ""
    revision: str = ""
    prefix: str = ""


EMBEDDING_PROFILES = {p.id: p for p in (
    # 후보 모델. 팀이 모델·차원을 정하면 revision 을 commit hash 로 고정한다 (ai-communication.md 34행).
    EmbeddingProfile("embeddings-e5-small-1", "sentence-transformers", 384,
                     model="intfloat/multilingual-e5-small", revision="main", prefix="query: "),
    # 모델 파일 없이 쓰는 개발·테스트용. 한국어↔영어 표기를 잇지 못하므로 실제 분석에 쓰지 않는다.
    EmbeddingProfile("embeddings-hash-dev-1", "hash", 256),
)}


@dataclass(frozen=True)
class Settings:
    service_token: str | None
    concept_idf: Path
    max_inflight: int = 4
    relation_llm: bool = False
    relation_cache: Path | None = Path("data/private/cache/relations")
    embedding_profiles: tuple[str, ...] = ()
    dev_ui: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            service_token=env("AI_SERVICE_TOKEN"),
            concept_idf=Path(env("AI_CONCEPT_IDF", "data/private/concepts/idf.json")),
            max_inflight=int(env("AI_MAX_INFLIGHT", "4")),
            relation_llm=env("AI_RELATION_LLM", "0") == "1",
            relation_cache=Path(env("AI_RELATION_CACHE", "data/private/cache/relations")),
            embedding_profiles=tuple(p.strip() for p in env("AI_EMBEDDING_PROFILES", "").split(",") if p.strip()),
            dev_ui=env("AI_DEV_UI", "0") == "1",
        )
