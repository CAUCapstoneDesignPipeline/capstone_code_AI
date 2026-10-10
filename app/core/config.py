"""서비스 설정. 환경 변수에서 읽고, 테스트에서는 create_app 인자로 바꾼다.

  AI_SERVICE_TOKEN   BE 서비스 secret (Bearer). 없으면 모든 요청에 503
  AI_CONCEPT_IDF     기준 IDF 파일 (기본 data/private/concepts/idf.json). 없으면 ready·extract 가 503
  AI_MAX_INFLIGHT    동시에 받을 추출 요청 수 (기본 4). 넘으면 429 AI_BUSY
"""

import os
from dataclasses import dataclass
from pathlib import Path

from app.services.concepts import ExtractConfig

API_PREFIX = "/internal/v1"
SCHEMA_VERSION = "1.0"
MAX_BODY_BYTES = 2 * 1024 * 1024        # 요청 본문 2 MiB (ai-openapi.yaml info.description)

# 등록된 불변 추론 profile. 이름만 바꿔 다른 설정을 쓰지 않는다 (ai-api.md §2, §5).
EXTRACT_PROFILE = "extract-concepts-det-1"
PROFILES = {EXTRACT_PROFILE: ExtractConfig()}


@dataclass(frozen=True)
class Settings:
    service_token: str | None
    concept_idf: Path
    max_inflight: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            service_token=os.environ.get("AI_SERVICE_TOKEN"),
            concept_idf=Path(os.environ.get("AI_CONCEPT_IDF", "data/private/concepts/idf.json")),
            max_inflight=int(os.environ.get("AI_MAX_INFLIGHT", "4")),
        )
