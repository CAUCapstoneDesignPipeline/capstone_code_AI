"""LLM 제공자 공용 계층. 모든 LLM 호출은 여기를 거친다.

  complete_json(system, user, schema, name) → JSON 객체 하나 (구조화 출력으로 스키마를 강제)

제공자 (각 회사의 공식 SDK 를 그대로 쓴다)
  anthropic   ANTHROPIC_API_KEY            기본 모델 claude-opus-5-5
  openai      OPENAI_API_KEY               기본 모델 없음 (AI_LLM_MODEL 로 지정)
  google      GEMINI_API_KEY / GOOGLE_API_KEY  기본 모델 gemini-3.5-flash

계약(ai-communication.md)에 따라 SDK 자동 재시도 0, 자동 모델 fallback 없음. 재시도는 BE 가 한다.
오류는 LLMError(kind) 하나로 통일한다: timeout | rate_limit | auth | provider | invalid_output | refusal
"""

import json
import os
from dataclasses import dataclass, field

PROVIDERS = ("anthropic", "openai", "google")
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "google": "gemini-3.5-flash"}


class LLMError(Exception):
    def __init__(self, kind: str, message: str = ""):
        self.kind = kind
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True)
class LLMResult:
    data: dict
    provider: str
    model: str            # 제공자가 알려 준 실제 모델 id (없으면 요청 모델)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    timeout: float = 90.0             # 제공자 한 번 호출 (ai-communication.md AI_PROVIDER_TIMEOUT_SECONDS)
    max_output_tokens: int = 16000
    effort: str | None = None         # anthropic output_config.effort (없으면 어댑터 기본값)
    # 직접 넘긴 API 키 (없으면 각 SDK 가 표준 환경 변수에서 읽는다). 로그·repr 에 찍히지 않게 숨긴다
    api_key: str | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_env(cls, role: str | None = None) -> "LLMConfig | None":
        """AI_{ROLE}_LLM_PROVIDER / _MODEL 이 있으면 그것, 없으면 AI_LLM_PROVIDER / AI_LLM_MODEL.
        역할별로 따로 둘 수 있어야 검증 모델을 생성 모델과 다르게 고를 수 있다 (계약 §5)."""
        env = os.environ.get
        prefix = f"AI_{role.upper()}_" if role else "AI_"
        provider = (env(f"{prefix}LLM_PROVIDER") or env("AI_LLM_PROVIDER") or "").strip().lower()
        if not provider:
            return None
        if provider not in PROVIDERS:
            raise ValueError(f"알 수 없는 LLM 제공자 {provider!r}. {', '.join(PROVIDERS)} 중 하나")
        model = (env(f"{prefix}LLM_MODEL") or env("AI_LLM_MODEL") or DEFAULT_MODELS.get(provider, "")).strip()
        if not model:
            raise ValueError(f"{provider} 는 기본 모델이 없다. AI_LLM_MODEL 로 모델 id 를 정한다")
        return cls(provider, model, float(env("AI_LLM_TIMEOUT_SECONDS", "90")))


def make_llm(config: LLMConfig, client=None):
    """제공자별 어댑터. client 는 테스트에서 가짜를 넣을 때만 쓴다."""
    if config.provider == "anthropic":
        from .anthropic_llm import AnthropicLLM
        return AnthropicLLM(config, client)
    if config.provider == "openai":
        from .openai_llm import OpenAILLM
        return OpenAILLM(config, client)
    if config.provider == "google":
        from .google_llm import GoogleLLM
        return GoogleLLM(config, client)
    raise ValueError(config.provider)


def parse_object(text: str) -> dict:
    """구조화 출력의 본문 → dict. 형식이 어긋나면 invalid_output."""
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError) as e:
        raise LLMError("invalid_output", f"json: {e}")
    if not isinstance(data, dict):
        raise LLMError("invalid_output", "not an object")
    return data


__all__ = ["DEFAULT_MODELS", "LLMConfig", "LLMError", "LLMResult", "PROVIDERS", "make_llm", "parse_object"]
