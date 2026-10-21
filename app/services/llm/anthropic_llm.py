"""Anthropic (Claude). 구조화 출력: output_config.format. Opus 5.5 는 thinking 이 항상 켜져 있어 effort 로만 조절한다."""

from . import LLMConfig, LLMError, LLMResult, parse_object


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, config: LLMConfig, client=None, effort: str = "low"):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=config.api_key, max_retries=0, timeout=config.timeout)
        self.client, self.config, self.model, self.effort = client, config, config.model, config.effort or effort

    def complete_json(self, system: str, user: str, schema: dict, name: str) -> LLMResult:
        import anthropic

        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_output_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APITimeoutError as e:          # APIConnectionError 의 하위 클래스라 먼저 잡는다
            raise LLMError("timeout", str(e))
        except anthropic.RateLimitError as e:
            raise LLMError("rate_limit", str(e))
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise LLMError("auth", str(e))
        except anthropic.APIStatusError as e:
            raise LLMError("provider", f"{e.status_code}")
        except anthropic.APIConnectionError as e:
            raise LLMError("provider", str(e))

        if response.stop_reason == "refusal":           # 안전 분류기가 거절 (다른 모델로 자동 전환하지 않는다)
            raise LLMError("refusal", f"category={getattr(getattr(response, 'stop_details', None), 'category', None)}")
        if response.stop_reason != "end_turn":          # max_tokens 등
            raise LLMError("invalid_output", f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        return LLMResult(parse_object(text), self.provider, getattr(response, "model", None) or self.model)
