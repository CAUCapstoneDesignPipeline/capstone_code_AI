"""OpenAI. Responses API 의 구조화 출력: text.format = json_schema (strict). 응답은 저장하지 않는다 (store=False)."""

from . import LLMConfig, LLMError, LLMResult, parse_object


class OpenAILLM:
    provider = "openai"

    def __init__(self, config: LLMConfig, client=None):
        if client is None:
            import openai
            client = openai.OpenAI(api_key=config.api_key, max_retries=0, timeout=config.timeout)
        self.client, self.config, self.model = client, config, config.model

    def complete_json(self, system: str, user: str, schema: dict, name: str) -> LLMResult:
        import openai

        try:
            response = self.client.responses.create(
                model=self.config.model,
                instructions=system,
                input=user,
                max_output_tokens=self.config.max_output_tokens,
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                store=False,
            )
        except openai.APITimeoutError as e:             # APIConnectionError 의 하위 클래스라 먼저 잡는다
            raise LLMError("timeout", str(e))
        except openai.RateLimitError as e:
            raise LLMError("rate_limit", str(e))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
            raise LLMError("auth", str(e))
        except openai.APIStatusError as e:
            raise LLMError("provider", f"{e.status_code}")
        except openai.APIConnectionError as e:
            raise LLMError("provider", str(e))

        if response.status != "completed":              # incomplete (토큰 한도 등)
            reason = getattr(response.incomplete_details, "reason", None)
            raise LLMError("invalid_output", f"status={response.status} {reason or ''}".strip())
        text = response.output_text
        if not text:                                    # 거절이면 본문 대신 refusal 이 온다
            raise LLMError("invalid_output", "empty output (refusal?)")
        return LLMResult(parse_object(text), self.provider, getattr(response, "model", None) or self.model)
