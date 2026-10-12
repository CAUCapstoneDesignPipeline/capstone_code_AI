"""Google (Gemini). 구조화 출력: response_mime_type=application/json + response_json_schema.
SDK 자체 재시도는 끄고 (조용히 멈춘 것처럼 보이므로), 자동 함수 호출도 끈다."""

from . import LLMConfig, LLMError, LLMResult, parse_object


class GoogleLLM:
    provider = "google"

    def __init__(self, config: LLMConfig, client=None):
        if client is None:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=config.api_key, http_options=types.HttpOptions(
                timeout=int(config.timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1)))
        self.client, self.config, self.model = client, config, config.model

    def complete_json(self, system: str, user: str, schema: dict, name: str) -> LLMResult:
        import httpx
        from google.genai import errors, types

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    max_output_tokens=self.config.max_output_tokens,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.APIError as e:
            if e.code == 429:
                raise LLMError("rate_limit", e.message or "")
            if e.code in (401, 403):
                raise LLMError("auth", e.message or "")
            raise LLMError("provider", f"{e.code}")
        except httpx.TimeoutException as e:
            raise LLMError("timeout", str(e))
        except httpx.TransportError as e:
            raise LLMError("provider", str(e))

        if not response.candidates:
            raise LLMError("invalid_output", "no candidates (blocked?)")
        finish = response.candidates[0].finish_reason
        finish = getattr(finish, "value", str(finish))
        if finish != "STOP":
            raise LLMError("invalid_output", f"finish_reason={finish}")
        return LLMResult(parse_object(response.text), self.provider, response.model_version or self.model)
