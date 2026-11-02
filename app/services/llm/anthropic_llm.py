"""Anthropic (Claude). 구조화 출력: output_config.format. Opus 5.5 는 thinking 이 항상 켜져 있어 effort 로만 조절한다.

system prompt 는 역할마다 고정이라(설명 생성·관계·검증) 요청마다 바뀌는 user 앞에 cache 경계를 둔다. 같은 역할의 요청이
5분 안에 이어지면 system 은 캐시에서 읽는다 (Opus 5.5 최소 512 토큰보다 짧은 prompt 는 조용히 캐시되지 않을 뿐이다).
"""

from dataclasses import dataclass, field

from . import LLMConfig, LLMError, LLMResult, parse_object

WEB_SEARCH_TOOL = "web_search_20260209"      # Opus 5.5 · Sonnet 5.5 · Sonnet 5 에서 쓰는 동적 필터링판 (코드 실행으로 결과를 거른다)
WEB_SEARCH_BASIC = "web_search_20250305"     # 기본판. 코드 실행 없이 검색 결과를 바로 읽는다
MAX_PAUSES = 4


@dataclass(frozen=True)
class WebResult:
    """웹 검색을 쓴 호출의 결과. sources 는 이 호출에서 검색이 실제로 돌려준 페이지들이다 (모델이 지어낸 URL 을 가려내는 기준)."""
    data: dict
    provider: str
    model: str
    sources: list[dict] = field(default_factory=list)      # [{url, title}]
    searches: int = 0
    usage: dict = field(default_factory=dict)              # 입력·출력 토큰 합 (여러 턴이면 더한다)


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
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
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

    def complete_web(self, system: str, user: str, schema: dict, name: str, max_searches: int = 3,
                     tool: str = WEB_SEARCH_TOOL) -> WebResult:
        """웹 검색 도구를 쓰고 구조화 출력으로 답한다. 서버 도구가 한 턴에 끝나지 않으면(pause_turn) 이어서 받는다 (최대 4번).
        sources: 검색 결과에 실제로 나온 URL 들. 자동 재시도·모델 fallback 은 없다."""
        import anthropic

        messages = [{"role": "user", "content": user}]
        sources: dict[str, str] = {}
        searches = 0
        usage: dict[str, int] = {}
        for _ in range(MAX_PAUSES + 1):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_output_tokens,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=messages,
                    tools=[{"type": tool, "name": "web_search", "max_uses": max_searches}],
                    output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                )
            except anthropic.APITimeoutError as e:
                raise LLMError("timeout", str(e))
            except anthropic.RateLimitError as e:
                raise LLMError("rate_limit", str(e))
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
                raise LLMError("auth", str(e))
            except anthropic.APIStatusError as e:
                raise LLMError("provider", f"{e.status_code}")
            except anthropic.APIConnectionError as e:
                raise LLMError("provider", str(e))
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                usage[k] = usage.get(k, 0) + (getattr(response.usage, k, 0) or 0)
            for b in response.content:
                if b.type == "server_tool_use" and getattr(b, "name", "") == "web_search":
                    searches += 1
                if b.type == "web_search_tool_result" and isinstance(b.content, list):
                    for r in b.content:
                        if getattr(r, "url", None):
                            sources.setdefault(r.url, getattr(r, "title", "") or "")
            if response.stop_reason == "pause_turn":
                messages = [*messages, {"role": "assistant", "content": response.content}]
                continue
            if response.stop_reason == "refusal":
                raise LLMError("refusal", f"category={getattr(getattr(response, 'stop_details', None), 'category', None)}")
            if response.stop_reason != "end_turn":
                raise LLMError("invalid_output", f"stop_reason={response.stop_reason}")
            text = next((b.text for b in reversed(response.content) if b.type == "text"), None)
            return WebResult(parse_object(text), self.provider, getattr(response, "model", None) or self.model,
                             [{"url": u, "title": t} for u, t in sources.items()], searches, usage)
        raise LLMError("provider", "pause_turn 이 계속된다")
