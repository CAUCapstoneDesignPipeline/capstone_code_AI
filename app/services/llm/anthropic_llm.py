"""Anthropic (Claude). 구조화 출력: output_config.format. Opus 5.5 는 thinking 이 항상 켜져 있어 effort 로만 조절한다.

system prompt 는 역할마다 고정이라(설명 생성·관계·검증) 요청마다 바뀌는 user 앞에 cache 경계를 둔다. 같은 역할의 요청이
5분 안에 이어지면 system 은 캐시에서 읽는다 (Opus 5.5 최소 512 토큰보다 짧은 prompt 는 조용히 캐시되지 않을 뿐이다).
"""

import json
from dataclasses import dataclass, field

from . import LLMConfig, LLMError, LLMResult, parse_object
from .partial_json import JsonStream

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
                     tool: str = WEB_SEARCH_TOOL, on_event=None, submit: str | None = None) -> WebResult:
        """웹 검색 도구를 쓰고 구조화 출력으로 답한다. 서버 도구가 한 턴에 끝나지 않으면(pause_turn) 이어서 받는다 (최대 4번).
        sources: 검색 결과에 실제로 나온 URL 들. 자동 재시도·모델 fallback 은 없다.
        on_event(dict): 진행 상황을 그때그때 알린다 — thinking · search_start · search(query) · search_results(results) · writing(chars) ·
        field(path, delta | value): 모델이 쓰는 JSON 의 문자열 값이 쓰이는 대로 (결론이 토큰마다 보이게), 값 하나가 끝나면 value.
        콜백이 예외를 던지면 스트림을 닫아 모델 호출을 멈춘다 (화면에서 중단했을 때 토큰을 더 쓰지 않으려고).
        submit: 결과를 구조화 출력 대신 이 이름의 클라이언트 도구 호출로 받는다 (input_schema = schema, eager_input_streaming). 웹 검색 뒤에 구조화 출력을 쓰면
        결과 글이 끝에 한꺼번에 오지만 (실측: 모든 조각이 0.01초 안에), 도구 입력은 쓰이는 대로 온다 (결론 70조각이 3.5초에 걸쳐). 도구 입력은 코드가 엄격하게 다시 파싱한다."""
        import anthropic

        emit = on_event or (lambda e: None)
        messages = [{"role": "user", "content": user}]
        sources: dict[str, str] = {}
        searches = 0
        usage: dict[str, int] = {}
        for _ in range(MAX_PAUSES + 1):
            blocks: dict[int, dict] = {}
            chars = last = 0
            fields = JsonStream()
            submitted: list[str] = []
            tools = [{"type": tool, "name": "web_search", "max_uses": max_searches}]
            output = {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}}
            if submit:
                tools.append({"name": submit, "description": "검색을 마친 뒤 마지막에 한 번 호출해 결과를 제출한다.", "input_schema": schema, "eager_input_streaming": True})
                output = {"effort": self.effort}
            try:
                with self.client.messages.stream(
                        model=self.config.model,
                        max_tokens=self.config.max_output_tokens,
                        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                        messages=messages,
                        tools=tools,
                        output_config=output) as stream:
                    for ev in stream:
                        if ev.type == "content_block_start":
                            b = ev.content_block
                            blocks[ev.index] = {"type": b.type, "name": getattr(b, "name", ""), "json": ""}
                            if b.type == "thinking":
                                emit({"type": "thinking"})
                            elif b.type == "server_tool_use" and b.name == "web_search":
                                emit({"type": "search_start"})
                            elif b.type == "web_search_tool_result":
                                found = [{"url": r.url, "title": getattr(r, "title", "") or ""} for r in b.content] if isinstance(b.content, list) else []
                                emit({"type": "search_results", "results": found})
                            elif b.type == "text":
                                emit({"type": "writing", "chars": 0})
                        elif ev.type == "content_block_delta":
                            d = ev.delta
                            if d.type == "input_json_delta" and ev.index in blocks:
                                blocks[ev.index]["json"] += d.partial_json
                                if submit and blocks[ev.index]["name"] == submit:           # 제출 도구의 입력: 쓰이는 대로 필드를 흘린다
                                    submitted.append(d.partial_json)
                                    for fe in fields.feed(d.partial_json):
                                        emit({"type": "field", **fe})
                            elif d.type == "text_delta":
                                chars += len(d.text)
                                for fe in fields.feed(d.text):            # 쓰이는 JSON 의 필드: 결론 등이 글자가 오는 대로 보인다
                                    emit({"type": "field", **fe})
                                if chars - last >= 200:                   # 글자 수 알림은 200자마다
                                    last = chars
                                    emit({"type": "writing", "chars": chars})
                        elif ev.type == "content_block_stop":
                            blk = blocks.get(ev.index)
                            if blk and blk["type"] == "server_tool_use" and blk["name"] == "web_search":
                                try:
                                    emit({"type": "search", "query": str(json.loads(blk["json"] or "{}").get("query", ""))})
                                except ValueError:
                                    emit({"type": "search", "query": ""})
                    response = stream.get_final_message()
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
            if submit and response.stop_reason == "tool_use":
                call = next((b for b in response.content if b.type == "tool_use" and b.name == submit), None)
                if call is None:
                    raise LLMError("invalid_output", "tool_use without the submit tool")
                data = parse_object("".join(submitted))                  # SDK 의 너그러운 파서가 아니라 우리가 받은 글 그대로 엄격하게
            elif response.stop_reason == "end_turn":
                data = parse_object(next((b.text for b in reversed(response.content) if b.type == "text"), None))
            else:
                raise LLMError("invalid_output", f"stop_reason={response.stop_reason}")
            return WebResult(data, self.provider, getattr(response, "model", None) or self.model,
                             [{"url": u, "title": t} for u, t in sources.items()], searches, usage)
        raise LLMError("provider", "pause_turn 이 계속된다")

    def stream_json(self, system: str, user: str, schema: dict, name: str, on_event=None) -> LLMResult:
        """complete_json 과 같은 결과를 내되, 모델이 쓰는 JSON 의 필드를 쓰이는 대로 알린다 (선별·관계 확인이 토큰마다 보이게).
        on_event(dict): field(path, delta | value). 콜백이 예외를 던지면 스트림을 닫아 모델 호출을 멈춘다. 오류·거절 처리는 complete_json 과 같다."""
        import anthropic

        emit = on_event or (lambda e: None)
        fields = JsonStream()
        try:
            with self.client.messages.stream(
                    model=self.config.model,
                    max_tokens=self.config.max_output_tokens,
                    system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": user}],
                    output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}}) as stream:
                for ev in stream:
                    if ev.type == "content_block_delta" and ev.delta.type == "text_delta":
                        for fe in fields.feed(ev.delta.text):
                            emit({"type": "field", **fe})
                response = stream.get_final_message()
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
        if response.stop_reason == "refusal":
            raise LLMError("refusal", f"category={getattr(getattr(response, 'stop_details', None), 'category', None)}")
        if response.stop_reason != "end_turn":
            raise LLMError("invalid_output", f"stop_reason={response.stop_reason}")
        text = next((b.text for b in response.content if b.type == "text"), None)
        return LLMResult(parse_object(text), self.provider, getattr(response, "model", None) or self.model)
