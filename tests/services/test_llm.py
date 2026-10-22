"""세 제공자 어댑터. 실제 SDK 대신 각 SDK 응답 모양의 가짜 client 로 검사한다."""

import json
from types import SimpleNamespace as NS

import pytest

from app.services.llm import LLMConfig, LLMError, make_llm

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
          "additionalProperties": False}


class Recorder:
    """client.<a>.<b>(**kw) 호출을 기록하고 정해 둔 응답을 돌려준다."""

    def __init__(self, path, response):
        self.calls, self._path, self._response = [], path, response

    def __getattr__(self, name):
        if name == self._path[0]:
            return Recorder(self._path[1:], self._response) if len(self._path) > 1 else self._call
        raise AttributeError(name)

    def _call(self, **kw):
        Recorder.last = kw
        return self._response


def anthropic_reply(text, stop="end_turn"):
    return NS(stop_reason=stop, content=[NS(type="text", text=text)], model="claude-opus-5-5")


def openai_reply(text, status="completed"):
    return NS(status=status, output_text=text, incomplete_details=NS(reason="max_output_tokens") if status != "completed" else None,
              model="openai-model-x")


def google_reply(text, finish="STOP"):
    return NS(candidates=[NS(finish_reason=NS(value=finish))], text=text, model_version="gemini-3.5-flash")


CASES = {
    "anthropic": (("messages", "create"), anthropic_reply, lambda: anthropic_reply("", stop="refusal")),
    "openai": (("responses", "create"), openai_reply, lambda: openai_reply("", status="incomplete")),
    "google": (("models", "generate_content"), google_reply, lambda: google_reply("", finish="SAFETY")),
}


@pytest.mark.parametrize("provider", list(CASES))
def test_adapter_ok(provider):
    path, ok, _ = CASES[provider]
    llm = make_llm(LLMConfig(provider, "some-model"), client=Recorder(path, ok(json.dumps({"ok": True}))))
    res = llm.complete_json("sys", "user", SCHEMA, name="t")
    assert res.data == {"ok": True} and res.provider == provider and res.model


@pytest.mark.parametrize("provider", list(CASES))
def test_adapter_refusal_or_truncation_is_invalid_output(provider):
    path, _, bad = CASES[provider]
    llm = make_llm(LLMConfig(provider, "some-model"), client=Recorder(path, bad()))
    with pytest.raises(LLMError) as e:
        llm.complete_json("sys", "user", SCHEMA, name="t")
    # anthropic 은 stop_reason=refusal 을 따로 알린다 (노트 하나만 규칙 관계로 두는 데 쓴다)
    assert e.value.kind == ("refusal" if provider == "anthropic" else "invalid_output")


@pytest.mark.parametrize("provider", list(CASES))
def test_adapter_bad_json_is_invalid_output(provider):
    path, ok, _ = CASES[provider]
    llm = make_llm(LLMConfig(provider, "some-model"), client=Recorder(path, ok("not json")))
    with pytest.raises(LLMError) as e:
        llm.complete_json("sys", "user", SCHEMA, name="t")
    assert e.value.kind == "invalid_output"


def test_request_shapes():
    make_llm(LLMConfig("anthropic", "claude-opus-5-5"), client=Recorder(("messages", "create"), anthropic_reply('{"ok":true}'))) \
        .complete_json("s", "u", SCHEMA, "t")
    kw = Recorder.last
    assert kw["output_config"]["format"] == {"type": "json_schema", "schema": SCHEMA}
    assert "fallbacks" not in kw and "betas" not in kw            # 계약: 자동 모델 fallback 금지

    make_llm(LLMConfig("openai", "m"), client=Recorder(("responses", "create"), openai_reply('{"ok":true}'))) \
        .complete_json("s", "u", SCHEMA, "t")
    kw = Recorder.last
    assert kw["text"]["format"] == {"type": "json_schema", "name": "t", "schema": SCHEMA, "strict": True}
    assert kw["store"] is False and kw["instructions"] == "s"

    make_llm(LLMConfig("google", "m"), client=Recorder(("models", "generate_content"), google_reply('{"ok":true}'))) \
        .complete_json("s", "u", SCHEMA, "t")
    cfg = Recorder.last["config"]
    assert cfg.response_json_schema == SCHEMA and cfg.response_mime_type == "application/json"
    assert cfg.automatic_function_calling.disable is True


def test_config_from_env(monkeypatch):
    for k in ("AI_LLM_PROVIDER", "AI_LLM_MODEL", "AI_RELATION_LLM_PROVIDER", "AI_RELATION_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    assert LLMConfig.from_env("relation") is None
    monkeypatch.setenv("AI_LLM_PROVIDER", "google")
    assert LLMConfig.from_env("relation").model == "gemini-3.5-flash"          # 제공자 기본 모델
    monkeypatch.setenv("AI_RELATION_LLM_PROVIDER", "anthropic")
    assert LLMConfig.from_env("relation").provider == "anthropic"               # 역할별 설정이 우선
    monkeypatch.setenv("AI_LLM_PROVIDER", "openai")
    monkeypatch.delenv("AI_RELATION_LLM_PROVIDER")
    with pytest.raises(ValueError):                                            # openai 는 모델 필수
        LLMConfig.from_env("relation")
    monkeypatch.setenv("AI_LLM_PROVIDER", "mistral")
    with pytest.raises(ValueError):
        LLMConfig.from_env()
