import json
import random

import pytest

from app.services.llm.partial_json import JsonStream

DOC = {"status": "ok", "newKeyword": "이중 버퍼링", "n": 3, "pi": -1.5e2, "yes": True, "no": False, "nothing": None,
       "conclusion": "DB 도 \"buffer pool\" 에 올려요.\n두 줄째 \\ 역슬래시, 탭\t, 이모지 🧠, 한글 ✓",
       "external": [{"url": "https://a.example/x?q=1&r=2", "quote": "한 문장."}, {"url": "https://b.example", "quote": "둘째 é"}],
       "results": [{"id": "p1", "label": "useful"}, {"id": "p2", "label": "vague"}], "tags": ["가", "나\n다", 7], "empty": "", "reason": ""}


def leaves(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, f"{path}.{i}" if path else str(i))
    else:
        yield path, obj


def run(text, sizes):
    js, events, i = JsonStream(), [], 0
    for n in sizes:
        events += js.feed(text[i:i + n])
        i += n
    events += js.feed(text[i:])
    return events


def check(events, doc):
    values = {e["path"]: e["value"] for e in events if "value" in e}
    deltas = {}
    for e in events:
        if "delta" in e:
            deltas[e["path"]] = deltas.get(e["path"], "") + e["delta"]
    for path, v in leaves(doc):
        assert values[path] == v and type(values[path]) is type(v), path
        if isinstance(v, str) and v:
            assert deltas[path] == v, path                                  # 조각 글을 이으면 값이 된다
    assert all(isinstance(v, str) for v in deltas.values())
    return values


@pytest.mark.parametrize("ascii_only", [True, False])
def test_every_value_is_reported_whatever_the_chunking(ascii_only):
    text = json.dumps(DOC, ensure_ascii=ascii_only, indent=1)                # ascii 면 한글·이모지가 \uXXXX (서로게이트 쌍) 로 온다
    check(run(text, []), DOC)
    check(run(text, [1] * len(text)), DOC)                                   # 글자마다
    for seed in range(30):
        rng = random.Random(seed)
        check(run(text, [rng.randint(1, 9) for _ in range(len(text))]), DOC)


def test_same_events_for_any_split():
    text = json.dumps(DOC, ensure_ascii=True)
    norm = lambda evs: [(e["path"], e.get("value"), "".join(x["delta"] for x in evs if x["path"] == e["path"] and "delta" in x)) for e in evs if "value" in e]
    base = norm(run(text, []))
    for seed in range(10):
        rng = random.Random(seed)
        assert norm(run(text, [rng.randint(1, 5) for _ in range(len(text))])) == base


def test_text_streams_while_it_is_still_being_written():
    js = JsonStream()
    first = js.feed('{"status": "ok", "conclusion": "DB 도 버')
    assert {"path": "status", "value": "ok"} in first and {"path": "conclusion", "delta": "DB 도 버"} in first
    assert not any(e.get("path") == "conclusion" and "value" in e for e in first)       # 아직 안 끝났다
    second = js.feed('퍼를 두 번')
    assert second == [{"path": "conclusion", "delta": "퍼를 두 번"}]
    assert js.feed('" , "x": 1}') == [{"path": "conclusion", "value": "DB 도 버퍼를 두 번"}, {"path": "x", "value": 1}]


def test_escape_split_across_chunks():
    js, out = JsonStream(), []
    for piece in ['{"a": "x\\', 'n y \\u', 'd83e\\ud', 'de0 z"}']:
        out += js.feed(piece)
    assert "".join(e["delta"] for e in out if "delta" in e) == "x\n y 🧠 z" and out[-1] == {"path": "a", "value": "x\n y 🧠 z"}


def test_keys_are_never_reported_as_text_and_truncated_or_broken_input_does_not_raise():
    evs = run('{"conclusion": "끝나지 않', [])
    assert evs == [{"path": "conclusion", "delta": "끝나지 않"}]
    for junk in ['', '}}}{{[', '"loose string', '{"a" "b"}', '[1,,2', '{"a": tru', '\\\\', '{"a": "\\u12', '{"a": "\\ud83e"}']:
        run(junk, [1, 2, 3])                                                # 예외가 나지 않는다
