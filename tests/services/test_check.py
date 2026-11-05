import json

import pytest

from app.services.links.check import KnowledgeChecker
from app.services.llm import LLMError, LLMResult

A = {"keyword": "Master Secret", "text": "Pre Master Secret + 랜덤 데이터로 Master Secret 생성"}
B = {"keyword": "개인키", "text": "Client는 개인키로 복호화해서 Server에게 보낸다."}
EXT = [{"url": "https://a.example/tls", "title": "TLS handshake", "quote": "모델이 쓴 요약 1"},
       {"url": "https://b.example/patent", "title": "Content fetching by mobile device", "quote": "모델이 쓴 요약 2"},
       {"url": "https://c.example/ssh", "title": "SSH userauth", "quote": "모델이 쓴 요약 3"}]


def reply(**kw):
    return {"onTopic": True, "bothUsed": True, "newInfo": True, "correct": True, "noteError": False,
            "sources": [{"id": "S1", "support": "yes"}, {"id": "S2", "support": "no"}, {"id": "S3", "support": "partly"}], "reason": "", **kw}


class Fake:
    provider, model = "anthropic", "fake"

    def __init__(self, data):
        self.data, self.seen = data, []

    def complete_json(self, system, user, schema, name):
        self.seen.append((system, user, name))
        return LLMResult(self.data, self.provider, self.model)


def check(data, cache=None):
    return KnowledgeChecker(Fake(data), cache).check(A, B, "TLS 키", "결론이에요", EXT)


def test_sources_are_judged_by_title_and_url_only_so_the_generators_own_summary_cannot_mislead_the_checker():
    f = Fake(reply())
    KnowledgeChecker(f).check(A, B, "TLS 키", "결론이에요", EXT)
    user = json.loads(f.seen[0][1])
    assert [list(s) for s in user["sources"]] == [["id", "title", "url"]] * 3 and "모델이 쓴" not in f.seen[0][1]
    assert user["A"]["text"] == A["text"] and user["conclusion"] == "결론이에요"


def test_good_result_keeps_supporting_sources_and_lists_the_dropped_ones():
    r = check(reply())
    assert r["ok"] and r["problems"] == [] and r["warnings"] == []
    assert [e["url"] for e in r["external"]] == ["https://a.example/tls", "https://c.example/ssh"]
    assert [e["url"] for e in r["dropped"]] == ["https://b.example/patent"]


@pytest.mark.parametrize("flag, text", [("onTopic", "주제"), ("bothUsed", "한쪽만"), ("newInfo", "새로운 내용")])
def test_structural_problems_hold_the_result(flag, text):
    r = check(reply(**{flag: False, "reason": "이유"}))
    assert not r["ok"] and any(text in p for p in r["problems"]) and r["reason"] == "이유"


def test_factual_doubt_and_note_errors_are_warnings_that_do_not_block():
    r = check(reply(correct=False, noteError=True))
    assert r["ok"] and len(r["warnings"]) == 2 and r["problems"] == []


def test_a_source_the_model_did_not_rate_is_not_kept_and_no_supporting_source_holds_the_result():
    r = check(reply(sources=[{"id": "S1", "support": "yes"}]))
    assert [e["url"] for e in r["external"]] == ["https://a.example/tls"] and len(r["dropped"]) == 2
    r = check(reply(sources=[{"id": "S1", "support": "no"}, {"id": "S2", "support": "no"}, {"id": "S3", "support": "no"}]))
    assert not r["ok"] and any("출처" in p for p in r["problems"])


@pytest.mark.parametrize("bad", [{"onTopic": None}, {"sources": None}])
def test_malformed_output_is_an_error(bad):
    data = reply()
    data.update(bad)
    if "onTopic" in bad:
        del data["onTopic"]
    with pytest.raises(LLMError):
        check(data)


def test_checks_are_cached_by_content_and_a_bad_answer_is_not_cached(tmp_path):
    f = Fake(reply())
    ck = KnowledgeChecker(f, tmp_path)
    ck.check(A, B, "k", "결론이에요", EXT)
    ck.check(A, B, "k", "결론이에요", EXT)
    assert len(f.seen) == 1
    ck.check(A, B, "k", "다른 결론이에요", EXT)
    assert len(f.seen) == 2
    bad = Fake({"sources": []})
    with pytest.raises(LLMError):
        KnowledgeChecker(bad, tmp_path).check(A, B, "z", "또 다른 결론", EXT)
    assert len(list(tmp_path.glob("*.json"))) == 2
