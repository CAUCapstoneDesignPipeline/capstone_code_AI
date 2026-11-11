"""외부 지식 찾기: 지어낸·안전하지 않은 URL 거르기, 중복·긴 요약 버리기, 출처 판정, maxSources, none."""

import pytest

from app.services.links.external import ExternalKnowledgeFinder, SourceJudge, safe_url
from app.services.llm import LLMError
from app.services.llm.anthropic_llm import WebResult

CLAIM = {"conclusion": "WAL 덕분에 장애가 나도 되돌릴 정보가 로그에 남아서 트랜잭션을 통째로 취소할 수 있어요.",
         "evidence": [{"label": "A", "keyword": "WAL", "text": "WAL은 로그를 먼저 디스크에 쓴다."},
                      {"label": "B", "keyword": "Atomicity", "text": "Atomicity는 전부 반영되거나 전혀 반영되지 않는 성질이다."}]}
U1, U2, U3, U4 = ("https://example.com/aries", "https://example.org/wal", "https://example.net/acid", "https://example.io/misc")
FOUND = [{"url": U1, "title": "ARIES recovery"}, {"url": U2, "title": "Write-ahead logging"}, {"url": U3, "title": "ACID"}, {"url": U4, "title": "Misc"}]


class FakeWeb:
    provider, model = "anthropic", "fake-gen"

    def __init__(self, sources, found=FOUND, usage=None):
        self.sources, self.found, self.calls = sources, found, []
        self.usage = usage if usage is not None else {"input_tokens": 190, "cache_read_input_tokens": 4000, "output_tokens": 1080}

    def complete_web(self, system, user, schema, name, max_searches=3, tool=None, on_event=None, submit=None):
        self.calls.append(dict(user=user, max_searches=max_searches, tool=tool, submit=submit))
        return WebResult({"sources": self.sources}, self.provider, self.model, list(self.found), 1, self.usage)


class FakeJudge:
    model = "fake-judge"

    def __init__(self, verdicts):
        self.verdicts, self.asked = verdicts, []

    def judge(self, conclusion, sources):
        self.asked.append(sources)
        return {s["id"]: self.verdicts.get(s["url"], "no") for s in sources}


def src(url, summary="그 페이지가 설명하는 내용을 한 문장으로 옮긴 거예요."):
    return {"url": url, "summary": summary}


def find(sources, verdicts, max_sources=1, **kw):
    web, judge = FakeWeb(sources, **kw), FakeJudge(verdicts)
    return ExternalKnowledgeFinder(web, judge).find(CLAIM, max_sources), web, judge


def test_a_supported_source_comes_back_with_the_search_title_and_usage():
    f, web, judge = find([src(U1)], {U1: "yes"})
    assert f.status == "found" and len(f.items) == 1
    it = f.items[0]
    assert it["sourceUrl"] == U1 and it["title"] == "ARIES recovery" and it["support"] == "yes" and it["retrievedAt"].endswith("Z")
    assert (f.searches, f.input_tokens, f.output_tokens, f.model) == (1, 4190, 1080, "fake-gen")
    assert web.calls[0]["max_searches"] == 1 and web.calls[0]["submit"] and "web_search_20250305" == web.calls[0]["tool"]
    assert [list(s) for s in judge.asked[0]] == [["id", "title", "url"]]          # 판정은 제목과 URL 만 본다 (모델이 쓴 요약은 주지 않는다)


def test_a_url_the_search_never_returned_is_dropped():
    f, _, judge = find([src("https://made.up/page"), src(U1)], {U1: "yes", "https://made.up/page": "yes"}, max_sources=3)
    assert [i["sourceUrl"] for i in f.items] == [U1] and len(judge.asked[0]) == 1


def test_a_trailing_slash_difference_still_matches_the_search_result():
    f, _, _ = find([src(U1 + "/")], {U1: "yes"})
    assert [i["sourceUrl"] for i in f.items] == [U1]                              # 검색 결과의 URL 로 돌려준다


@pytest.mark.parametrize("url", ["http://localhost:8000/x", "https://10.0.0.5/a", "https://user:pw@example.com/a", "https://ai.internal/x",
                                 "ftp://example.com/a", "javascript:alert(1)", "https://127.0.0.1/"])
def test_an_unsafe_url_is_dropped_even_if_the_search_returned_it(url):
    f, _, judge = find([src(url)], {url: "yes"}, found=[{"url": url, "title": "x"}])
    assert f.status == "none" and f.items == [] and judge.asked == []            # 남은 출처가 없으면 판정도 부르지 않는다


def test_duplicates_and_overlong_or_empty_summaries_are_dropped_not_cut():
    f, _, judge = find([src(U1), src(U1 + "/"), src(U2, "가" * 301), src(U3, "   "), src(U4, "가" * 300)],
                       {U1: "yes", U2: "yes", U3: "yes", U4: "partly"}, max_sources=3)
    assert [i["sourceUrl"] for i in f.items] == [U1, U4] and f.items[1]["summary"] == "가" * 300
    assert [s["url"] for s in judge.asked[0]] == [U1, U4]


def test_the_judge_drops_unrelated_sources_and_yes_comes_before_partly():
    f, _, _ = find([src(U1), src(U2), src(U3), src(U4)], {U1: "partly", U2: "no", U3: "yes", U4: "yes"}, max_sources=3)
    assert [(i["sourceUrl"], i["support"]) for i in f.items] == [(U3, "yes"), (U4, "yes"), (U1, "partly")]


def test_max_sources_caps_the_result_after_ordering():
    f, _, _ = find([src(U1), src(U2), src(U3)], {U1: "partly", U2: "yes", U3: "yes"}, max_sources=2)
    assert [i["sourceUrl"] for i in f.items] == [U2, U3]
    f, _, _ = find([src(U1), src(U2)], {U1: "yes", U2: "yes"})                  # 기본 1
    assert len(f.items) == 1


def test_when_nothing_is_supported_the_answer_is_none_not_an_error():
    f, _, _ = find([src(U1), src(U2)], {})
    assert (f.status, f.items) == ("none", []) and f.searches == 1
    f, _, judge = find([], {})
    assert (f.status, f.items) == ("none", []) and judge.asked == []


def test_a_model_that_returns_malformed_sources_is_an_invalid_output():
    web = FakeWeb([])
    web.complete_web = lambda *a, **k: WebResult("not an object", "anthropic", "fake", [], 1, {})
    with pytest.raises(LLMError) as e:
        ExternalKnowledgeFinder(web, FakeJudge({})).find(CLAIM, 1)
    assert e.value.kind == "invalid_output"
    f, _, _ = find([None, {"url": 3, "summary": "x"}, {"summary": "y"}, src(U1)], {U1: "yes"}, max_sources=3)   # 모양이 틀린 항목은 건너뛴다
    assert [i["sourceUrl"] for i in f.items] == [U1]


def test_the_model_title_is_never_used_and_a_long_search_title_is_capped():
    f, _, _ = find([{**src(U1), "title": "모델이 쓴 제목"}], {U1: "yes"}, found=[{"url": U1, "title": "가" * 250}])
    assert f.items[0]["title"] == "가" * 200
    f, _, _ = find([src(U1)], {U1: "yes"}, found=[{"url": U1, "title": ""}])
    assert f.items[0]["title"] is None


def test_the_finder_needs_a_web_search_capable_llm():
    with pytest.raises(ValueError):
        ExternalKnowledgeFinder(object(), FakeJudge({}))


class JsonLLM:
    model = "fake-verify"

    def __init__(self, data):
        self.data = data

    def complete_json(self, system, user, schema, name):
        from types import SimpleNamespace
        return SimpleNamespace(data=self.data)


SOURCES = [{"id": "S1", "title": "t1", "url": U1}, {"id": "S2", "title": "t2", "url": U2}]


def test_source_judge_treats_a_missing_answer_as_no():
    got = SourceJudge(JsonLLM({"sources": [{"id": "S1", "support": "yes"}, {"id": "S9", "support": "yes"}]})).judge("c", SOURCES)
    assert got == {"S1": "yes", "S2": "no"}


def test_source_judge_rejects_a_malformed_answer():
    with pytest.raises(LLMError):
        SourceJudge(JsonLLM({"nope": []})).judge("c", SOURCES)


@pytest.mark.parametrize("url, ok", [("https://example.com/a", True), ("http://docs.python.org/3/", True), ("https://en.wikipedia.org/wiki/Page_(x)", True),
                                     ("https://[2606:4700::1111]/x", True), ("https://[::1]/x", False), ("https://intranet/page", False),
                                     ("https://a b.com/x", False), ("https://" + "a" * 2050 + ".com", False), ("", False), ("https://192.168.0.1/", False),
                                     ("https://169.254.169.254/latest", False), ("https://printer.local/", False)])
def test_safe_url(url, ok):
    assert safe_url(url) is ok
