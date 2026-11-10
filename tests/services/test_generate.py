import pytest

from app.services.links.generate import KnowledgeGenerator
from app.services.llm.anthropic_llm import WebResult

N1, N2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
CID = "33333333-3333-4333-8333-333333333333"
A = {"keyword": "WAL", "note": "DB-WAL", "noteId": N1, "text": "WAL은 데이터 페이지보다 로그를 먼저 디스크에 쓴다.", "start": 0, "end": 30, "quote": "WAL은 데이터 페이지보다 로그를 먼저 디스크에 쓴다."}
B = {"keyword": "Atomicity", "note": "ACID", "noteId": N2, "text": "Atomicity는 트랜잭션이 전부 반영되거나 전혀 반영되지 않는 성질이다.", "start": 5, "end": 50, "quote": "Atomicity는 트랜잭션이 전부 반영되거나 전혀 반영되지 않는 성질이다."}
URL = "https://example.com/aries"
GOOD = {"status": "ok", "newKeyword": "undo 로그", "fromA": "로그를 먼저 디스크에 쓴다", "fromB": "전부 반영되거나 전혀 반영되지 않는",
        "conclusion": "WAL 덕분에 장애가 나도 로그에 되돌릴 정보가 남아 있어서, 끝나지 않은 트랜잭션을 통째로 취소할 수 있어요. 그래서 원자성이 지켜져요.",
        "external": [{"url": URL, "title": "ARIES", "quote": "ARIES 는 undo 정보를 로그에 먼저 남겨 롤백한다.", "supports": "되돌릴 정보가 로그에 남는다"}],
        "reason": "두 문장을 이으면 나오는 결론"}


class Fake:
    provider, model = "anthropic", "fake"

    def __init__(self, data, sources=({"url": URL, "title": "ARIES"},)):
        self.data, self.sources, self.calls = data, list(sources), 0

    def complete_web(self, system, user, schema, name, max_searches=3, submit=None):
        self.calls += 1
        return WebResult(self.data, self.provider, self.model, self.sources, 2)


def run(data, **kw):
    return KnowledgeGenerator(Fake(data, **kw), None).generate(A, B, CID)


def test_good_result_becomes_a_generated_concept_with_both_pages_and_the_url():
    r = run(GOOD)
    assert r["status"] == "candidate" and r["reasons"] == []
    c = r["concept"]
    assert c["origin"] == "generated" and c["explanation"] == GOOD["conclusion"] and c["conceptId"] == CID
    assert [s["noteId"] for s in c["sources"]] == [N1, N2] and c["external"][0]["url"] == URL and c["external"][0]["retrievedAt"].endswith("Z")


@pytest.mark.parametrize("change, reason", [
    ({"external": [{"url": "https://made.up/page", "title": "x", "quote": "q", "supports": "s"}]}, "외부 출처"),     # 검색에 없던 URL
    ({"external": []}, "외부 출처"),
    ({"fromA": "노트에 없는 문장"}, "A 문장"),
    ({"fromB": ""}, "B 문장"),
    ({"conclusion": "짧아요"}, "짧"),
    ({"status": "no_new_knowledge"}, "새로 말할 수 있는 것이 없다"),
    ({"status": "conflict"}, "서로 맞지 않는다"),
])
def test_unchecked_claims_are_held_not_proposed(change, reason):
    r = run({**GOOD, **change})
    assert r["status"] == "held" and r["concept"] is None and any(reason in x for x in r["reasons"])


def test_invented_urls_are_dropped_but_real_ones_stay():
    r = run({**GOOD, "external": [*GOOD["external"], {"url": "https://made.up/x", "title": "x", "quote": "q", "supports": "s"}]})
    assert r["status"] == "candidate" and [e["url"] for e in r["external"]] == [URL]


def test_conclusion_that_restates_a_note_sentence_is_known():
    r = run({**GOOD, "conclusion": A["text"], "fromA": "로그를 먼저 디스크에 쓴다"})
    assert r["status"] == "known" and r["concept"] is None


def test_trailing_slash_in_url_still_matches_and_result_is_cached(tmp_path):
    fake = Fake({**GOOD, "external": [{**GOOD["external"][0], "url": URL + "/"}]})
    g = KnowledgeGenerator(fake, tmp_path)
    assert g.generate(A, B, CID)["status"] == "candidate"
    assert g.generate(A, B, CID)["cached"] is True and fake.calls == 1


def test_search_tool_is_passed_through_only_when_chosen():
    seen = []

    class Tool(Fake):
        def complete_web(self, system, user, schema, name, max_searches=3, tool="default", submit=None):
            seen.append(tool)
            return super().complete_web(system, user, schema, name, max_searches)

    KnowledgeGenerator(Tool(GOOD), None, "web_search_20250305").generate(A, B, CID)
    KnowledgeGenerator(Tool(GOOD), None).generate(A, B, CID)
    assert seen == ["web_search_20250305", "default"]


def test_generator_reports_progress_and_a_stopped_listener_leaves_no_cache(tmp_path):
    class Fake2(Fake):
        def complete_web(self, system, user, schema, name, max_searches=3, tool=None, on_event=None, submit=None):
            on_event({"type": "search", "query": "q"})
            return super().complete_web(system, user, schema, name, max_searches)

    seen = []
    g = KnowledgeGenerator(Fake2(GOOD), tmp_path)
    assert g.generate(A, B, CID, on_event=seen.append)["status"] == "candidate"
    assert [e["type"] for e in seen] == ["start", "search", "validating"]
    seen.clear()
    assert g.generate(A, B, CID, on_event=seen.append)["cached"] is True and [e["type"] for e in seen] == ["cached"]

    class Stop(Exception):
        pass

    def stop_on_search(e):
        if e["type"] == "search":
            raise Stop()
    fresh = KnowledgeGenerator(Fake2(GOOD), tmp_path / "other")
    with pytest.raises(Stop):
        fresh.generate(A, B, CID, on_event=stop_on_search)
    assert not list((tmp_path / "other").glob("*.json"))           # 멈춘 호출은 결과를 남기지 않는다


def test_titles_come_from_the_search_results_and_the_model_only_writes_url_and_quote():
    from app.services.links.generate import MAX_SEARCHES, SCHEMA
    item = SCHEMA["properties"]["external"]["items"]
    assert set(item["properties"]) == {"url", "quote"} and "maxItems" not in SCHEMA["properties"]["external"] and MAX_SEARCHES == 1
    r = run({**GOOD, "external": [{"url": URL, "quote": "ARIES 는 undo 정보를 로그에 먼저 남긴다."}]})
    assert r["status"] == "candidate" and r["external"] == [{"url": URL, "title": "ARIES", "quote": "ARIES 는 undo 정보를 로그에 먼저 남긴다."}]
    assert r["concept"]["external"][0]["title"] == "ARIES"


def test_at_most_three_sources_are_kept_even_if_the_model_gives_more():
    many = [{"url": f"https://example.com/{i}", "quote": "q"} for i in range(5)]
    r = run({**GOOD, "external": many}, sources=[{"url": e["url"], "title": f"T{i}"} for i, e in enumerate(many)])
    assert [e["url"] for e in r["external"]] == [e["url"] for e in many[:3]]


# ---- 생성 뒤 확인 ----

class Checker:
    model, prompt_version = "check-model", "knowledgecheck-llm-2"

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def check(self, a, b, kw, conclusion, external):
        self.calls.append((kw, conclusion, [e["url"] for e in external]))
        if self.error:
            raise self.error
        return {"ok": True, "problems": [], "warnings": [], "external": external, "dropped": [], "reason": "", "reach": "extended", **(self.result or {})}


def verified(data, checker, cache=None, **kw):
    return KnowledgeGenerator(Fake(data, **kw), cache, checker=checker).generate(A, B, CID)


def test_failed_check_holds_the_result_with_the_reasons_and_no_concept():
    r = verified(GOOD, Checker({"ok": False, "problems": ["결론이 두 노트 문장의 주제를 벗어났다"], "reason": "SSH 로 갈아탔다"}))
    assert r["status"] == "held" and r["concept"] is None and r["reasons"] == ["결론이 두 노트 문장의 주제를 벗어났다", "SSH 로 갈아탔다"]
    assert r["check"]["model"] == "check-model" and r["conclusion"] == GOOD["conclusion"]               # 결론은 남겨 두어 왜 걸렀는지 볼 수 있다


def test_sources_the_check_dropped_are_removed_from_the_card_and_the_concept():
    other = {"url": "https://example.com/other", "title": "other"}
    data = {**GOOD, "external": [*GOOD["external"], {"url": other["url"], "title": "other", "quote": "q", "supports": "s"}]}
    ck = Checker({"external": [{"url": URL, "title": "ARIES", "quote": "q"}], "dropped": [{"url": other["url"], "title": "other", "quote": "q"}]})
    r = verified(data, ck, sources=[{"url": URL, "title": "ARIES"}, other])
    assert r["status"] == "candidate" and [e["url"] for e in r["external"]] == [URL] and [e["url"] for e in r["concept"]["external"]] == [URL]
    assert [e["url"] for e in r["check"]["dropped"]] == [other["url"]]


def test_warnings_are_kept_and_the_card_stays_a_candidate():
    r = verified(GOOD, Checker({"warnings": ["노트 문장이 부정확할 수 있다"]}))
    assert r["status"] == "candidate" and r["concept"] and r["check"]["warnings"] == ["노트 문장이 부정확할 수 있다"]


def test_a_check_that_errors_does_not_lose_the_card():
    from app.services.llm import LLMError
    r = verified(GOOD, Checker(error=LLMError("timeout", "x")))
    assert r["status"] == "candidate" and r["concept"] and r["check"]["error"] == "timeout"


def test_only_candidates_are_checked_and_a_cached_generation_is_checked_again_without_regenerating(tmp_path):
    ck = Checker()
    assert verified({**GOOD, "status": "no_new_knowledge"}, ck)["status"] == "held" and ck.calls == []
    fake = Fake(GOOD)
    g = KnowledgeGenerator(fake, tmp_path, checker=ck)
    g.generate(A, B, CID)
    assert g.checker is ck and len(ck.calls) == 1
    g.checker = Checker({"ok": False, "problems": ["새 기준"]})              # 확인 기준이 바뀌면 캐시된 생성도 새 기준으로 다시 본다
    r = g.generate(A, B, CID)
    assert r["cached"] is True and r["status"] == "held" and fake.calls == 1


def test_checking_event_is_emitted_after_validating():
    class Streaming(Fake):
        def complete_web(self, system, user, schema, name, max_searches=3, submit=None, on_event=None):
            return super().complete_web(system, user, schema, name, max_searches, submit)

    seen = []
    KnowledgeGenerator(Streaming(GOOD), None, checker=Checker()).generate(A, B, CID, on_event=lambda e: seen.append(e["type"]))
    assert seen[-2:] == ["validating", "checking"]


LEAK = "결국 B에 나온 세 가지 환경은 동기화가 얼마나 자주 일어나느냐에 따라 고를 수 있는 선택지예요."
LEAK_PROBLEM = "결론이 입력 문장을 'A'·'B' 같은 라벨로 가리켜 혼자 읽히지 않는다"


def test_the_prompt_tells_the_model_not_to_call_the_inputs_a_or_b():
    from app.services.links.generate import SYSTEM
    assert "혼자 읽혀야" in SYSTEM and "라벨" in SYSTEM


def test_results_stored_under_the_previous_prompt_are_still_used(tmp_path):
    """prompt 를 한 줄 고쳤다고 저장해 둔 카드를 모두 다시 만들지 않는다 (라벨 문제는 확인 단계가 잡는다)."""
    import hashlib, json
    from app.services.links.generate import COMPATIBLE
    old = KnowledgeGenerator(Fake(GOOD), tmp_path)
    old.prompt_version = COMPATIBLE[0]
    old.generate(A, B, CID)                                                  # 이전 prompt 로 저장
    fresh = Fake({**GOOD, "conclusion": "다시 만들면 이 결론이 나와서 달라져요. " + GOOD["conclusion"]})
    g = KnowledgeGenerator(fresh, tmp_path)
    r = g.generate(A, B, CID)
    assert r["cached"] is True and fresh.calls == 0 and r["conclusion"] == GOOD["conclusion"]
    assert len(list(tmp_path.glob("*.json"))) == 1                          # 새 prompt 의 항목을 따로 만들지 않았다
    other = KnowledgeGenerator(Fake(GOOD), tmp_path / "empty")
    assert other.generate(A, B, CID)["cached"] is False and len(list((tmp_path / "empty").glob("*.json"))) == 1


def test_a_card_held_for_pointing_at_a_or_b_is_not_kept_in_the_cache_so_it_can_be_made_again(tmp_path):
    fake = Fake({**GOOD, "conclusion": LEAK + " 그래서 같은 일도 다르게 풀 수 있다는 새 설명이 나와요 정말로."})
    g = KnowledgeGenerator(fake, tmp_path, checker=Checker({"ok": False, "problems": [LEAK_PROBLEM], "reason": "B에 나온"}))
    r = g.generate(A, B, CID)
    assert r["status"] == "held" and r["reasons"][0] == LEAK_PROBLEM and fake.calls == 1
    assert list(tmp_path.glob("*.json")) == []                               # 저장하지 않는다
    r2 = g.generate(A, B, CID)
    assert fake.calls == 2 and r2["cached"] is False                         # 다시 누르면 다시 만든다 (같은 결과로 굳지 않는다)
    other = KnowledgeGenerator(Fake(GOOD), tmp_path, checker=Checker({"ok": False, "problems": ["결론이 두 노트 문장의 주제를 벗어났다"]}))
    other.generate(A, B, CID)
    assert len(list(tmp_path.glob("*.json"))) == 1                           # 다른 이유로 보류된 것은 지금처럼 저장한다


def test_reach_is_shown_on_the_card_and_never_changes_its_status():
    for reach in ("derived", "extended", "hop"):
        r = verified(GOOD, Checker({"reach": reach}))
        assert r["status"] == "candidate" and r["check"]["reach"] == reach and r["concept"] is not None
