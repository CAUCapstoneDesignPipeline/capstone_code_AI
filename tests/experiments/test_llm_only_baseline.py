"""LLM 만으로 개념 뽑기 기준선: 출력 해석과 근거 확인 (LLM 없음)."""

from app.services.concepts.gold import loose
from experiments.llm_only.baseline import ground, parse

TEXT = "WAL(Write Ahead Logging)은 로그를 먼저 쓴다. 체크포인트 이후의 로그만 다시 적용한다."


def test_parse_skips_malformed_entries():
    data = {"concepts": [{"surface": " WAL ", "aliases": ["Write Ahead Logging", 3]}, {"surface": ""}, {"aliases": []}, "x", {"surface": "체크포인트", "aliases": "oops"}]}
    assert parse(data) == [("WAL", ["Write Ahead Logging"]), ("체크포인트", [])]
    assert parse({"concepts": "no"}) == [] and parse(None) == []


def test_ground_keeps_only_concepts_in_the_note_ignoring_case_and_hyphens():
    items = [("wal", ["Write-Ahead Logging"]), ("체크포인트", []), ("Raft", ["합의 알고리즘"]), ("write ahead logging", [])]
    got, dropped = ground(items, TEXT, loose)
    assert dropped == 1                                                  # Raft 는 원문에 없다
    assert [g[0] for g in got] == ["wal", "체크포인트"]                 # 같은 개념(WAL = write ahead logging)은 한 번만
    assert got[0][1] == frozenset({loose("wal"), loose("Write Ahead Logging")})


def test_an_alias_in_the_note_rescues_a_surface_that_is_not():
    got, dropped = ground([("Write Ahead Logs", ["WAL"])], TEXT, loose)
    assert dropped == 0 and got[0][0] == "WAL"


def test_latin_forms_need_word_boundaries_but_korean_forms_take_particles():
    text = "The model uses 체크포인트를 쓴다. Flow-Matching works."
    got, dropped = ground([("ode", []), ("체크포인트", []), ("flow matching", []), ("match", [])], text, loose)
    assert [g[0] for g in got] == ["체크포인트", "flow matching"] and dropped == 2
