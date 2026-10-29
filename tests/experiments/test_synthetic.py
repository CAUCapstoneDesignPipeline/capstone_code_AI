"""합성 노트 생성기의 LLM 없는 부분: 명세·정답, 명세 밖 Big-O 검사, 노트 검증."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments/synthetic"))

import build  # noqa: E402
from catalog import ALGOS, COMPONENTS  # noqa: E402


def cases(n=150, seed=0):
    rng = random.Random(seed)
    return [build.Case(i, build.pick(rng, build.SCENES), rng) for i in range(n)]


def test_catalog_roles_are_consistent():
    roles = {a["role"] for a in ALGOS.values()}
    assert roles <= {c["role"] for c in COMPONENTS.values()}               # 모든 역할에 구현이 있다
    for a in ALGOS.values():
        assert a["needs"] and a["size"] and a["extra"]


def test_expected_results_follow_the_scene():
    seen = {}
    for c in cases():
        e = c._expected()
        seen.setdefault(c.scene, []).append(e)
        if c.scene == "negative":
            assert e == []                                                   # 무관한 쌍은 아무것도 이어지지 않는다
        if c.scene == "contradiction":
            assert [x["status"] for x in e if x["rule"] == "R1"] == ["rejected"] and len(c.notes) == 3
        if c.scene == "known":
            assert all(x["status"] in ("known", "conclusion") for x in e) and len(c.notes) == 3
    assert set(seen) == {s for s, _ in build.SCENES}


def test_dijkstra_binary_indexed_heap_expectation():
    rng = random.Random(0)
    for i in range(400):
        c = build.Case(i, "conclusion", rng)
        if (c.meta["algorithm"], c.meta["component"]) == ("dijkstra", "indexed_binary_heap") and not c.meta["vars"]:
            r2 = next(x for x in c._expected() if x["rule"] == "R2")
            assert r2["status"] == "conclusion" and r2["result"] == "O((V + E) log V)" and r2["case"] == "worst"
            return
    raise AssertionError("not sampled")


def test_stray_big_o_and_note_check():
    note = {"facts": [{"id": "n#1", "claim": "", "fact": {"expr": "n+m"}},
                      {"id": "n#2", "claim": "", "fact": {"expr": "1"}}], "withheld": []}
    assert build.stray_big_o("전체 O(n + m), 연산은 $O(1)$", note) == []
    assert build.stray_big_o("전체 O(n log n)", note) == ["O(n log n)"]
    assert build.stray_big_o("전체 O(몰라)", note) == ["O(몰라)"]
    ok = {"body": "# 제목\n- 전체 O(n+m)", "quotes": [{"id": "n#1", "quote": "전체 O(n+m)"}, {"id": "n#2", "quote": "# 제목"}]}
    assert build.Writer.check(note, ok) == ""
    assert "글자 그대로" in build.Writer.check(note, {**ok, "quotes": [{"id": "n#1", "quote": "없는 구절"}, ok["quotes"][1]]})
    assert "quote 가 없는" in build.Writer.check(note, {**ok, "quotes": ok["quotes"][:1]})
    assert "첫 줄" in build.Writer.check(note, {**ok, "body": "제목 없음"})
    assert "Big-O" in build.Writer.check(note, {**ok, "body": ok["body"] + "\n- 총 O(n^3)"})


def test_held_notes_record_what_they_withhold():
    held = [c for c in cases() if c.scene == "held"]
    assert held and all(n["withheld"] for c in held for n in c.notes if any(
        "unknown" in f["claim"] for f in n["facts"]))
