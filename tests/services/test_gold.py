"""정답 개념 채점 (docs/eval/gold-concepts.md)."""

from app.services.concepts.gold import Extracted, load_gold, loose, score


def test_load_gold_keeps_reviewed_unless_draft_included(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("note_id,concept,level,status\n"
                 "n,WAL|Write Ahead Logging,core,reviewed\n"
                 "n,체크포인트,mentioned,reviewed\n"
                 "n,저널링,core,draft\n", encoding="utf-8")
    assert [g.label for g in load_gold(p)["n"]] == ["WAL", "체크포인트"]
    assert len(load_gold(p, include_draft=True)["n"]) == 3


def test_score_matches_any_alias_and_counts_precision(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("note_id,concept,level,status\n"
                 "n,WAL|Write Ahead Logging,core,reviewed\n"
                 "n,페이지 캐시,core,reviewed\n"
                 "n,fsync,mentioned,reviewed\n", encoding="utf-8")
    gold = load_gold(p)["n"]
    ex = lambda label, *more: Extracted(label, frozenset(loose(x) for x in (label, *more)), 1.0)
    s = score(gold, [ex("write-ahead logging"), ex("fsync"), ex("디스크")])   # 대소문자·하이픈 무시, 별칭으로 맞음
    assert (s.core_hit, s.core, s.all_hit, s.all) == (1, 2, 2, 3)
    assert (s.extracted_hit, s.extracted) == (2, 3) and s.missed == ["페이지 캐시"] and s.wrong == ["디스크"]
    assert score(gold, [ex("페이지")]).core_hit == 0                     # 부분 일치는 틀림
