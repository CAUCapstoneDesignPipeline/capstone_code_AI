"""평가 묶음: 신뢰 구간, dev/test 나누기, 보고서 비교, 사람 판정 표."""

import pytest

from app.services.evaluation import (Estimate, assign_splits, bootstrap, compare, labelled_explanations, ratio)


def test_bootstrap_is_deterministic_and_brackets_value():
    units = [1, 0, 1, 1, 0, 1, 1, 1]
    mean = lambda us: sum(us) / len(us)
    a, b = bootstrap(units, mean), bootstrap(units, mean)
    assert a == b and a.lo <= a.value <= a.hi and a.n == 8
    assert bootstrap([], mean) == Estimate(0.0, 0.0, 0.0, 0)
    pooled = ratio(lambda u: u[0], lambda u: u[1])
    assert pooled([(1, 2), (3, 4)]) == pytest.approx(4 / 6)       # 노트 평균이 아니라 개수를 모은 비율


def test_splits_keep_share_and_respect_overrides():
    notes = [f"note-{i}" for i in range(10)]
    s = assign_splits(notes)
    assert sum(v == "test" for v in s.values()) == 4 and assign_splits(list(reversed(notes))) == s
    o = assign_splits(notes, {"note-0": "test", "note-1": "dev", "note-2": ""})
    assert o["note-0"] == "test" and o["note-1"] == "dev" and sum(v == "test" for v in o.values()) == 4


def test_compare_flags_change_outside_previous_interval():
    old = {"a": {"value": 0.8, "lo": 0.7, "hi": 0.9}, "b": {"value": 0.5, "lo": 0.4, "hi": 0.6},
           "c": {"value": 0.5, "lo": 0.4, "hi": 0.6}}
    new = {"a": {"value": 0.6}, "b": {"value": 0.55}, "c": {"value": 0.7}, "d": {"value": 1.0}}
    assert [(k, v) for k, _, _, v in compare(new, old)] == [("a", "나빠짐"), ("b", "비슷함"), ("c", "좋아짐")]


def test_explanation_labels_join_blind_sheet_and_key():
    sheet = [{"id": "x1", "human": "valid"}, {"id": "x2", "human": ""}, {"id": "x3", "human": "invalid"}]
    key = [{"id": "x1", "system": "candidate"}, {"id": "x2", "system": "hypothesis"}, {"id": "x3", "system": "candidate"}]
    assert labelled_explanations(sheet, key) == [("valid", "valid"), ("valid", "invalid")]
    with pytest.raises(ValueError):
        labelled_explanations([{"id": "x9", "human": "valid"}], key)
