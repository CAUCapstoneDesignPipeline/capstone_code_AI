"""라우팅 실험의 순수 함수: AUC 와 F1 (LLM 없음)."""

import math

from experiments.distill.routing import auc, f1


def test_auc_is_one_for_perfect_order_half_for_ties_and_zero_for_reverse():
    assert auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5
    assert auc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert math.isnan(auc([1, 2], [1, 1]))                      # 한쪽 라벨만 있으면 정의되지 않는다


def test_f1_is_the_harmonic_mean():
    assert abs(f1(0.5, 0.5) - 0.5) < 1e-6
    assert abs(f1(1.0, 0.5) - 2 / 3) < 1e-6
    assert f1(0, 0) == 0
