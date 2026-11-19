"""후보 쌍 실험 도구(experiments/proposals): 사람 라벨 점수, AUC, 관계 그래프 특징."""

from types import SimpleNamespace

import numpy as np
from experiments.proposals.human_pairs import kappa, score
from experiments.proposals.relation_features import Graph, auc, pair_features


def test_auc_is_the_probability_a_useful_pair_outranks_a_not_useful_one():
    assert auc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auc([1, 0], [0.5, 0.5]) == 0.5                      # 동점은 절반
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert np.isnan(auc([1, 1], [0.1, 0.2]))                   # 한쪽뿐이면 정할 수 없다


def test_kappa_is_zero_for_chance_and_one_for_identical_labels():
    assert kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    assert kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == 0.0


def test_human_score_counts_only_marked_rows_and_compares_with_the_ai_labels():
    sheet = [{"id": str(i), "human": h} for i, h in enumerate(["useful", "vague", "useful", "unrelated", ""], 1)]
    key = [{"id": "1", "label_a": "useful", "label_b": "useful"}, {"id": "2", "label_a": "useful", "label_b": "vague"},
           {"id": "3", "label_a": "vague", "label_b": "vague"}, {"id": "4", "label_a": "unrelated", "label_b": "unrelated"},
           {"id": "5", "label_a": "useful", "label_b": "useful"}]
    r = score(sheet, key)
    assert r["labelled"] == 4 and r["total"] == 5               # 빈 칸은 점수에서 빠진다
    assert r["ai_useful_labelled"] == 2 and r["ai_useful_precision"] == 0.5     # AI 가 useful: 1·2 → 사람 useful 은 1 만
    assert r["ai_not_useful_labelled"] == 2 and r["ai_miss"] == 0.5             # AI 가 아니라 한 3·4 → 사람 useful 은 3
    assert r["human_useful"] == 2


def note(concepts, relations):
    return {"concepts": {k: {"surface": k, "mentions": [[s, s + 1]]} for k, s in concepts.items()},
            "relations": [{"source": s, "target": t, "code": c, "confidence": w, "evidence": [[e, e + 1]]} for s, t, c, w, e in relations]}


def el(note_name, start, end):
    return SimpleNamespace(note=note_name, start=start, end=end, text="x")


def test_relation_features_see_a_bridge_concept_shared_across_notes():
    # 노트 N1: heap -is_a-> tree. 노트 N2: avl -is_a-> tree. 문장 A(N1) 은 heap, 문장 B(N2) 는 avl 을 말한다 → 둘은 tree 를 거쳐 이어진다
    g = Graph({"N1": note({"heap": 0, "tree": 5}, [("heap", "tree", "is_a", 0.9, 0)]),
               "N2": note({"avl": 0, "tree": 5}, [("avl", "tree", "is_a", 0.8, 0)])})
    f = pair_features(g, el("N1", 0, 3), el("N2", 0, 3))
    assert f["direct"] == 0.0 and f["shared_nb"] == 0.8 and f["same_parent"] == 1          # 이웃은 약한 쪽 확신도(min)
    assert f["rel_in_a"] == 1 and f["rel_in_b"] == 1 and f["n_concepts_a"] == 1


def test_a_direct_relation_between_the_two_sentences_concepts_is_a_direct_feature():
    g = Graph({"N1": note({"wal": 0}, []), "N2": note({"wal": 0, "acid": 5}, [("wal", "acid", "part_of", 0.7, 0)])})
    f = pair_features(g, el("N1", 0, 3), el("N2", 4, 9))
    assert f["direct"] == 0.7
