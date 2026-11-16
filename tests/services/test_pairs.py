"""빠른 관계 추출 (pairs-1): 쌍 특징, numpy 트리, Decisions 질문, 관계·근거 (가짜 Decisions 와 손으로 만든 작은 모델)."""

import numpy as np
import pytest
from fake_decisions import FakePairs, stump, tiny_model

from app.services.concepts import Mention, Method, Span
from app.services.relations import Limits
from app.services.relations import pairs as P
from app.services.relations.candidates import candidate_pairs

TEXT = "# 로그\n\nWAL은 로그 기법이다. 체크포인트는 복구를 빠르게 한다.\n\n- WAL\n- 체크포인트\n"
POLICY = [("is_a", "종류", "x"), ("part_of", "부분", "x"), ("contrasts_with", "대비", "x")]


def mentions(text, keys):
    out = {}
    for k in keys:
        start = 0
        while (i := text.find(k, start)) >= 0:
            out.setdefault(k, []).append(Mention(span=Span("n", 1, i, i + len(k)), surface=k, key=k, method=Method.TFIDF, method_version="t", score=1.0))
            start = i + len(k)
    return out


def relations(client, **kw):
    pr = P.PairRelations(P.DecisionsPairs(client, "fake"), tiny_model(**kw))
    return pr.extract(TEXT, mentions(TEXT, ["WAL", "로그", "체크포인트", "복구"]), POLICY, Limits())


def test_a_pair_the_judge_calls_a_relation_becomes_a_relation_with_the_sentence_as_evidence():
    client = FakePairs({("로그", "WAL"): {"none": 0.1, "is_a_ba": 0.9}})
    evidence, rels = relations(client)
    assert [(r.source, r.code, r.target) for r in rels] == [("WAL", "is_a", "로그")]          # is_a_ba: 두 번째(b)가 주어
    assert rels[0].confidence >= 0.5 and rels[0].method == "pairs"
    assert [evidence[i].quote for i in rels[0].evidence] == ["WAL은 로그 기법이다."]
    assert all(TEXT[e.start16:e.end16] == e.quote for e in evidence)


def test_candidates_not_called_a_relation_are_dropped_and_codes_outside_the_policy_are_skipped():
    assert relations(FakePairs())[1] == []
    assert relations(FakePairs(default={"none": 0.1, "is_a_ab": 0.9}), threshold=0.99)[1] == []     # 확률이 임계값 아래
    only_part = P.PairRelations(P.DecisionsPairs(FakePairs(default={"none": 0.1, "is_a_ab": 0.9}), "fake"), tiny_model())
    assert only_part.extract(TEXT, mentions(TEXT, ["WAL", "로그"]), [("part_of", "부분", "x")], Limits())[1] == []     # 정책에 is_a 가 없다


def test_refused_answers_do_not_break_the_note():
    client = FakePairs(default={"none": 0.1, "is_a_ab": 0.9}, refuse=[("WAL", "로그")])
    _, rels = relations(client)
    assert rels                                                    # 거절된 쌍은 빈 확률로 다뤄져도 다른 쌍은 그대로 간다


def test_the_pair_count_is_capped_and_oversized_requests_are_split(monkeypatch):
    monkeypatch.setattr(P, "MAX_PAIRS", 2)
    client = FakePairs()
    relations(client)
    assert len(client.calls) == 1 and len(client.calls[0][2]) == 2
    monkeypatch.setattr(P, "MAX_PAIRS", 10)
    monkeypatch.setattr(P, "MAX_QUESTIONS", 2)
    client = FakePairs()
    relations(client)
    n = len(candidate_pairs(TEXT, mentions(TEXT, ["WAL", "로그", "체크포인트", "복구"])))
    assert len(client.calls) == -(-n // 2) and all(len(c[2]) <= 2 for c in client.calls)


def test_no_pairs_means_no_request():
    client = FakePairs()
    pr = P.PairRelations(P.DecisionsPairs(client, "fake"), tiny_model())
    assert pr.extract("하나만 있다", mentions("하나만 있다", ["하나만"]), POLICY, Limits()) == ([], [])
    assert client.calls == []


def test_features_have_the_declared_width_and_do_not_depend_on_the_judge():
    ms = mentions(TEXT, ["WAL", "로그", "체크포인트"])
    pairs = candidate_pairs(TEXT, ms)
    X = P.structure_features(TEXT, P.positions(TEXT, ms), pairs)
    assert X.shape == (len(pairs), P.N_STRUCT) and len(P.FEATURES) == P.N_STRUCT + 1 + len(P.CLASSES)
    D = P.decision_features(pairs, {(a, b): {"is_a_ab": 0.7, "none": 0.2} for a, b, _ in pairs})
    assert D.shape == (len(pairs), 1 + len(P.CLASSES)) and np.allclose(D[:, 0], 0.8)


def test_the_forest_matches_a_hand_computed_value():
    f = P.Forest([stump(1, -2.0, 3.0), stump(0, 0.5, -1.0)], baseline=0.25, depth=1)
    assert f.raw(np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])).tolist() == [0.25 - 2.0 + 0.5, 0.25 + 3.0 - 1.0, 0.25 + 3.0 + 0.5]


def test_the_shipped_model_loads_and_is_deterministic():
    m = P.PairModel.load()
    assert m.id == "pairs-1" and 0 < m.threshold < 1 and m.version.startswith("pairs-1+") and m.trained["notes"] == 59
    X = np.random.default_rng(0).random((20, len(P.FEATURES)))
    p, t = m.predict(X)
    assert ((0 <= p) & (p <= 1)).all() and set(t) <= set(range(len(P.CLASSES)))
    assert all(np.array_equal(a, b) for a, b in zip((p, t), m.predict(X)))


def test_a_model_file_whose_features_differ_from_the_code_is_refused():
    bad = {"id": "x", "features": P.FEATURES[:-1], "classes": P.CLASSES, "threshold": 0.5}
    with pytest.raises(ValueError):
        P.PairModel(bad)
