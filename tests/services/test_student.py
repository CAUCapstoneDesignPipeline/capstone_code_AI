"""학생 개념 선택기(extract-student-1): 모델 파일이 코드와 맞고, 결정적이며, 응답에 실을 위치가 원문과 맞는다. 내보낸 트리가 sklearn 과 같다."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.api.v1.extract import check_output, to_concepts
from app.services.concepts import ConceptExtractor, ExtractConfig
from app.services.concepts.extract import Idf
from app.services.concepts.student import FEATURES, Student, rows, suppress_nested

NOTE = ("# 데이터베이스 로그 🧠\n\n**WAL**(Write Ahead Logging)은 데이터 페이지보다 로그를 먼저 디스크에 쓰는 규칙이다. WAL 덕분에 장애가 나도 복구할 수 있다.\n"
        "체크포인트는 복구 시간을 줄인다. Flow Matching is called a simulation-free method for the vector field.\n\n## 풀이\n\nExercise 1 에서 p.20 의 표를 본다.\n")


@pytest.fixture(scope="module")
def ex():
    return ConceptExtractor(ExtractConfig(), Idf({}, 10, "t"))


def test_the_shipped_model_matches_the_code():
    s = Student.load()
    path = Path(__file__).resolve().parents[2] / "app/services/concepts/student-1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["features"] == FEATURES and 0.0 < s.threshold < 1.0 and s.depth >= 1
    assert all(len(t["feature"]) == len(t["left"]) == len(t["right"]) == len(t["value"]) == len(t["threshold"]) for t in data["trees"])
    assert data["trained"]["teacher"] and data["trained"]["notes"] > 0
    assert s.version.startswith("extract-student-1+")


def test_a_model_for_other_features_is_refused():
    data = json.loads((Path(__file__).resolve().parents[2] / "app/services/concepts/student-1.json").read_text(encoding="utf-8"))
    data["features"] = list(reversed(data["features"]))
    with pytest.raises(ValueError):
        Student(data)


def test_rows_have_one_value_per_feature(ex):
    _, rs = rows(ex, NOTE)
    assert rs and all(len(r["x"]) == len(FEATURES) for r in rs)
    assert all(np.isfinite(np.array(r["x"], dtype=float)).all() for r in rs)


def test_select_is_deterministic_and_positions_point_at_the_original_text(ex):
    s = Student.load()
    a = s.select(ex, "n", 1, NOTE)
    b = s.select(ex, "n", 1, NOTE)
    assert a and list(a) == list(b) and all(len(ms) <= 20 for ms in a.values())
    concepts, _ = to_concepts(a, NOTE)
    check_output(NOTE, concepts, [], [], set())                   # 모든 span 이 원문 UTF-16 조각과 같다 (이모지 뒤에서도)
    assert all(0.0 <= c.confidence <= 1.0 for c in concepts)
    assert [c.label for c in concepts] == [c.label for c in to_concepts(b, NOTE)[0]]


def test_empty_and_symbol_only_notes_give_no_concepts(ex):
    s = Student.load()
    assert s.select(ex, "n", 1, "") == {} and s.select(ex, "n", 1, "---\n\n123 456\n") == {}


def test_nested_candidates_are_dropped_only_when_every_mention_is_inside_a_kept_longer_one():
    rs = [dict(surface="flow matching", spans=[(0, 13)]), dict(surface="flow", spans=[(0, 4)]), dict(surface="matching", spans=[(5, 13), (40, 48)])]
    assert suppress_nested(rs, [0.9, 0.8, 0.7], 0.5) == [0, 2]          # 'flow' 는 모두 안에 있어 빠지고, 'matching' 은 따로 나온 곳이 있어 남는다
    assert suppress_nested(rs, [0.9, 0.8, 0.7], 0.95) == []


def test_numpy_trees_equal_sklearn_on_a_fresh_model():
    sklearn = pytest.importorskip("sklearn.ensemble")
    from experiments.distill.train_student import export_trees
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, len(FEATURES)))
    y = (X[:, 0] + 0.5 * X[:, 3] * X[:, 5] + rng.normal(scale=0.3, size=600) > 0.4).astype(int)
    model = sklearn.HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=25, random_state=0).fit(X, y)
    s = Student({"id": "t", "features": FEATURES, "threshold": 0.5, **export_trees(model)})
    assert np.abs(s.probabilities(X) - model.predict_proba(X)[:, 1]).max() < 1e-9
