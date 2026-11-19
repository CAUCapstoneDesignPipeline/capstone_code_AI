"""내보낸 트리가 sklearn 과 같은 값을 낸다 (다중 클래스 포함)."""

import numpy as np
import pytest

pytest.importorskip("sklearn")


def test_exported_forests_reproduce_sklearn():
    from sklearn.ensemble import HistGradientBoostingClassifier

    from app.services.relations.pairs import Forest
    from experiments.relations.train_pair import export_forest
    rng = np.random.default_rng(0)
    X = rng.random((400, 6))
    y = (X[:, 0] + X[:, 1] > 1).astype(int) + (X[:, 2] > 0.7)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=20, random_state=0).fit(X, y)
    raw = np.stack([(lambda e: Forest(e["trees"], e["baseline"], e["max_depth"]).raw(X))(export_forest(m, k)) for k in range(3)], axis=1)
    assert (m.classes_[raw.argmax(1)] == m.predict(X)).all()
    b = HistGradientBoostingClassifier(max_depth=3, max_iter=20, random_state=0).fit(X, y > 0)
    e = export_forest(b)
    p = 1 / (1 + np.exp(-Forest(e["trees"], e["baseline"], e["max_depth"]).raw(X)))
    assert np.abs(p - b.predict_proba(X)[:, 1]).max() < 1e-9
