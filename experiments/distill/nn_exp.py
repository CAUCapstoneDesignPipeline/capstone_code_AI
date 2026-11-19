"""작은 신경망(MLP)이 같은 62개 특징 위에서 부스팅 트리보다 나은지 잰다 (출고 학생과 같은 학습 자료·같은 채점).

  uv run --env-file .env python experiments/distill/nn_exp.py

임계값은 dev 의 F1 로만 고르고, test 와 새 분야는 그 값에서 한 번만 잰다. 5개 시드의 평균을 낸다.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    argparse.ArgumentParser().parse_args()
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import rows, suppress_nested
    from experiments.distill.features import teacher_concepts
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    Xs, ys = [], []
    for _, text in [*teacher_notes().items(), *web.items()]:
        g = teacher_concepts(text)
        if g is None:
            continue
        forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
        t2, rs = rows(ex, text)
        Xs += [r["x"] for r in rs]
        ys += [int(bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms)) for r in rs]
    X, y = np.array(Xs, float), np.array(ys)
    print(f"training: {len(y)} candidates, {y.mean():.1%} positive", flush=True)

    data = sets_available(["dev", "test", "unseen"])
    prepared = {}
    for name, (gold, texts) in data.items():
        per = {}
        for n in sorted(gold):
            t2, rs = rows(ex, texts[n])
            per[n] = (t2, rs, np.array([r["x"] for r in rs], float) if rs else np.zeros((0, X.shape[1])))
        prepared[name] = (gold, per)

    def evaluate(name, prob_fn, thr):
        gold, per = prepared[name]
        sc = {}
        for n, (t2, rs, x) in per.items():
            ext = []
            if rs:
                p = prob_fn(x)
                for i in suppress_nested(rs, p, thr):
                    r = rs[i]
                    ext.append(Extracted(r["surface"], frozenset({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"])}), float(p[i])))
            sc[n] = score(gold[n], ext)
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        return R, P, 2 * R * P / (R + P + 1e-9)

    def report(label, prob_fn):
        grid = np.arange(0.15, 0.81, 0.05)
        thr = max(grid, key=lambda t: evaluate("dev", prob_fn, t)[2])
        out = {k: evaluate(k, prob_fn, thr) for k in ("dev", "test", "unseen")}
        return thr, out

    def make_gbt(seed):
        m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=30, l2_regularization=1.0, random_state=seed).fit(X, y)
        return lambda x: m.predict_proba(x)[:, 1]

    scaler = StandardScaler().fit(np.sign(X) * np.log1p(np.abs(X)))

    def tf(x):
        return scaler.transform(np.sign(x) * np.log1p(np.abs(x)))

    def make_mlp(seed, hidden=(64, 32), alpha=1e-3):
        m = MLPClassifier(hidden, alpha=alpha, early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, max_iter=200, random_state=seed).fit(tf(X), y)
        return lambda x: m.predict_proba(tf(x))[:, 1], m

    variants = {
        "boosted trees (shipped settings)": lambda s: make_gbt(s),
        "MLP 64-32": lambda s: make_mlp(s)[0],
        "MLP 128-64-32": lambda s: make_mlp(s, (128, 64, 32))[0],
        "MLP 32": lambda s: make_mlp(s, (32,))[0],
        "MLP 64-32 + trees average": lambda s: (lambda a, b: (lambda x: (a(x) + b(x)) / 2))(make_gbt(s), make_mlp(s)[0]),
    }
    print(f"\n{'model':34s} {'thr':>5s}   dev R/P (F1)         test R/P (F1)        new-domain R/P (F1)   fit s")
    for label, build in variants.items():
        rows_out = []
        t0 = time.time()
        for seed in range(3):
            fn = build(seed)
            rows_out.append(report(label, fn))
        thr = np.mean([r[0] for r in rows_out])
        cell = lambda k: "{:5.1%}/{:5.1%} ({:.3f})".format(*[np.mean([r[1][k][i] for r in rows_out]) for i in range(3)])
        print(f"{label:34s} {thr:5.2f}   {cell('dev')}   {cell('test')}   {cell('unseen')}   {(time.time() - t0) / 3:5.0f}", flush=True)


if __name__ == "__main__":
    main()
