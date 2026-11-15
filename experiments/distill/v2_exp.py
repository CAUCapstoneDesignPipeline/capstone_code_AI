"""후보 생성 1판(v1)과 2판(v2, 이름 속 기호를 잇는다)으로 학생을 학습·채점해 견준다. 바꾸는 것은 후보 생성기뿐이다.

  uv run --env-file .env python experiments/distill/v2_exp.py

학습 자료(과목 노트 100 + 공개 글 98)·교사 라벨·트리 설정·채점은 nn_exp.py 와 같고, 임계값은 dev F1 로 고르며 3 시드 평균을 낸다.
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

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import rows, suppress_nested
    from experiments.distill.features import teacher_concepts
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    notes = []
    for _, text in [*teacher_notes().items(), *web.items()]:
        g = teacher_concepts(text)
        if g is not None:
            notes.append((text, frozenset().union(*[k for _, k in g]) if g else frozenset()))
    data = sets_available(["dev", "test", "unseen"])
    grid = np.arange(0.15, 0.81, 0.05)

    def run(symbols):
        t0 = time.time()
        Xs, ys, n_pos_notes = [], [], 0
        for text, forms in notes:
            t2, rs = rows(ex, text, symbols=symbols)
            Xs += [r["x"] for r in rs]
            ys += [int(bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms)) for r in rs]
        X, y = np.array(Xs, float), np.array(ys)
        prepared = {}
        for name, (gold, texts) in data.items():
            per = {}
            for n in sorted(gold):
                t2, rs = rows(ex, texts[n], symbols=symbols)
                per[n] = (t2, rs, np.array([r["x"] for r in rs], float) if rs else np.zeros((0, X.shape[1])))
            prepared[name] = (gold, per)

        def evaluate(name, fn, thr):
            gold, per = prepared[name]
            sc = {}
            for n, (t2, rs, x) in per.items():
                ext = []
                if rs:
                    p = fn(x)
                    for i in suppress_nested(rs, p, thr):
                        r = rs[i]
                        ext.append(Extracted(r["surface"], frozenset({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"])}), float(p[i])))
                sc[n] = score(gold[n], ext)
            R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
            P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
            return R, P, 2 * R * P / (R + P + 1e-9)

        outs = []
        for seed in range(3):
            m = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=30, l2_regularization=1.0, random_state=seed).fit(X, y)
            fn = lambda x, m=m: m.predict_proba(x)[:, 1]
            thr = max(grid, key=lambda t: evaluate("dev", fn, t)[2])
            outs.append((thr, {k: evaluate(k, fn, thr) for k in ("dev", "test", "unseen")}))
        cell = lambda k: "{:5.1%}/{:5.1%} ({:.3f})".format(*[np.mean([o[1][k][i] for o in outs]) for i in range(3)])
        print(f"{'v2 (symbols)' if symbols else 'v1 (shipped)':14s} {len(y):7d} {y.mean():5.1%}  {np.mean([o[0] for o in outs]):5.2f}   {cell('dev')}   {cell('test')}   {cell('unseen')}   ({time.time() - t0:.0f}s)", flush=True)
        return outs

    print(f"{'generator':14s} {'rows':>7s} {'pos':>5s}  {'thr':>5s}   dev R/P (F1)         test R/P (F1)        new-domain R/P (F1)")
    a, b = run(False), run(True)
    for k in ("dev", "test", "unseen"):
        d = [b[i][1][k][2] - a[i][1][k][2] for i in range(3)]
        print(f"  F1 change v2 - v1 on {k}: " + ", ".join(f"{x:+.3f}" for x in d) + f"  (mean {np.mean(d):+.3f})")


if __name__ == "__main__":
    main()
