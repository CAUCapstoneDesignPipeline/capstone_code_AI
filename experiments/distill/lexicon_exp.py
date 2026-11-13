"""공개 어휘집 실험: 공개 글(위키백과)에서 교사(LLM)가 개념으로 뽑은 정도를 센 사전(키·첫 낱말·끝 낱말별 비율)을 특징으로 더하면 학생이 나아지는가.

  uv run --env-file .env python experiments/distill/lexicon_exp.py

세 변형을 같은 정답 노트(dev · test · 다른 분야 논문 리뷰)에서 견준다. 후보 생성은 모두 2판(이름 속 기호를 잇는다).
  A  과목 노트 100 + 공개 글 98 로 학습 (출고한 student-1 과 같은 자료)
  B  A + 어휘집용 공개 글 (자료만 늘린 효과)
  C  B + 공개 어휘집 특징 6개 (공개 글에서만 센 사전; 학습 줄의 공개 글은 자기 몫을 뺀다). 사전은 공개 글에서만 세므로 개인 노트의 낱말이 들어가지 않는다.
임계값은 dev 정답 노트의 F1 이 가장 높은 값으로 고른다 (test · 논문 리뷰는 그 값으로 한 번만 잰다).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-lex", type=int, default=0, help="어휘집용 글 수 상한 (0 = 전부)")
    ap.add_argument("--min-seen", type=int, default=1, help="사전에서 이 횟수 미만으로 나온 항목은 뺀다 (파일 크기를 줄인다)")
    ap.add_argument("--only-c", action="store_true", help="변형 C 만 돌린다")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from app.services.concepts.student import FEATURES, rows, suppress_nested
    from experiments.distill.features import Lexicon, teacher_concepts
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))

    def load(folder):
        return {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private" / folder).glob("*.md"))}

    pools = {"course": teacher_notes(), "web": load("notes-web"), "lex": load("notes-lex")}
    if args.max_lex:
        pools["lex"] = dict(list(pools["lex"].items())[:args.max_lex])
    docs = {}
    t0 = time.time()
    for pool, items in pools.items():
        docs[pool] = []
        for name, text in items.items():
            g = teacher_concepts(text)
            if g is None:
                continue
            forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
            t2, rs = rows(ex, text, symbols=True)
            ys = [bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms) for r in rs]
            docs[pool].append((rs, ys))
        print(f"{pool}: {len(docs[pool])} labelled notes, {sum(len(r) for r, _ in docs[pool])} rows ({time.time() - t0:.0f}s)", flush=True)

    public = docs["web"] + docs["lex"]
    lex = Lexicon()
    for rs, ys in public:
        lex.add(rs, ys)
    lex.prior = sum(sum(y) for _, y in public) / max(1, sum(len(y) for _, y in public))
    if args.min_seen > 1:
        for table in (lex.k, lex.f, lex.l):
            for k in [k for k, (pos, seen) in table.items() if seen < args.min_seen]:
                del table[k]
    print(f"public lexicon (min seen {args.min_seen}): {len(lex.k)} phrases, {len(lex.f)} first words, {len(lex.l)} last words (prior {lex.prior:.3f})", flush=True)

    def matrix(items, with_lex, own):
        out = []
        for rs, ys in items:
            for r, y in zip(rs, ys):
                out.append(r["x"] + (lex.feats(r, y if own else None) if with_lex else []))
        return np.array(out, float), np.array([int(v) for _, ys in items for v in ys])

    def fit(X, y):
        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=30, l2_regularization=1.0, random_state=0).fit(X, y)

    data = sets_available(["dev", "test", "unseen"])
    prepared = {}
    for name, (gold, texts) in data.items():
        prepared[name] = {n: rows(ex, texts[n], symbols=True) for n in sorted(gold)}

    def run(model, with_lex):
        probs = {}
        for name, per in prepared.items():
            probs[name] = {}
            for n, (text, rs) in per.items():
                X = np.array([r["x"] + (lex.feats(r) if with_lex else []) for r in rs], float)
                probs[name][n] = model.predict_proba(X)[:, 1]

        def at(name, thr):
            gold, _ = data[name]
            sc = {}
            for n, (text, rs) in prepared[name].items():
                p = probs[name][n]
                idx = suppress_nested(rs, p, thr)
                sc[n] = score(gold[n], [Extracted(rs[i]["surface"], frozenset({loose(rs[i]["surface"]), *(loose(text[s:e]) for s, e in rs[i]["spans"])}), float(p[i])) for i in idx])
            R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
            P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
            return R, P
        grid = [round(x, 2) for x in np.arange(0.2, 0.76, 0.05)]
        f1 = lambda r, p: 2 * r * p / (r + p + 1e-9)
        best = max(grid, key=lambda t: f1(*at("dev", t)))
        return best, {n: at(n, best) for n in data}, {t: at("dev", t) for t in grid}

    print(f"\n{'variant':44s} thr   dev R/P (F1)       test R/P (F1)      paper-reviews R/P (F1)")
    for label, pool_names, with_lex in (("A  course+web (as shipped data)", ["course", "web"], False),
                                         ("B  A + lexicon-corpus notes (more data)", ["course", "web", "lex"], False),
                                         ("C  B + public lexicon features", ["course", "web", "lex"], True)):
        if args.only_c and not with_lex:
            continue
        items = [d for p in pool_names for d in docs[p]]
        # 공개 글(web, lex)은 사전에서 자기 몫을 빼고, 과목 노트는 사전에 없으므로 그대로
        parts = [matrix(docs[p], with_lex, own=p in ("web", "lex")) for p in pool_names if docs[p]]
        X = np.vstack([a for a, _ in parts])
        y = np.concatenate([b for _, b in parts])
        t1 = time.time()
        model = fit(X, y)
        thr, res, curve = run(model, with_lex)
        f1 = lambda r, p: 2 * r * p / (r + p + 1e-9)
        print(f"{label:44s} {thr:.2f}  " + "   ".join(f"{res[k][0]:.1%}/{res[k][1]:.1%} ({f1(*res[k]):.2f})" for k in ("dev", "test", "unseen")) + f"   [{len(y)} rows, {time.time() - t1:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
