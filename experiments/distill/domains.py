"""분야 일반화 실험: 학생 모델이 본 적 없는 분야에서 얼마나 교사(LLM 만으로 뽑은 개념)를 따라가는가.

  uv run --env-file .env python experiments/distill/domains.py [--variants course,course+web] [--drop corpus,lex]

자료(모두 정답 없이 교사 라벨만 쓴다):
  course   정답 없는 과목 노트 100개 (분야: OS · DB · 네트워크 · 컴파일러 · AI · 알고리즘 …)
  web      data/private/notes-web 의 공개 글 (분야 12개 × 영어·한국어)
평가 1 — 분야 하나씩 빼고 학습(leave-one-domain-out): 뺀 분야의 글에서 교사 개념을 얼마나 찾고(재현율) 뽑은 것이 얼마나 교사와 맞는가(정밀도).
평가 2 — 정답(AI 초안) 노트 dev · test · unseen(논문 리뷰): 학습에 web 을 더하면 달라지는가.
임계값은 학습 분야들의 교사 F1 이 가장 높은 값으로 고른다 (평가 분야는 보지 않는다).
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def f1(r, p):
    return 2 * r * p / (r + p + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", default="", help="corpus(df·idf·old_score) lex(교사 사전) 를 뺀다")
    ap.add_argument("--no-lodo", action="store_true")
    ap.add_argument("--scaling", action="store_true", help="학습에 쓰는 web 분야 수(0·3·6·11)에 따른 뺀 분야의 재현율을 잰다")
    ap.add_argument("--no-gold", action="store_true")
    ap.add_argument("--df-from", default="all", choices=["course", "all"], help="말뭉치 문서 빈도를 만들 글: 과목 노트만 / 과목+web (라벨 없는 통계)")
    args = ap.parse_args()

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from experiments.distill.candidates import generate
    from experiments.distill.features import FEATURES, Lexicon, note_rows, old_scores, teacher_forms
    from experiments.distill.student import fit_model, nms
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    course_all = {nfc(p.stem): nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes").glob("*.md"))}
    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    df = defaultdict(int)
    corpus = dict(course_all) if args.df_from == "course" else {**course_all, **web}
    for t in corpus.values():
        for k in generate(ex, t)[1]:
            df[k] += 1
    n_docs = len(corpus)
    print(f"df over {n_docs} docs ({args.df_from}), {len(df)} keys", flush=True)

    def build(name, text, forms):
        text, rows = note_rows(ex, text, df, n_docs, old_scores(ex, name, text))
        ys = [bool({loose(r["surface"]), *(loose(text[s:e]) for s, e in r["spans"][:3])} & forms) for r in rows]
        return dict(name=name, text=text, rows=rows, y=ys, forms=forms)

    notes = []                                                     # (분야, 노트)
    for n, t in teacher_notes().items():
        f = teacher_forms(t)
        if f is not None:
            notes.append(("course", build(n, t, f)))
    miss = 0
    for n, t in web.items():
        f = teacher_forms(t)
        if f is None:
            miss += 1
            continue
        notes.append((n.split("__")[0], build(n, t, f)))
    print(f"teacher-labelled notes: course {sum(d == 'course' for d, _ in notes)}, web {sum(d != 'course' for d, _ in notes)} (web without teacher labels: {miss})", flush=True)

    drop = set()
    for g in filter(None, args.drop.split(",")):
        drop |= {"corpus": {"log_df", "idf", "old_score"}, "lex": {f"lex{i}" for i in range(6)}}[g]
    names = FEATURES[:-6] + [f"lex{i}" for i in range(6)]
    keep_cols = [i for i, nme in enumerate(names) if nme not in drop]

    def train(items):
        lex = Lexicon()
        for _, d in items:
            lex.add(d["rows"], d["y"])
        lex.prior = sum(sum(d["y"]) for _, d in items) / max(1, sum(len(d["y"]) for _, d in items))
        X = np.array([r["x"] + lex.feats(r, y) for _, d in items for r, y in zip(d["rows"], d["y"])], float)[:, keep_cols]
        y = np.array([int(v) for _, d in items for v in d["y"]])
        return fit_model(X, y), lex

    def probs(model, lex, d):
        return model.predict_proba(np.array([r["x"] + lex.feats(r) for r in d["rows"]], float)[:, keep_cols])[:, 1]

    def vs_teacher(model, lex, items, thr):
        hit = tot = tp = kept = 0
        for _, d in items:
            p = probs(model, lex, d)
            idx = nms(d["rows"], p, thr)
            got = set()
            for i in idx:
                r = d["rows"][i]
                got |= {loose(r["surface"]), *(loose(d["text"][s:e]) for s, e in r["spans"])}
            hit += sum(bool(f & got) if isinstance(f, frozenset) else f in got for f in [frozenset([x]) for x in d["forms"]])
            tot += len(d["forms"])
            tp += sum(d["y"][i] for i in idx)
            kept += len(idx)
        return hit / max(1, tot), tp / max(1, kept), kept

    def pick_threshold(model, lex, items):
        grid = [round(x, 2) for x in np.arange(0.1, 0.91, 0.1)]
        return max(grid, key=lambda t: f1(*vs_teacher(model, lex, items, t)[:2]))

    domains = sorted({d for d, _ in notes if d != "course"})
    if args.scaling:
        import random
        print("\n== 학습 분야 수에 따른 재현율 (뺀 분야: 6개, 학습 분야는 무작위로 k 개) ==")
        held = domains[::2]
        for k in (0, 3, 6, 11):
            agg = [0, 0, 0, 0]
            for dom in held:
                rng = random.Random(f"{dom}-{k}")
                others = rng.sample([d for d in domains if d != dom], min(k, len(domains) - 1))
                pool = [(d, x) for d, x in notes if d == "course" or d in others]
                model, lex = train(pool)
                thr = pick_threshold(model, lex, pool[:: max(1, len(pool) // 40)])
                test = [(d, x) for d, x in notes if d == dom]
                r, p, kk = vs_teacher(model, lex, test, thr)
                tot = sum(len(x["forms"]) for _, x in test)
                for i, v in enumerate((r * tot, tot, p * kk, kk)):
                    agg[i] += v
            print(f"  web domains in training: {k:2d}  recall {agg[0]/agg[1]:.1%}  precision {agg[2]/agg[3]:.1%}", flush=True)
        return
    if not args.no_lodo and domains:
        print("\n== 분야 하나씩 빼고 학습 → 뺀 분야 (교사 개념을 얼마나 찾고, 뽑은 것이 얼마나 교사와 맞는가) ==")
        print(f"{'held-out domain':16s} {'course only':>22s} {'course + other web domains':>30s}")
        agg = defaultdict(lambda: [0, 0, 0, 0])
        for dom in domains:
            test = [(d, x) for d, x in notes if d == dom]
            row = []
            for label, pool in (("course", [(d, x) for d, x in notes if d == "course"]), ("all", [(d, x) for d, x in notes if d != dom])):
                model, lex = train(pool)
                thr = pick_threshold(model, lex, pool[:: max(1, len(pool) // 40)])
                r, p, k = vs_teacher(model, lex, test, thr)
                row.append((r, p, thr))
                for i, v in enumerate((r * sum(len(x["forms"]) for _, x in test), sum(len(x["forms"]) for _, x in test), p * k, k)):
                    agg[label][i] += v
            print(f"{dom:16s}   R {row[0][0]:.0%} P {row[0][1]:.0%} (t={row[0][2]})   R {row[1][0]:.0%} P {row[1][1]:.0%} (t={row[1][2]})", flush=True)
        for label in ("course", "all"):
            a = agg[label]
            print(f"{'micro-average':16s} {label:>8s}: recall {a[0]/a[1]:.1%}  precision {a[2]/a[3]:.1%}")

    # --- 평가 2: 정답(AI 초안) 노트
    if args.no_gold:
        return
    print("\n== 정답 노트 (AI 초안): 학습 자료에 따라 ==")
    data = sets_available(["dev", "test", "unseen"])
    prepared = {n_: {n: note_rows(ex, tx[n], df, n_docs, old_scores(ex, n, tx[n])) for n in sorted(g)} for n_, (g, tx) in data.items()}
    for label, pool in (("course", [(d, x) for d, x in notes if d == "course"]), ("course + web", notes)):
        model, lex = train(pool)
        thr = pick_threshold(model, lex, pool[:: max(1, len(pool) // 40)])
        out = []
        for name, (gold, texts) in data.items():
            sc = {}
            for n, (text, rows) in prepared[name].items():
                p = model.predict_proba(np.array([r["x"] + lex.feats(r) for r in rows], float)[:, keep_cols])[:, 1]
                idx = nms(rows, p, thr)
                ext = [Extracted(rows[i]["surface"], frozenset({loose(rows[i]["surface"]), *(loose(text[s:e]) for s, e in rows[i]["spans"])}), float(p[i])) for i in idx]
                sc[n] = score(gold[n], ext)
            R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
            P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
            out.append(f"{name} {R:.1%}/{P:.1%}")
        print(f"  {label:14s} thr {thr}: " + " | ".join(out), flush=True)


if __name__ == "__main__":
    main()
