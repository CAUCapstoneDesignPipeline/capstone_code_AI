"""폴백 라우팅: 빠른 학생 모델로 먼저 뽑고, 낯선 노트만 LLM(LLM 만으로 뽑기)에 맡긴다.

  uv run --env-file .env python experiments/distill/routing.py [--measure-llm 3]

노트마다 학생의 출력에서 정답 없이 계산하는 신호(낯선 낱말 비율, 애매한 후보 비율 …)를 만들고, 신호가 임계값 이상이면 그 노트는 LLM 결과를 쓴다.
학습: 정답 없는 과목 노트 100 + 공개 글 98 (교사 라벨). 평가: 정답(AI 초안) dev · test · unseen. 신호와 임계값은 dev 에서만 고른다.
출력: 라우팅 비율에 따른 재현율·정밀도·F1 과 노트당 평균 지연 시간(학생 측정값 + LLM 측정값).
"""

import argparse
import sys
import time
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def f1(r, p):
    return 2 * r * p / (r + p + 1e-9)


def auc(scores, labels):
    """순위 합으로 구한 AUC (라벨 1 이 점수가 높을수록 1.0)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def signals(rows, probs, text, df, in_corpus: bool, kept_idx):
    """노트 하나의 라우팅 신호 (모두 정답 없이 학생의 출력과 말뭉치 통계로 계산)."""
    near = [i for i, p in enumerate(probs) if p >= 0.1]
    unfamiliar = [i for i in near if df.get(rows[i]["key"], 0) - (1 if in_corpus else 0) <= 0]
    kept_p = [probs[i] for i in kept_idx]
    latin_only = [i for i in kept_idx if rows[i]["surface"].isascii()]
    return {
        "unfamiliar_share": len(unfamiliar) / max(1, len(near)),
        "ambiguity": float(np.mean([4 * probs[i] * (1 - probs[i]) for i in near])) if near else 0.0,
        "low_conf_share": sum(p < 0.6 for p in kept_p) / max(1, len(kept_p)),
        "mean_p_kept": float(np.mean(kept_p)) if kept_p else 0.0,
        "density": len(kept_idx) / max(1.0, len(text) / 1000),
        "latin_share": len(latin_only) / max(1, len(kept_idx)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-llm", type=int, default=0, help="캐시를 피해 LLM 호출 시간을 이 횟수만큼 잰다 (비용이 든다)")
    args = ap.parse_args()

    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.markdown import nfc
    from experiments.distill.candidates import generate
    from experiments.distill.features import FEATURES, Lexicon, note_rows, old_scores, teacher_concepts
    from experiments.distill.student import fit_model, nms
    from experiments.distill.teacher import teacher_notes
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    course_all = {nfc(p.stem): nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes").glob("*.md"))}
    web = {p.stem: nfc(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/private/notes-web").glob("*.md"))}
    corpus = {**course_all, **web}
    df = defaultdict(int)
    for t in corpus.values():
        for k in generate(ex, t)[1]:
            df[k] += 1
    n_docs = len(corpus)

    # --- 학습 (교사 라벨)
    train = []
    for name, text in [*teacher_notes().items(), *web.items()]:
        g = teacher_concepts(text)
        if g is None:
            continue
        forms = frozenset().union(*[k for _, k in g]) if g else frozenset()
        t2, rows = note_rows(ex, text, df, n_docs, old_scores(ex, name, text))
        train.append((rows, [bool({loose(r["surface"]), *(loose(t2[s:e]) for s, e in r["spans"][:3])} & forms) for r in rows]))
    lex = Lexicon()
    for rows, ys in train:
        lex.add(rows, ys)
    lex.prior = sum(sum(y) for _, y in train) / max(1, sum(len(y) for _, y in train))
    X = np.array([r["x"] + lex.feats(r, y) for rows, ys in train for r, y in zip(rows, ys)], float)
    y = np.array([int(v) for _, ys in train for v in ys])
    model = fit_model(X, y)
    thr = 0.4
    print(f"student trained on {len(train)} notes ({len(y)} rows, {y.mean():.1%} positive); keep threshold {thr}", flush=True)

    # --- 정답 노트: 학생 · LLM 의 노트별 결과와 신호
    data = sets_available(["dev", "test", "unseen"])
    per = {}                                                      # set → note → dict
    t_student = []
    for name, (gold, texts) in data.items():
        per[name] = {}
        for n in sorted(gold):
            t0 = time.perf_counter()
            text, rows = note_rows(ex, texts[n], df, n_docs, old_scores(ex, n, texts[n]))
            p = model.predict_proba(np.array([r["x"] + lex.feats(r) for r in rows], float))[:, 1]
            idx = nms(rows, p, thr)
            t_student.append((time.perf_counter() - t0) * 1000)
            stu = score(gold[n], [Extracted(rows[i]["surface"], frozenset({loose(rows[i]["surface"]), *(loose(text[s:e]) for s, e in rows[i]["spans"])}), float(p[i])) for i in idx])
            g = teacher_concepts(texts[n])
            llm = score(gold[n], [Extracted(sf, k, 1.0) for sf, k in g])
            per[name][n] = dict(stu=stu, llm=llm, sig=signals(rows, p, text, df, n in course_all, idx), chars=len(text))
    print(f"student latency per note (features + model, this machine): median {statistics.median(t_student):.0f} ms, p90 {np.percentile(t_student, 90):.0f} ms\n")

    t_llm = None
    if args.measure_llm:
        from experiments.distill.teacher import asker
        ask = asker("medium")
        ts = []
        picks = sorted(((len(tx), n, tx) for _, (g, texts) in data.items() for n, tx in texts.items()))
        mid = picks[len(picks) // 3: len(picks) // 3 + args.measure_llm]
        for _, n, tx in mid:
            t0 = time.perf_counter()
            ask(tx + "\n\n")                                       # 끝에 줄바꿈을 더해 캐시를 피한다 (같은 내용)
            ts.append(time.perf_counter() - t0)
        t_llm = statistics.median(ts)
        print(f"LLM-only call latency (uncached, {len(ts)} notes of ~{statistics.median([c for c, _, _ in mid]):.0f} chars): median {t_llm:.1f}s  ({', '.join(f'{x:.1f}' for x in ts)})\n")

    # --- 신호 고르기: dev 에서 '학생보다 LLM 이 낫다' 를 가장 잘 가르는 신호 (노트 F1 차이 > 0.1)
    def note_f1(sc):
        r = sc.core_hit / max(1, sc.core)
        pr = sc.extracted_hit / max(1, sc.extracted)
        return f1(r, pr)

    names = list(next(iter(per["dev"].values()))["sig"])
    print("signal AUC for 'LLM beats the student by >0.1 note-F1' (0.5 = useless):")
    best, best_auc = None, 0
    for s in names:
        row = []
        for st in ("dev", "test", "unseen"):
            ns = list(per[st].values())
            row.append(auc([x["sig"][s] for x in ns], [note_f1(x["llm"]) - note_f1(x["stu"]) > 0.1 for x in ns]))
        # 낮을수록 낯선 신호는 방향을 뒤집는다
        flip = row[0] < 0.5
        row = [1 - a if flip else a for a in row]
        print(f"  {s:18s} dev {row[0]:.2f}  test {row[1]:.2f}  unseen {row[2]:.2f}   ({'higher = route' if not flip else 'lower = route'})")
        if row[0] > best_auc:
            best, best_auc, best_flip = s, row[0], flip
    sign = -1 if best_flip else 1
    print(f"-> routing signal chosen on dev: {best} (route when {'low' if best_flip else 'high'})\n")

    # --- 라우팅 비율에 따른 결과
    def agg(items, route):
        c = [0, 0, 0, 0]
        n_route = 0
        for x in items:
            sc = x["llm"] if route(x) else x["stu"]
            n_route += route(x)
            c[0] += sc.core_hit; c[1] += sc.core; c[2] += sc.extracted_hit; c[3] += sc.extracted
        r, p = c[0] / max(1, c[1]), c[2] / max(1, c[3])
        return r, p, n_route / max(1, len(items))

    dev_vals = sorted(sign * x["sig"][best] for x in per["dev"].values())
    print("routing sweep (threshold from dev quantiles; same threshold applied to test/unseen):")
    print(f"  {'target':>8s} | " + " | ".join(f"{s:^34s}" for s in ("dev", "test", "unseen")))
    t_stu = statistics.median(t_student) / 1000
    for frac in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
        tau = float("inf") if frac == 0 else (-float("inf") if frac == 1 else dev_vals[min(len(dev_vals) - 1, int((1 - frac) * len(dev_vals)))])
        route = lambda x, tau=tau: sign * x["sig"][best] >= tau
        cells = []
        for st in ("dev", "test", "unseen"):
            r, p, fr = agg(list(per[st].values()), route)
            lat = f" ~{t_stu + fr * t_llm:.1f}s" if t_llm else ""
            cells.append(f"{fr:4.0%} → LLM  R {r:.0%} P {p:.0%} F1 {f1(r, p):.2f}{lat}")
        print(f"  {frac:>8.0%} | " + " | ".join(f"{c:34s}" for c in cells))
    print("  (R = core recall, P = precision vs AI-draft gold; ~Ns = expected mean seconds per note = student median + routed share × LLM median)")

    # --- 진단: 세 집합을 합쳐 신호마다 F1 곡선 (라우팅 비율 → 전체 F1), 정답을 아는 이상적인 라우팅과 비교
    pool = [x for st in ("dev", "test", "unseen") for x in per[st].values()]
    gain = lambda x: note_f1(x["llm"]) - note_f1(x["stu"])
    def curve(order):
        out = {}
        for frac in (0.1, 0.2, 0.3, 0.5):
            chosen = set(id(x) for x in order[: int(round(frac * len(order)))])
            r, p, _ = agg(pool, lambda x: id(x) in chosen)
            out[frac] = f1(r, p)
        return out
    r0, p0, _ = agg(pool, lambda x: False)
    r1, p1, _ = agg(pool, lambda x: True)
    print(f"\npooled over {len(pool)} notes: student-only F1 {f1(r0, p0):.3f} · LLM-only F1 {f1(r1, p1):.3f}; F1 when routing the top-X% notes by each signal")
    print(f"  {'signal':18s} {'pooled AUC':>10s}   10%    20%    30%    50%")
    rows = []
    for sname in names:
        a = auc([x["sig"][sname] for x in pool], [gain(x) > 0.1 for x in pool])
        sgn = 1 if a >= 0.5 else -1
        order = sorted(pool, key=lambda x: -sgn * x["sig"][sname])
        c = curve(order)
        rows.append((sname, max(a, 1 - a), c))
    for sname, a, c in rows:
        print(f"  {sname:18s} {a:10.2f}   " + "  ".join(f"{c[f]:.3f}" for f in (0.1, 0.2, 0.3, 0.5)))
    c = curve(sorted(pool, key=lambda x: -gain(x)))
    print(f"  {'oracle':18s} {'1.00':>10s}   " + "  ".join(f"{c[f]:.3f}" for f in (0.1, 0.2, 0.3, 0.5)))


if __name__ == "__main__":
    main()
