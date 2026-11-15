"""후보 쌍 분류기 (빠른 관계 모델): 쌍의 구조·위치 특징과 Decisions 확률을 받아, LLM(Sonnet) 이 낸 관계를 흉내 낸다. 노트 단위 교차검증.

  uv run --env-file .env python experiments/relations/pair_model.py [--no-decisions] [--no-judge]

쌍 = candidates.py (같은 문장 ∪ 같은 절 ∪ 제목↔절). 라벨(교사) = Sonnet LLM-only 가 그 쌍에 관계를 냈는가 (쌍 판별), 냈다면 어떤 유형·방향인가 (유형 분류, 쌍이 a→b 순서).
특징: 쌍의 종류(문장·절·제목), 문장·글자 거리, 같은 줄(목록·표), 제목 줄 여부·수준, 언급 수, 첫 언급 위치, 표기 모양, 사이 글에 든 단서 낱말, (선택) Decisions 객관식 확률 15개.
59개 정답 노트를 5겹 교차검증(노트 단위)으로 예측해 풀링 판정(pool.py)으로 채점한다. 임계값은 dev 노트의 교차검증 예측에서 strict F1 이 가장 높은 값.
"""

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SYM = ("contrasts_with", "analogous_to")
CUES = ["은", "는", "이다", "입니다", "한다", "된다", "의", "와", "과", "및", "또는", "→", "->", ":", "(", ")", ",", "를", "을", "에서", "으로", "로", "위해", "때문", "필요", "달리", "반면", "예", "등", "처럼", "vs", "VS", "|", "-"]
CLASSES = ["is_a_ab", "is_a_ba", "part_of_ab", "part_of_ba", "causes_ab", "causes_ba", "requires_ab", "requires_ba", "used_for_ab", "used_for_ba",
           "example_of_ab", "example_of_ba", "contrasts_with", "analogous_to"]


def features(note, ms, pairs, decisions=None):
    """쌍마다 특징 벡터. decisions: {(a, b): {선택지: 확률}} 또는 None."""
    from app.services.relations.extract import _py_offsets
    from app.services.relations.sentences import sentence_spans
    text = note.text
    P = {}
    for o in _py_offsets(text, ms):
        P.setdefault(o.key, []).append((o.start, o.end))
    sents = sentence_spans(text)
    sent_of = lambda p: next((i for i, (s, e) in enumerate(sents) if s <= p < e), -1)
    lines, pos = [], 0
    for line in text.split("\n"):
        lines.append((pos, pos + len(line), line))
        pos += len(line) + 1
    line_of = lambda p: next((i for i, (s, e, _) in enumerate(lines) if s <= p <= e), -1)
    head_level = lambda i: (len(re.match(r"^#+", lines[i][2]).group(0)) if i >= 0 and re.match(r"^#+\s", lines[i][2]) else 0)
    is_list = lambda i: i >= 0 and bool(re.match(r"^\s*([-*+]|\d+[.)])\s", lines[i][2]))
    is_table = lambda i: i >= 0 and lines[i][2].lstrip().startswith("|")
    n_chars, n_lines = max(1, len(text)), max(1, len(lines))
    X = []
    for a, b, kinds in pairs:
        pa, pb = P[a], P[b]
        sa, sb = [sent_of(s) for s, _ in pa], [sent_of(s) for s, _ in pb]
        la, lb = [line_of(s) for s, _ in pa], [line_of(s) for s, _ in pb]
        sd = min(abs(x - y) for x in sa for y in sb)
        both_sent = len(set(sa) & set(sb))
        both_line = len(set(la) & set(lb))
        cd = min(abs(s2 - e1) if s2 >= e1 else abs(s1 - e2) for s1, e1 in pa for s2, e2 in pb for e2 in [e2] if True) if True else 0
        # 가장 가까운 언급 쌍 사이 글
        best = min(((abs(s2 - e1) if s2 >= e1 else abs(s1 - e2)), s1, e1, s2, e2) for s1, e1 in pa for s2, e2 in pb)
        _, s1, e1, s2, e2 = best
        between = text[min(e1, e2):max(s1, s2)] if max(s1, s2) > min(e1, e2) else ""
        between = between[:80]
        ha = [head_level(i) for i in la]
        hb = [head_level(i) for i in lb]
        f = [("sent" in kinds), ("section" in kinds), ("head" in kinds), sd, both_sent, both_line, min(cd, 400), len(between),
             max(ha), max(hb), min(h for h in ha if h) if any(ha) else 0, min(h for h in hb if h) if any(hb) else 0,
             len(pa), len(pb), len(pa) * len(pb), pa[0][0] / n_chars, pb[0][0] / n_chars, min(len(pa), len(pb)) / max(len(pa), len(pb)),
             any(is_list(i) for i in la), any(is_list(i) for i in lb), any(is_table(i) for i in la), any(is_table(i) for i in lb),
             pa[0][0] < pb[0][0], len(a), len(b), bool(re.search(r"[A-Za-z]", a)), bool(re.search(r"[A-Za-z]", b)), bool(re.search(r"[가-힣]", a)), bool(re.search(r"[가-힣]", b)),
             a.lower() in b.lower() or b.lower() in a.lower(), text[e1:e1 + 1] in ("(", "（"), text[e2:e2 + 1] in ("(", "（"),
             s1 < s2 and (re.search(r"\*\*", text[max(0, s1 - 2):s1]) is not None), (re.search(r"\*\*", text[max(0, s2 - 2):s2]) is not None)]
        f += [between.count(c) for c in CUES]
        if decisions is not None:
            d = decisions[(a, b)]
            f += [1.0 - d.get("none", 0.0)] + [d.get(c, 0.0) for c in CLASSES]
        X.append(f)
    return np.array(X, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-decisions", action="store_true", help="Decisions 확률 없이 구조·위치 특징만")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--named", action="store_true", help="Decisions 선택지에 실제 표기를 넣은 질문의 확률을 쓴다")
    ap.add_argument("--teacher", default="llm:anthropic:claude-sonnet-5-5")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from app.services.concepts.gold import loose
    from experiments.relations import decisions_pairs as D
    from experiments.relations import harness, llm_only
    from experiments.relations.candidates import candidate_pairs
    from experiments.relations.pool import Pool, canon

    import openai
    data = harness.load()
    cache = pickle.load(open(D.CACHE, "rb"))
    client = openai.OpenAI(max_retries=2, timeout=120)
    teacher = llm_only.make(args.teacher[4:])
    rows = []                                       # 노트마다: (집합, note, pairs, X, y_pair, y_type)
    for name, notes in data.items():
        for note in notes.values():
            ms = harness.mentions_for(note)
            pairs = candidate_pairs(note.text, ms)
            probs, _ = D.ask_all(client, "gpt-6-luna", note, pairs, cache, named=args.named)
            dec = {(a, b): (p or {}) for (a, b, _), p in zip(pairs, probs)}
            by_loose = {loose(f): c for c, forms in note.concepts.items() for f in forms}
            teach = {}
            for s, t, ty in llm_only.ask(teacher, list(note.concepts), note.text)[0]:
                s, t = by_loose.get(loose(s)), by_loose.get(loose(t))
                if s and t and s != t:
                    teach.setdefault(frozenset((s, t)), (s, t, ty))
            yp = np.array([int(frozenset((a, b)) in teach) for a, b, _ in pairs])
            yt = []
            for (a, b, _), y in zip(pairs, yp):
                if not y:
                    yt.append(-1)
                    continue
                s, t, ty = teach[frozenset((a, b))]
                cls = ty if ty in SYM else f"{ty}_{'ab' if (s, t) == (a, b) else 'ba'}"
                yt.append(CLASSES.index(cls))
            X = features(note, ms, pairs, None if args.no_decisions else dec)
            rows.append((name, note, pairs, X, yp, np.array(yt)))
    X = np.vstack([r[3] for r in rows])
    yp = np.concatenate([r[4] for r in rows])
    yt = np.concatenate([r[5] for r in rows])
    groups = np.concatenate([[i] * len(r[4]) for i, r in enumerate(rows)])
    print(f"쌍 {len(yp)}개 (노트 {len(rows)}), 교사가 관계를 낸 쌍 {yp.mean():.1%}, 특징 {X.shape[1]}개{' (Decisions 없음)' if args.no_decisions else ''}", flush=True)

    mk = lambda **kw: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=0, **kw)
    p_pair, t_pred = np.zeros(len(yp)), np.zeros(len(yp), int)
    for tr, te in GroupKFold(args.folds).split(X, yp, groups):
        p_pair[te] = mk().fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1]
        pos = tr[yt[tr] >= 0]
        m = mk().fit(X[pos], yt[pos])
        proba = m.predict_proba(X[te])
        t_pred[te] = m.classes_[proba.argmax(1)]
    print("교사 쌍 판별 정확도(AUC 대신 PR 평균): ", end="")
    from sklearn.metrics import average_precision_score
    print(f"{average_precision_score(yp, p_pair):.3f} · 유형 맞힘(교사가 낸 쌍 중) {(t_pred[yt >= 0] == yt[yt >= 0]).mean():.1%}", flush=True)

    # 노트별로 되돌려 채점
    off, out = 0, {}
    for name, note, pairs, _, _, _ in rows:
        n = len(pairs)
        out[(name, note.name)] = (pairs, p_pair[off:off + n], t_pred[off:off + n])
        off += n

    def preds(name, note, thr):
        pairs, pp, tt = out[(name, note.name)]
        res = []
        for (a, b, _), p, t in zip(pairs, pp, tt):
            if p >= thr:
                cls = CLASSES[t]
                if cls in SYM:
                    res.append((a, b, cls))
                else:
                    ty, d = cls.rsplit("_", 1)
                    res.append((a, b, ty) if d == "ab" else (b, a, ty))
        return list(dict.fromkeys(res))

    pool = Pool()
    low = 0.2
    if not args.no_judge:
        n = pool.adjudicate([(name, note, preds(name, note, low)) for name, notes in data.items() for note in notes.values()], 4)
        print(f"풀링 판정: 새로 {n}개", flush=True)
    f1 = lambda p, r: 2 * p * r / (p + r + 1e-9)

    def evaluate(name, thr):
        tot = dict(pred=0, ok=0, v=0, pp=0, pok=0, pv=0)
        for note in data[name].values():
            V, Vp = pool.valid_set(name, note), pool.valid_pairs(name, note)
            p = preds(name, note, thr)
            pk, pq = {canon(r) for r in p}, {frozenset(r[:2]) for r in p}
            tot["pred"] += len(pk); tot["ok"] += len(pk & V); tot["v"] += len(V)
            tot["pp"] += len(pq); tot["pok"] += len(pq & Vp); tot["pv"] += len(Vp)
        P, R = tot["ok"] / max(1, tot["pred"]), tot["ok"] / tot["v"]
        Pp, Rp = tot["pok"] / max(1, tot["pp"]), tot["pok"] / tot["pv"]
        return tot["pred"], P, R, f1(P, R), Pp, Rp, f1(Pp, Rp)

    grid = [round(x, 2) for x in np.arange(0.2, 0.9, 0.05)]
    best = max(grid, key=lambda t: evaluate("dev", t)[3])
    print(f"\n임계값 {best} (dev 교차검증 예측에서 strict F1 최고):")
    for n in data:
        k, P, R, F, Pp, Rp, Fp = evaluate(n, best)
        print(f"  {n:7s} 낸 관계 {k:5d}  strict P/R/F1 {P:.1%} / {R:.1%} / {F:.3f}   pair {Pp:.1%} / {Rp:.1%} / {Fp:.3f}")


if __name__ == "__main__":
    main()
