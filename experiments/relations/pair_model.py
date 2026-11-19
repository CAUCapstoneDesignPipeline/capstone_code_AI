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

from app.services.relations.pairs import CLASSES, SYM, decision_features, positions, structure_features  # noqa: E402


def features(note, ms, pairs, decisions=None):
    """쌍마다 특징 벡터 (구조·위치 + 선택적으로 Decisions 확률). 특징 정의는 서비스 코드(app/services/relations/pairs.py)와 같다."""
    X = structure_features(note.text, positions(note.text, ms), pairs)
    return X if decisions is None else np.column_stack([X, decision_features(pairs, decisions)])


def teacher_labels(teacher, note, pairs):
    """교사(LLM-only 가 이 쌍에 관계를 냈는가, 냈다면 CLASSES 번호) → (yp, yt). 안 냈으면 yt = -1."""
    from app.services.concepts.gold import loose
    from experiments.relations import llm_only
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
        yt.append(CLASSES.index(ty if ty in SYM else f"{ty}_{'ab' if (s, t) == (a, b) else 'ba'}"))
    return yp, np.array(yt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-decisions", action="store_true", help="Decisions 확률 없이 구조·위치 특징만")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--from-saved", action="store_true", help="교차검증 예측을 다시 학습하지 않고 저장된 것(pair_model_enc_preds.pkl)을 쓴다")
    ap.add_argument("--encoder", action="store_true", help="문맥 인코더(multilingual-e5-small)로 읽은 두 개념의 벡터를 탐침으로 학습해 특징에 더한다")
    ap.add_argument("--named", action="store_true", help="Decisions 선택지에 실제 표기를 넣은 질문의 확률을 쓴다")
    ap.add_argument("--teacher", default="llm:anthropic:claude-sonnet-5-5")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

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
    enc, E_all = None, []
    if args.encoder and not args.from_saved:
        from experiments.distill.encoder_exp import Encoder, candidate_vectors
        from app.services.relations.extract import _py_offsets
        enc = Encoder()
    for name, notes in data.items():
        for note in notes.values():
            ms = harness.mentions_for(note)
            pairs = candidate_pairs(note.text, ms)
            probs, _ = D.ask_all(client, "gpt-6-luna", note, pairs, cache, named=args.named)
            dec = {(a, b): (p or {}) for (a, b, _), p in zip(pairs, probs)}
            yp, yt = teacher_labels(teacher, note, pairs)
            X = features(note, ms, pairs, None if args.no_decisions else dec)
            if enc is not None:
                H, offs = enc.tokens(note.text)
                P = {}
                for o in _py_offsets(note.text, ms):
                    P.setdefault(o.key, []).append((o.start, o.end))
                keys = list(P)
                V, _ = candidate_vectors(H, offs, [{"spans": P[k]} for k in keys])
                idx = {k: i for i, k in enumerate(keys)}
                E_all.append(np.array([np.concatenate([V[idx[a]], V[idx[b]], V[idx[a]] * V[idx[b]]]) for a, b, _ in pairs], np.float32).reshape(len(pairs), -1))
            rows.append((name, note, pairs, X, yp, yt))
    X = np.vstack([r[3] for r in rows])
    yp = np.concatenate([r[4] for r in rows])
    yt = np.concatenate([r[5] for r in rows])
    groups = np.concatenate([[i] * len(r[4]) for i, r in enumerate(rows)])
    print(f"쌍 {len(yp)}개 (노트 {len(rows)}), 교사가 관계를 낸 쌍 {yp.mean():.1%}, 특징 {X.shape[1]}개{' (Decisions 없음)' if args.no_decisions else ''}", flush=True)

    E = np.vstack(E_all) if (args.encoder and not args.from_saved) else None
    mk = lambda **kw: HistGradientBoostingClassifier(max_depth=5, learning_rate=0.08, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=0, **kw)
    def probe_fit_predict(Ea, ypa, yta, Eb):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Ea)
        Ea, Eb = sc.transform(Ea), sc.transform(Eb)
        pb = LogisticRegression(C=0.02, max_iter=100).fit(Ea, ypa).predict_proba(Eb)[:, 1]
        pos = yta >= 0
        lt = LogisticRegression(C=0.02, max_iter=100).fit(Ea[pos], yta[pos])
        tp = np.zeros((len(Eb), len(CLASSES)))
        tp[:, lt.classes_] = lt.predict_proba(Eb)
        return np.column_stack([pb, tp])

    if args.from_saved:
        import pickle as _pk
        from experiments.cache import scratch
        p_pair, t_pred, enc_pair, enc_type = _pk.load(open(scratch("pair_model_enc_preds.pkl"), "rb"))
    else:
        p_pair, t_pred = np.zeros(len(yp)), np.zeros(len(yp), int)
        enc_pair, enc_type = np.zeros(len(yp)), np.zeros(len(yp), int)
        for k, (tr, te) in enumerate(GroupKFold(args.folds).split(X, yp, groups)):
            Xtr, Xte = X[tr], X[te]
            if args.encoder:
                oof = np.zeros((len(tr), 1 + len(CLASSES)))
                for a, b in GroupKFold(4).split(E[tr], yp[tr], groups[tr]):
                    oof[b] = probe_fit_predict(E[tr][a], yp[tr][a], yt[tr][a], E[tr][b])
                pf = probe_fit_predict(E[tr], yp[tr], yt[tr], E[te])
                enc_pair[te], enc_type[te] = pf[:, 0], pf[:, 1:].argmax(1)
                Xtr, Xte = np.column_stack([Xtr, oof]), np.column_stack([Xte, pf])
                print(f"  fold {k + 1}/{args.folds} (탐침 학습 끝)", flush=True)
            p_pair[te] = mk().fit(Xtr, yp[tr]).predict_proba(Xte)[:, 1]
            pos = np.where(yt[tr] >= 0)[0]
            m = mk().fit(Xtr[pos], yt[tr][pos])
            proba = m.predict_proba(Xte)
            t_pred[te] = m.classes_[proba.argmax(1)]
    print("교사 쌍 판별 정확도(AUC 대신 PR 평균): ", end="")
    from sklearn.metrics import average_precision_score
    print(f"{average_precision_score(yp, p_pair):.3f} · 유형 맞힘(교사가 낸 쌍 중) {(t_pred[yt >= 0] == yt[yt >= 0]).mean():.1%}", flush=True)

    # 노트별로 되돌려 채점
    def split_out(pp_all, tt_all):
        off, o = 0, {}
        for name, note, pairs, _, _, _ in rows:
            n = len(pairs)
            o[(name, note.name)] = (pairs, pp_all[off:off + n], tt_all[off:off + n])
            off += n
        return o
    if args.encoder and not args.from_saved:
        import pickle as _pk
        from experiments.cache import scratch
        _pk.dump((p_pair, t_pred, enc_pair, enc_type), open(scratch("pair_model_enc_preds.pkl"), "wb"))
    variants = {"학습한 분류기" + (" + 인코더 탐침 특징" if args.encoder else ""): split_out(p_pair, t_pred)}
    if args.encoder:
        variants["인코더 탐침만"] = split_out(enc_pair, enc_type)
    out = None

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
        for vname, vout in variants.items():
            out = vout
            n = pool.adjudicate([(name, note, preds(name, note, low)) for name, notes in data.items() for note in notes.values()], 4)
            print(f"풀링 판정 ({vname}): 새로 {n}개", flush=True)
    f1 = lambda p, r: 2 * p * r / (p + r + 1e-9)

    def evaluate(name, thr):
        tot = dict(pred=0, ok=0, v=0, pp=0, pok=0, pv=0)
        for note in data[name].values():
            V, Vp = pool.valid_set(name, note), pool.valid_pairs(name, note)
            p = preds(name, note, thr)
            pk, pq = {canon(r) for r in p}, {frozenset(r[:2]) for r in p}
            have = pool.store.get(f"{name}/{note.name}", {})
            gk = {canon((g["source"], g["target"], g["type"])) for g in note.gold}
            tot["unj"] = tot.get("unj", 0) + sum(1 for r in set(p) if canon(r) not in gk and "|".join(r) not in have)
            tot["pred"] += len(pk); tot["ok"] += len(pk & V); tot["v"] += len(V)
            tot["pp"] += len(pq); tot["pok"] += len(pq & Vp); tot["pv"] += len(Vp)
        evaluate.judged, evaluate.total = evaluate.__dict__.get("judged", 0), evaluate.__dict__.get("total", 0)
        P, R = tot["ok"] / max(1, tot["pred"]), tot["ok"] / tot["v"]
        Pp, Rp = tot["pok"] / max(1, tot["pp"]), tot["pok"] / tot["pv"]
        evaluate.unjudged = tot.get("unj", 0) / max(1, tot["pred"])
        return tot["pred"], P, R, f1(P, R), Pp, Rp, f1(Pp, Rp)

    grid = [round(x, 2) for x in np.arange(0.2, 0.9, 0.05)]
    for vname, vout in variants.items():
        out = vout
        best = max(grid, key=lambda t: evaluate("dev", t)[3])
        print(f"\n{vname} — 임계값 {best} (dev 교차검증 예측에서 strict F1 최고):")
        for n in data:
            k, P, R, F, Pp, Rp, Fp = evaluate(n, best)
            print(f"  {n:7s} 낸 관계 {k:5d}  strict P/R/F1 {P:.1%} / {R:.1%} / {F:.3f}   pair {Pp:.1%} / {Rp:.1%} / {Fp:.3f}   (판정 없는 예측 {evaluate.unjudged:.0%})")


if __name__ == "__main__":
    main()
