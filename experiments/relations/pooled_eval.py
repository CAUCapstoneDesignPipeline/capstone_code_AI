"""여러 관계 방법을 풀링 판정(pool.py)으로 견준다: 모든 방법의 예측을 모아 정답에 없는 것을 한 번씩 판정하고, 같은 '맞는 관계' 집합 V 로 채점한다.

  uv run --env-file .env python experiments/relations/pooled_eval.py --methods rules llm:anthropic:claude-sonnet-5-5 hybrid:anthropic:claude-sonnet-5-5:low

개념은 정답 개념으로 고정한다. 판정이 처음이면 Opus 를 부른다 (노트 내용이 LLM 제공자로 나간다). 판정은 저장되어 다음 실행에서 다시 하지 않는다.
V 는 저장된 판정 전체(지금까지 견준 모든 방법)에서 만들어지므로 방법을 더하면 V 가 커질 수 있다 — 방법은 같이 비교해야 한다.
출력: strict = 쌍·유형·방향이 V 에 든 것 (정밀도·재현율·F1), pair = 두 개념 사이 관계가 있다고 맞힌 것, gold 재현율 = 원래 정답 중 낸 비율.
"""

import argparse
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true", help="판정을 새로 하지 않고 저장된 판정만 쓴다")
    ap.add_argument("--all-stored", action="store_true", help="(쓰지 않음) V 는 항상 저장된 판정 전체")
    args = ap.parse_args()
    from experiments.relations import harness
    from experiments.relations.eval import build
    from experiments.relations.pool import Pool, canon

    data = harness.load(args.sets)
    preds, secs = {}, {}
    for m in args.methods:
        fn = build(m)
        preds[m], secs[m] = {}, {}
        for name, notes in data.items():
            with ThreadPoolExecutor(args.workers) as pool:
                outs = list(pool.map(lambda note: fn(note, harness.mentions_for(note)), notes.values()))
            for note, (p, s) in zip(notes.values(), outs):
                preds[m][(name, note.name)] = list(dict.fromkeys(p))
                secs[m][(name, note.name)] = s
    pool = Pool()
    if not args.no_judge:
        jobs = [(name, note, sum((preds[m][(name, note.name)] for m in args.methods), [])) for name, notes in data.items() for note in notes.values()]
        n = pool.adjudicate(jobs, args.workers)
        print(f"새로 판정한 관계 {n}개 (저장된 판정 {sum(len(v) for v in pool.store.values())}개)\n")

    print(f"{'방법':44s} {'set':7s} {'낸':>5s} {'V':>5s}  strict P / R (F1)          pair P / R (F1)            gold 재현율   지연(중앙값 · p90)")
    for m in args.methods:
        for name, notes in data.items():
            tot = dict(pred=0, ok=0, vsize=0, vhit=0, ppred=0, pok=0, pv=0, pvhit=0, g=0, ghit=0)
            for note in notes.values():
                V, Vp = pool.valid_set(name, note), pool.valid_pairs(name, note)
                p = preds[m][(name, note.name)]
                pk = {canon(r) for r in p}
                pp = {frozenset(r[:2]) for r in p}
                gk = {canon((r["source"], r["target"], r["type"])) for r in note.gold}
                tot["pred"] += len(pk)
                tot["ok"] += len(pk & V)
                tot["vsize"] += len(V)
                tot["vhit"] += len(pk & V)
                tot["ppred"] += len(pp)
                tot["pok"] += len(pp & Vp)
                tot["pv"] += len(Vp)
                tot["pvhit"] += len(pp & Vp)
                tot["g"] += len(gk)
                tot["ghit"] += len(pk & gk)
            P, R = tot["ok"] / max(1, tot["pred"]), tot["vhit"] / max(1, tot["vsize"])
            Pp, Rp = tot["pok"] / max(1, tot["ppred"]), tot["pvhit"] / max(1, tot["pv"])
            f = lambda p, r: 2 * p * r / (p + r + 1e-9)
            ss = [s for (n, _), s in secs[m].items() if n == name and s is not None]
            lat = f"{statistics.median(ss):.1f}s · {np.percentile(ss, 90):.1f}s" if ss else "-"
            print(f"{m:44s} {name:7s} {tot['pred']:5d} {tot['vsize']:5d}  {P:6.1%} / {R:6.1%} ({f(P, R):.3f})   {Pp:6.1%} / {Rp:6.1%} ({f(Pp, Rp):.3f})   {tot['ghit'] / tot['g']:8.1%}   {lat}")
        print()


if __name__ == "__main__":
    main()
