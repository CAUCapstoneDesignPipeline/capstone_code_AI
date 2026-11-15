"""OpenAI Decisions API 로 후보 쌍마다 관계를 고르게 한다: 쌍마다 객관식 질문 하나 (관계 없음 / 유형 × 방향 / 대칭 유형), 노트 원문은 한 번만 읽힌다.

  uv run --env-file .env python experiments/relations/decisions_pairs.py [--thr 0.2] [--workers 3]

후보 쌍은 candidates.py (같은 문장 ∪ 같은 절 ∪ 제목↔절 안), 개념은 정답 개념으로 고정한다. 관계 확률 = 1 - P(관계 없음), 유형·방향 = 가장 높은 선택지.
요청 하나에 질문 400개까지라 쌍이 더 많으면 요청을 나눠 동시에 보낸다. 낮은 임계값(--thr)에서 낸 관계를 모두 풀링 판정(pool.py)한 뒤 임계값 별로 채점한다.
노트 원문이 OpenAI 로 나간다. 답은 data/private/cache/experiments/relations_decisions.pkl 에 캐시한다.
"""

import argparse
import hashlib
import pickle
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cache import scratch  # noqa: E402

CACHE = scratch("relations_decisions.pkl")
MAX_Q = 400
QUESTION = ('Read the study note and decide what relation the NOTE ITSELF states between the two terms "{a}" (A) and "{b}" (B), in the sentence, list, table or heading structure where they appear. '
            'Use only what the note says, never outside knowledge. Mere co-occurrence, listing, or sitting in the same section is NOT a relation: choose "none" then.')
CHOICES = [("none", "the note states no relation between A and B"),
           ("is_a_ab", "A is a kind of B (A is a subtype or specific form of B)"), ("is_a_ba", "B is a kind of A"),
           ("part_of_ab", "A is a component or part of B"), ("part_of_ba", "B is a component or part of A"),
           ("causes_ab", "A causes, produces or leads to B"), ("causes_ba", "B causes, produces or leads to A"),
           ("requires_ab", "A needs B in order to hold or work"), ("requires_ba", "B needs A in order to hold or work"),
           ("used_for_ab", "A is used for B (B is the purpose or goal of A)"), ("used_for_ba", "B is used for A (A is the purpose or goal of B)"),
           ("example_of_ab", "A is a concrete example or instance of B"), ("example_of_ba", "B is a concrete example or instance of A"),
           ("contrasts_with", "the note contrasts or compares A and B (neither is the subject)"),
           ("analogous_to", "A and B work on a similar principle")]
CHOICE_LIST = [{"value": v, "description": d} for v, d in CHOICES]


PRED = ('Read the study note. Does the NOTE ITSELF state a relation between the two terms "{a}" and "{b}" - one is a kind of, a part of, a cause of, a requirement of, '
        'used for, or an example of the other, or the note explicitly contrasts or compares them - in a sentence, list, table or heading structure? '
        'Answer TRUE only if the note says it. Mere co-occurrence, listing, sitting in the same section, or outside knowledge is FALSE.')


def ask_pred(client, model, note, pairs, cache):
    """쌍마다 '노트가 관계를 말하는가' predicate 확률 → ([확률], 지연 초)."""
    labels = [(a, b) for a, b, _ in pairs]
    wording = "pred-" + hashlib.sha256(PRED.encode()).hexdigest()[:10]
    chunks = [labels[i:i + MAX_Q] for i in range(0, len(labels), MAX_Q)]

    def call(chunk):
        key = (model, wording, hashlib.sha256((note.text + "\x00" + "\x1f".join(f"{a}\x1e{b}" for a, b in chunk)).encode()).hexdigest())
        if key in cache:
            return cache[key]
        qs = [{"type": "predicate", "instructions": PRED.format(a=a, b=b)} for a, b in chunk]
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                d = client.decisions.create(model=model, input=note.text, questions=qs)
                break
            except Exception:
                if attempt == 7:
                    raise
                time.sleep(min(30, 3 + attempt * 3))
        cache[key] = dict(probs=[0.0 if a.type == "refusal" else a.probability for a in d.answers], seconds=time.perf_counter() - t0, tokens=d.usage.input_tokens)
        return cache[key]

    if not chunks:
        return [], 0.0
    with ThreadPoolExecutor(len(chunks)) as pool:
        outs = list(pool.map(call, chunks))
    return [p for o in outs for p in o["probs"]], max(o["seconds"] for o in outs)


def decode(value):
    """선택지 값 → (유형, a 가 주어인가) 또는 None."""
    if value == "none":
        return None
    if value in ("contrasts_with", "analogous_to"):
        return value, True
    t, d = value.rsplit("_", 1)
    return t, d == "ab"


def named_choices(a, b):
    """선택지 설명에 A·B 대신 실제 표기를 넣는다."""
    return [{"value": v, "description": d.replace("A and B", f'"{a}" and "{b}"').replace("A is", f'"{a}" is').replace("B is", f'"{b}" is').replace("A needs", f'"{a}" needs').replace("B needs", f'"{b}" needs').replace("A causes", f'"{a}" causes').replace("B causes", f'"{b}" causes').replace("A work", f'"{a}" work').replace(" of B", f' of "{b}"').replace(" of A", f' of "{a}"').replace("for B", f'for "{b}"').replace("for A", f'for "{a}"').replace("in B", f'in "{b}"').replace("to B", f'to "{b}"').replace("to A", f'to "{a}"')} for v, d in CHOICES]


def ask_all(client, model, note, pairs, cache, named=False):
    """쌍마다 선택지 확률 dict 를 돌려준다 → ([dict], 이 노트의 지연 초)."""
    labels = [(a, b) for a, b, _ in pairs]
    wording = hashlib.sha256(repr([QUESTION, CHOICES, "named" if named else ""]).encode()).hexdigest()[:10]
    chunks = [labels[i:i + MAX_Q] for i in range(0, len(labels), MAX_Q)]
    results, seconds = [], 0.0

    def call(chunk):
        key = (model, wording, hashlib.sha256((note.text + "\x00" + "\x1f".join(f"{a}\x1e{b}" for a, b in chunk)).encode()).hexdigest())
        if key in cache:
            return cache[key]
        qs = [{"type": "choice", "instructions": QUESTION.format(a=a, b=b), "choices": named_choices(a, b) if named else CHOICE_LIST} for a, b in chunk]
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                d = client.decisions.create(model=model, input=note.text, questions=qs)
                break
            except Exception as e:
                if attempt == 7:
                    raise
                time.sleep(min(30, 3 + attempt * 3))
        probs = [None if a.type == "refusal" else {x.value: x.probability for x in a.probabilities} for a in d.answers]
        cache[key] = dict(probs=probs, seconds=time.perf_counter() - t0, tokens=d.usage.input_tokens)
        return cache[key]

    if not chunks:
        return [], 0.0
    with ThreadPoolExecutor(len(chunks)) as pool:
        outs = list(pool.map(call, chunks))
    return [p for o in outs for p in o["probs"]], max(o["seconds"] for o in outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.2, help="이 이상인 관계를 풀링 판정에 올린다")
    ap.add_argument("--model", default="gpt-6-luna")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"])
    args = ap.parse_args()
    import openai

    from experiments.relations import harness
    from experiments.relations.candidates import candidate_pairs
    from experiments.relations.pool import Pool, canon

    client = openai.OpenAI(max_retries=2, timeout=120)
    cache = pickle.load(open(CACHE, "rb")) if CACHE.exists() else {}
    data = harness.load(args.sets)
    runs = {}                                      # (집합, 노트) → [(a, b, 유형, a가 주어, 확률)], 초
    t0 = time.time()
    for name, notes in data.items():
        def one(note):
            ms = harness.mentions_for(note)
            pairs = candidate_pairs(note.text, ms)
            probs, secs = ask_all(client, args.model, note, pairs, cache)
            rel = []
            for (a, b, _), p in zip(pairs, probs):
                if p is None:
                    continue
                best = max((v for v in p if v != "none"), key=lambda v: p[v], default=None)
                if best is not None:
                    rel.append((a, b, *decode(best), 1.0 - p.get("none", 0.0)))
            return rel, secs, len(pairs)
        with ThreadPoolExecutor(args.workers) as pool:
            for note, (rel, secs, n) in zip(notes.values(), pool.map(one, notes.values())):
                runs[(name, note.name)] = (rel, secs, n)
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"{name}: {len(notes)} 노트, 후보 쌍 평균 {np.mean([runs[(name, n)][2] for n in notes]):.0f}개, 요청 지연 중앙값 {statistics.median([runs[(name, n)][1] for n in notes]):.2f}s ({time.time() - t0:.0f}s 경과)", flush=True)

    def preds(name, note, thr):
        out = []
        for a, b, t, a_src, conf in runs[(name, note.name)][0]:
            if conf >= thr:
                out.append((a, b, t) if a_src else (b, a, t))
        return list(dict.fromkeys(out))

    pool = Pool()
    jobs = [(name, note, preds(name, note, args.thr)) for name, notes in data.items() for note in notes.values()]
    print(f"\n풀링 판정: 새로 {pool.adjudicate(jobs, 4)}개", flush=True)

    f = lambda p, r: 2 * p * r / (p + r + 1e-9)

    def evaluate(name, thr):
        tot = dict(pred=0, ok=0, vsize=0, pok=0, ppred=0, pv=0)
        for note in data[name].values():
            V, Vp = pool.valid_set(name, note), pool.valid_pairs(name, note)
            p = preds(name, note, thr)
            pk, pp = {canon(r) for r in p}, {frozenset(r[:2]) for r in p}
            tot["pred"] += len(pk); tot["ok"] += len(pk & V); tot["vsize"] += len(V)
            tot["ppred"] += len(pp); tot["pok"] += len(pp & Vp); tot["pv"] += len(Vp)
        P, R = tot["ok"] / max(1, tot["pred"]), tot["ok"] / max(1, tot["vsize"])
        Pp, Rp = tot["pok"] / max(1, tot["ppred"]), tot["pok"] / max(1, tot["pv"])
        return tot["pred"], P, R, f(P, R), Pp, Rp, f(Pp, Rp)

    grid = [round(x, 2) for x in np.arange(args.thr, 0.96, 0.05)]
    print(f"\nstrict F1 (맞는 관계 집합 V 기준) 와 임계값 — dev 에서 고른 값을 test·논문에 그대로 쓴다 ({'dev' in data and '고름' or ''})")
    if "dev" in data:
        best = max(grid, key=lambda t: evaluate("dev", t)[3])
        print(f"dev 에서 strict F1 이 가장 높은 임계값: {best}")
    else:
        best = 0.5
    for t in [x for x in grid if abs(x * 20 - round(x * 20)) < 1e-6 and round(x * 10) % 1 == 0][::2]:
        print(f"  thr {t:.2f}: " + " | ".join(f"{n} 낸 {evaluate(n, t)[0]} P/R {evaluate(n, t)[1]:.1%}/{evaluate(n, t)[2]:.1%} ({evaluate(n, t)[3]:.3f})" for n in data))
    print(f"\n임계값 {best} (dev 에서 고름):")
    for n in data:
        k, P, R, F, Pp, Rp, Fp = evaluate(n, best)
        lat = [runs[(n, x)][1] for x in data[n]]
        print(f"  {n:7s} 낸 관계 {k:5d}  strict P/R/F1 {P:.1%} / {R:.1%} / {F:.3f}   pair {Pp:.1%} / {Rp:.1%} / {Fp:.3f}   요청 지연 중앙값 {statistics.median(lat):.2f}s p90 {np.percentile(lat, 90):.2f}s")


if __name__ == "__main__":
    main()
