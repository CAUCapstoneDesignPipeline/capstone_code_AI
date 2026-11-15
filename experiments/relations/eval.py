"""관계 방법을 같은 정답·같은 채점으로 견준다 (개념은 정답 개념으로 고정).

  uv run --env-file .env python experiments/relations/eval.py [--methods reach rules] [--sets dev test unseen]

방법: reach = 정답 관계가 후보 쌍 규칙·문장·구조로 닿는 비율 (후보 쌍으로 일하는 방법의 재현율 상한), rules = 규칙+구조 (서비스의 기본 관계, LLM 없음).
"""

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def fmt(a):
    return " ".join(f"{k} {a[k][0]:.1%}/{a[k][1]:.1%} ({a[k][2]:.3f})" for k in ("strict", "type", "pair"))


def method_rules():
    from app.dev.router import _policy_types
    from app.services.relations.extract import Limits, extract_relations
    policy = list(_policy_types())

    def run(note, mentions):
        _, rels = extract_relations(note.text, mentions, policy, None, Limits(), structure=True)
        return [(r.source, r.target, r.code) for r in rels], None
    return run


def method_llm(spec):
    """LLM 만으로 (후보 쌍 없이): llm:provider:model[:effort]."""
    from app.services.concepts.gold import loose
    from experiments.relations import llm_only
    llm = llm_only.make(spec)

    def run(note, mentions):
        labels = list(note.concepts)
        by_loose = {loose(f): c for c, forms in note.concepts.items() for f in forms}
        preds, seconds = llm_only.ask(llm, labels, note.text)
        out = []
        for a, b, t in preds:
            a, b = by_loose.get(loose(a)), by_loose.get(loose(b))
            if a and b and a != b:
                out.append((a, b, t))
        return out, seconds
    return run


def method_hybrid(spec):
    """규칙+구조 + 규칙이 못 정한 문장 속 쌍을 LLM typer 로 (서비스의 extract-hybrid 계열): hybrid:provider:model[:effort].
    지연은 처음 불렀을 때의 노트당 시간을 기록해 두고 캐시 뒤에도 그 값을 쓴다."""
    import json
    import threading
    from app.dev.router import _policy_types
    from app.services.relations.extract import Limits, extract_relations
    from app.services.relations.typer import RelationTyper
    from experiments.relations import llm_only
    policy = list(_policy_types())
    cache = ROOT / "data/private/cache/relations-eval" / spec.replace(":", "_")
    cache.mkdir(parents=True, exist_ok=True)
    typer = RelationTyper(llm_only.make(spec), cache, chunk_size=40, max_items=200)
    lat_path, lock = cache / "latency.json", threading.Lock()
    lat = json.loads(lat_path.read_text()) if lat_path.exists() else {}

    def run(note, mentions):
        t0 = time.perf_counter()
        _, rels = extract_relations(note.text, mentions, policy, typer, Limits(), structure=True)
        took = time.perf_counter() - t0
        with lock:
            lat.setdefault(note.name, took)
            lat_path.write_text(json.dumps(lat))
            seconds = lat[note.name]
        return [(r.source, r.target, r.code) for r in rels], seconds
    return run


def method_decisions(spec, named=False):
    """후보 쌍(같은 문장·절·제목↔절)마다 Decisions 객관식 질문 하나: decisions:임계값."""
    import openai
    import pickle
    from experiments.relations import decisions_pairs as D
    from experiments.relations.candidates import candidate_pairs
    thr = float(spec)
    client = openai.OpenAI(max_retries=2, timeout=120)
    cache = pickle.load(open(D.CACHE, "rb")) if D.CACHE.exists() else {}

    def run(note, mentions):
        pairs = candidate_pairs(note.text, mentions)
        probs, secs = D.ask_all(client, "gpt-6-luna", note, pairs, cache, named=named)
        out = []
        for (a, b, _), p in zip(pairs, probs):
            if p is None or 1.0 - p.get("none", 0.0) < thr:
                continue
            best = max((v for v in p if v != "none"), key=lambda v: p[v])
            t, a_src = D.decode(best)
            out.append((a, b, t) if a_src else (b, a, t))
        return out, secs
    return run


def method_decisions2(spec):
    """Decisions 두 질문 (객관식 + '노트가 관계를 말하는가' predicate): 관계 확률 = 둘의 평균, 유형·방향 = 객관식. decisions2:임계값."""
    import openai
    import pickle
    from concurrent.futures import ThreadPoolExecutor
    from experiments.relations import decisions_pairs as D
    from experiments.relations.candidates import candidate_pairs
    thr = float(spec)
    client = openai.OpenAI(max_retries=2, timeout=120)
    cache = pickle.load(open(D.CACHE, "rb")) if D.CACHE.exists() else {}

    def run(note, mentions):
        pairs = candidate_pairs(note.text, mentions)
        with ThreadPoolExecutor(2) as pool:                     # 두 요청을 동시에 보낸다
            f1 = pool.submit(D.ask_all, client, "gpt-6-luna", note, pairs, cache)
            f2 = pool.submit(D.ask_pred, client, "gpt-6-luna", note, pairs, cache)
            (probs, s1), (pred, s2) = f1.result(), f2.result()
        out = []
        for (a, b, _), p, q in zip(pairs, probs, pred):
            if p is None or (1.0 - p.get("none", 0.0) + q) / 2 < thr:
                continue
            best = max((v for v in p if v != "none"), key=lambda v: p[v])
            t, a_src = D.decode(best)
            out.append((a, b, t) if a_src else (b, a, t))
        return out, max(s1, s2)
    return run


def build(m):
    return (method_rules() if m == "rules" else method_llm(m[4:]) if m.startswith("llm:") else method_hybrid(m[7:]) if m.startswith("hybrid:")
            else method_decisions(m[10:]) if m.startswith("decisions:") else method_decisions2(m[11:]) if m.startswith("decisions2:") else method_decisions(m[11:], named=True) if m.startswith("decisionsn:") else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["reach", "rules"])
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from experiments.relations import harness

    data = harness.load(args.sets)
    if "reach" in args.methods:
        print(f"{'set':7s} {'notes':>5s} {'gold':>5s} 후보쌍규칙  같은문장  구조  셋 중 하나  개념 언급 못 찾음")
        for name, notes in data.items():
            tot = dict(n=0, pool=0, sentence=0, structure=0, any=0, missing=0)
            for note in notes.values():
                r = harness.reach(note)
                tot["n"] += len(note.gold)
                tot["pool"] += len(r["pool"])
                tot["sentence"] += len(r["sentence"])
                tot["structure"] += len(r["structure"])
                tot["any"] += len(r["pool"] | r["sentence"] | r["structure"])
                tot["missing"] += len(r["unseen_mention"])
            n = tot["n"]
            print(f"{name:7s} {len(notes):5d} {n:5d} {tot['pool'] / n:9.1%} {tot['sentence'] / n:8.1%} {tot['structure'] / n:6.1%} {tot['any'] / n:10.1%} {tot['missing'] / n:12.1%}")
    for m in args.methods:
        if m == "reach":
            continue
        fn = build(m)
        if fn is None:
            print(f"알 수 없는 방법 {m}")
            continue
        print(f"\n== {m}")
        for name, notes in data.items():
            t0 = time.perf_counter()
            with ThreadPoolExecutor(args.workers) as pool:
                outs = list(pool.map(lambda note: fn(note, harness.mentions_for(note)), notes.values()))
            per = [harness.score_note(note.gold, preds) for note, (preds, _) in zip(notes.values(), outs)]
            a = harness.aggregate(per)
            secs = [sec for _, sec in outs if sec is not None]
            lat = f"지연 중앙값 {statistics.median(secs):.1f}s · p90 {np.percentile(secs, 90):.1f}s" if secs else f"{(time.perf_counter() - t0) / len(notes) * 1000:.0f} ms/노트"
            print(f"{name:7s} 정답 {a['n_gold']:4d} 낸 관계 {a['n_pred']:4d}  {fmt(a)}  draft 재현율 {a['draft_recall']:.1%}  ({lat})")


if __name__ == "__main__":
    main()
