"""정답에 없는 '오답' 관계 중 실제로 노트가 말하는 것이 얼마나 되는지 잰다 (정답이 불완전해서 정밀도가 낮게 나오는지 가린다).

  uv run --env-file .env python experiments/relations/adjudicate.py [--method llm:anthropic:claude-sonnet-5-5] [--per-note 6]

정답에 맞지 않은 예측 관계를 노트마다 무작위로 몇 개 뽑아 Opus 5.5 가 노트를 보고 엄격하게 판정한다 (노트가 그 관계를 말하는가, 어느 유형이 맞는가).
오답을 둘로 나눈다: 쌍 자체가 정답에 없는 것(새 쌍)과 쌍은 정답에 있는데 유형·방향이 다른 것. 표본에서 맞다고 본 비율로 전체 정밀도를 다시 어림한다.
판정자(Opus)가 정답 초안을 쓴 모델과 같으므로 (정답과 판단이 닮는다) 독립 검증은 아니다 — 정답 쪽 기준에 다시 비춰 본 값이다.
"""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.relations import gold as G  # noqa: E402
from experiments.relations.harness import SYMMETRIC  # noqa: E402

CACHE = ROOT / "data/private/cache/relations-gold"
SYSTEM = ("너는 노트에서 뽑은 개념 관계가 맞는지 엄격하게 판정한다. 각 관계에 대해: 노트가 그 두 개념의 관계를 말하는가 (상식이나 노트 밖 지식, 단순한 나열·동시 언급은 아니다)?\n"
          "- valid=true 이면 노트가 두 개념 사이에 어떤 관계를 말한다는 뜻이고, best_type 에 그 관계에 가장 맞는 유형을, swap 에 (방향이 있는 유형일 때) 주어가 target 쪽이면 true 를 적는다.\n"
          "- valid=false 이면 노트가 두 개념의 관계를 말하지 않는다. 확신이 없으면 false 다.\n" + G.RULES.format(types=G.TYPE_LINES))
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["judgments"], "properties": {"judgments": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "valid", "best_type", "swap"],
    "properties": {"id": {"type": "integer"}, "valid": {"type": "boolean"}, "best_type": {"type": "string", "enum": G.TYPES}, "swap": {"type": "boolean"}}}}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="llm:anthropic:claude-sonnet-5-5")
    ap.add_argument("--per-note", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    from app.main import role_llm_config
    from app.services.concepts.gold import loose
    from app.services.llm import make_llm
    from experiments.relations import harness, llm_only

    judge = make_llm(replace(role_llm_config("relation", "medium"), provider="anthropic", model=G.DRAFT_MODEL, effort="medium"))
    llm = llm_only.make(args.method[4:])
    data = harness.load()
    rng = random.Random(0)

    def predictions(note):
        by_loose = {loose(f): c for c, forms in note.concepts.items() for f in forms}
        out = []
        for a, b, t in llm_only.ask(llm, list(note.concepts), note.text)[0]:
            a, b = by_loose.get(loose(a)), by_loose.get(loose(b))
            if a and b and a != b:
                out.append((a, b, t))
        return list(dict.fromkeys(out))

    jobs = []
    stats = {}
    for name, notes in data.items():
        s = stats.setdefault(name, dict(pred=0, tp=0, new=0, wrongtype=0, sample_new=0, valid_new=0, same_new=0, sample_wt=0, valid_wt=0, same_wt=0))
        for note in notes.values():
            pred = predictions(note)
            gold_keys = {harness._key((r["source"], r["target"], r["type"])) for r in note.gold}
            gold_pairs = {frozenset((r["source"], r["target"])) for r in note.gold}
            fps = []
            for r in pred:
                s["pred"] += 1
                if harness._key(r) in gold_keys:
                    s["tp"] += 1
                else:
                    kind = "wt" if frozenset(r[:2]) in gold_pairs else "new"
                    s["new" if kind == "new" else "wrongtype"] += 1
                    fps.append((r, kind))
            for r, kind in rng.sample(fps, min(args.per_note, len(fps))):
                jobs.append((name, note, r, kind))

    by_note = {}
    for job in jobs:
        by_note.setdefault((job[0], job[1].name), []).append(job)

    def judge_note(item):
        (name, _), group = item
        note = group[0][1]
        listing = "\n".join(f"[{i}] {r[0]} --{r[2]}--> {r[1]}" for i, (_, _, r, _) in enumerate(group))
        user = "<개념 목록>\n" + "\n".join(f"- {c}" for c in note.concepts) + f"\n</개념 목록>\n\n<노트>\n{note.text}\n</노트>\n\n<판정할 관계>\n{listing}\n</판정할 관계>"
        key = hashlib.sha256(json.dumps([judge.model, SYSTEM, user], ensure_ascii=False).encode()).hexdigest()
        p = CACHE / f"adj-{key}.json"
        if p.exists():
            res = json.loads(p.read_text(encoding="utf-8"))
        else:
            res = judge.complete_json(SYSTEM, user, SCHEMA, "adjudicate").data
            p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return group, {j["id"]: j for j in res["judgments"]}

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(judge_note, by_note.items()))
    for group, js in results:
        for i, (name, note, r, kind) in enumerate(group):
            j = js.get(i)
            s = stats[name]
            tag = "new" if kind == "new" else "wt"
            s[f"sample_{tag}"] += 1
            if j and j["valid"]:
                s[f"valid_{tag}"] += 1
                best = j["best_type"]
                src, tgt = (r[1], r[0]) if (j["swap"] and best not in SYMMETRIC) else (r[0], r[1])
                if best == r[2] and (src, tgt) == (r[0], r[1]):
                    s[f"same_{tag}"] += 1

    print(f"{args.method} · 판정 {judge.model} · 노트마다 오답 {args.per_note}개씩 표본\n")
    print(f"{'set':7s} 낸 관계  정답과 일치   새 쌍(표본 중 맞음 / 유형까지 같음)   쌍은 있는데 유형·방향 다름(표본 중 맞음 / 유형까지 같음)   어림한 정밀도(strict → 노트가 말하는 관계)")
    for name, s in stats.items():
        v_new = s["valid_new"] / max(1, s["sample_new"])
        v_wt = s["valid_wt"] / max(1, s["sample_wt"])
        sm_new, sm_wt = s["same_new"] / max(1, s["sample_new"]), s["same_wt"] / max(1, s["sample_wt"])
        strict = (s["tp"] + s["new"] * sm_new + s["wrongtype"] * sm_wt) / s["pred"]
        any_valid = (s["tp"] + s["new"] * v_new + s["wrongtype"] * v_wt) / s["pred"]
        print(f"{name:7s} {s['pred']:6d} {s['tp']:6d} ({s['tp'] / s['pred']:.0%})   {s['new']:4d} ({v_new:.0%} / {sm_new:.0%})   {s['wrongtype']:4d} ({v_wt:.0%} / {sm_wt:.0%})   "
              f"{s['tp'] / s['pred']:.0%} → {strict:.0%} (유형 같음) · {any_valid:.0%} (어떤 관계든 노트가 말함)")


if __name__ == "__main__":
    main()
