"""합성 사례 평가: 노트에서 사실을 LLM 으로 뽑아(facts-llm-3) 규칙으로 도출하고 기대 결과와 비교한다. 문체별로 쪼개 어디가 약한지 본다.

  uv run python experiments/synthetic/evaluate.py [--data data/private/synthetic] [--limit 60] [--sound] [--workers 6]

기대 결과는 명세의 사실에 규칙 엔진을 돌려 얻은 것이라, 이 평가가 재는 것은 규칙이 아니라 '노트 문체가 달라도 사실을 제대로 뽑는가' 이다.
추가로 명세 사실의 quote 가 추출 사실의 근거와 겹치는지(사실 재현율)를 종류별로 본다.
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.main import role_llm_config  # noqa: E402
from app.services.derive.cases import load_note, locate, run_case, score_case  # noqa: E402
from app.services.derive.extract_llm import FactExtractor  # noqa: E402
from app.services.evaluation import bootstrap, ratio  # noqa: E402
from app.services.llm import LLMError, make_llm  # noqa: E402


def retrying(fn):
    for t in range(5):
        try:
            return fn()
        except LLMError as e:
            if e.kind not in ("timeout", "provider", "rate_limit") or t == 4:
                raise
            time.sleep(5 * (t + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/private/synthetic")
    ap.add_argument("--limit", type=int, default=0, help="사례를 이만큼만 (장면 비율 유지, seed 고정)")
    ap.add_argument("--sound", action="store_true", help="soundness 판정까지 (LLM 호출이 더 든다)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cases = json.loads((args.data / "cases-synthetic.json").read_text(encoding="utf-8"))["cases"]
    specs = json.loads((args.data / "specs.json").read_text(encoding="utf-8"))
    if args.limit and args.limit < len(cases):
        random.Random(args.seed).shuffle(cases)
        cases = cases[:args.limit]
    ex = FactExtractor(make_llm(role_llm_config("facts", "medium")), ROOT / "data/private/cache/relations/facts")
    judge = None
    if args.sound:
        from app.services.derive.sound_llm import SoundnessJudge
        judge = SoundnessJudge(make_llm(role_llm_config("sound", "medium")),
                               ROOT / "data/private/cache/verifications/soundness")

    def one(case):
        known = {"s": set(), "r": set(), "o": set()}          # 사례마다 따로 (사례끼리 key 를 알려 주지 않는다)

        def facts_of(n):
            got = retrying(lambda: ex.extract(n.note_id, 1, n.body, n.sha, sorted(known["s"]), sorted(known["r"]),
                                              sorted(known["o"])))
            known["s"].update(f.subject_key for f in got.facts)
            known["r"].update(f.role for f in got.facts if f.role)
            known["o"].update(f.op for f in got.facts if f.op)
            return got.facts
        try:
            got, used = run_case(case, args.data, facts_of)
        except LLMError as e:
            return {"id": case["id"], "error": e.kind}
        if judge is not None:
            from app.services.derive.sound_llm import Premise, SoundIn, apply
            by_id = {f.fact_id: f for f in used}
            todo = [d for d in got if d.status == "conclusion"]
            if todo:
                v, _, _ = retrying(lambda: judge.judge([SoundIn(
                    d.derivation_id, d.conclusion, tuple(t for t, _ in d.conditions), tuple(s.text for s in d.steps),
                    tuple(Premise(x, by_id[x].text(), by_id[x].evidence.quote) for x in d.premises)) for d in todo]))
                apply(got, {x.ref: x for x in v}, by_id)
        s = score_case(case, got)
        # 사실 재현율: 명세 사실의 quote 위치와 겹치는 추출 사실이 있는가 (노트 이름 → 추출 근거 구간)
        notes = {n: load_note(n, args.data) for n in case["notes"]}
        spans = defaultdict(list)
        for f in used:
            spans[f.note].append((f.evidence.start, f.evidence.end))
        cov = []
        for n in specs[str(int(case["id"].split("-")[1]))]["notes"]:
            body = notes[n["name"]].body
            for f in n["facts"]:
                sp = locate(body, f["quote"])
                hit = bool(sp) and any(a < sp[1] and sp[0] < b for a, b in spans[notes[n["name"]].note_id])
                cov.append((f["fact"]["kind"], hit))
        return {**s, "meta": case["meta"], "styles": case["styles"], "scene": case["scene"], "cov": cov,
                "got": [(d.rule, d.status, d.result, d.case) for d in got]}

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, cases))
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]

    def line(name, rs):
        pr = bootstrap(rs, ratio(lambda r: r["true_conclusions"], lambda r: r["conclusions"]))
        rc = bootstrap(rs, ratio(lambda r: r["found"], lambda r: r["expected"]))
        st = bootstrap(rs, ratio(lambda r: r["status_ok"], lambda r: r["status_total"]))
        neg = [r for r in rs if r["negative"]]
        fc = f"{sum(r['false_confirm'] for r in neg)}/{len(neg)}" if neg else "-"
        show = lambda e, den: f"{e.value:4.0%} [{e.lo:.0%}–{e.hi:.0%}]" if den else "  -           "
        print(f"  {name:<24} n={len(rs):<4} 정밀도 {show(pr, sum(r['conclusions'] for r in rs))}  "
              f"재현율 {show(rc, sum(r['expected'] for r in rs))}  상태 {show(st, sum(r['status_total'] for r in rs))}  "
              f"잘못된 확정 {fc}")

    print(f"사례 {len(rows)}개 평가 ({time.time() - t0:.0f}초, LLM 오류 {len(errors)}개) · soundness {'켬' if judge else '끔'}\n전체")
    line("전체", rows)
    for title, key in (("장면", lambda r: r["scene"]), ("알고리즘 역할", lambda r: r["meta"]["role"]),
                       ("알고리즘 노트 언어", lambda r: next(v["lang"] for k, v in r["styles"].items() if v["kind"] == "algorithm")),
                       ("구성 요소 노트 형식", lambda r: next(v["format"] for k, v in r["styles"].items() if v["kind"] == "component")),
                       ("알고리즘 노트 형식", lambda r: next(v["format"] for k, v in r["styles"].items() if v["kind"] == "algorithm"))):
        print(f"\n{title}")
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        for g in sorted(groups):
            line(g, groups[g])
    cov = defaultdict(lambda: [0, 0])
    for r in rows:
        for kind, hit in r["cov"]:
            cov[kind][0] += hit
            cov[kind][1] += 1
    print("\n명세 사실이 추출된 비율 (종류별)")
    for k, (a, b) in sorted(cov.items()):
        print(f"  {k:<14} {a / b:4.0%} ({a}/{b})")
    bad = [r for r in rows if r["mismatch"]]
    print(f"\n기대와 다른 사례 {len(bad)}개")
    for r in bad[:12]:
        print(f"  {r['id']} [{r['scene']}] {r['meta']['algorithm']} + {r['meta']['component']}: "
              + "; ".join(f"{m['key'].split('/')[1] if '/' in m['key'] else m['key']} 기대 {m['want']} 결과 {m['got']}"
                          for m in r["mismatch"][:3]))
    out = ROOT / f"data/private/eval/synthetic-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rows": rows, "errors": errors}, ensure_ascii=False, indent=1, default=list), encoding="utf-8")
    print(f"\n보고서 → {out}")


if __name__ == "__main__":
    main()
