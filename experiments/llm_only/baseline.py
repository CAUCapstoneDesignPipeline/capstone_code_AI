"""LLM 만으로 개념을 뽑는 기준선 (후보 추출기 없이). 같은 정답·같은 채점으로 `extract-det-3` 과 `extract-hybrid-3`(LLM 거르기) 옆에 놓는다.

  uv run --env-file .env python experiments/llm_only/baseline.py [--sets dev test unseen] [--effort medium] [--workers 4]

LLM 은 노트 전체를 읽고 개념 목록(표기·별칭)을 직접 낸다. 후보 추출기·IDF·점수는 쓰지 않는다. 코드가 하는 일은 두 가지뿐이다:
  1. 근거 확인: 낸 표기(또는 별칭)가 노트 원문에 (공백·대소문자·하이픈을 무시하고) 없으면 버린다 (지어낸 개념, 위치를 못 잡는 개념).
  2. 채점: `ai concepts eval` 과 같은 규칙 (표기 정확히 일치, 재현율 = core, 정밀도 = 정답의 어느 개념과 맞는 비율).
프롬프트는 docs/eval/gold-concepts.md 의 기준을 그대로 옮긴 것이고 어느 정답 노트를 보고 고치지 않았다 (한 번 쓰고 한 번 돌렸다).
집합: dev / test (과목 노트의 정답 23 · 18개), unseen (data/private/notes-unseen + gold-unseen/final-*.csv, 다른 분야 논문 리뷰 18개).
결과 캐시: data/private/cache/llm-only (같은 입력·모델·프롬프트면 다시 묻지 않는다). 노트 내용이 LLM 제공자로 나간다.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.concepts.llm_concepts import PROMPT_VERSION, SCHEMA, SYSTEM, ground, nfc, parse  # noqa: E402,F401


def sets_available(names: list[str]):
    from app.services.concepts.gold import load_gold
    from app.services.concepts.markdown import nfc as _nfc
    from app.services.evaluation import assign_splits

    gold_path = ROOT / "data/private/gold/concepts.csv"
    gold = load_gold(gold_path, include_draft=True)
    rows = list(csv.DictReader(gold_path.open(encoding="utf-8-sig")))
    splits = assign_splits(gold, {_nfc(r["note_id"]).strip(): (r.get("split") or "").strip() for r in rows})
    out = {}
    for s in ("dev", "test"):
        if s in names:
            from app.cli import read_notes
            notes = sorted(n for n in gold if splits[n] == s)
            texts = read_notes(ROOT / "data/private/notes", notes)
            out[s] = ({n: gold[n] for n in notes}, texts)
    if "unseen" in names:
        from app.services.concepts.gold import GoldConcept, loose
        g = {}
        for f in sorted((ROOT / "data/private/gold-unseen").glob("final-*.csv")):
            for r in csv.DictReader(f.open(encoding="utf-8-sig")):
                nm = [x.strip() for x in nfc(r["concept"]).split("|") if loose(x)]
                if nm:
                    g.setdefault(r["note_id"], []).append(GoldConcept(frozenset(loose(x) for x in nm), nm[0], r["level"]))
        out["unseen"] = (g, {n: nfc((ROOT / f"data/private/notes-unseen/{n}.md").read_text(encoding="utf-8")) for n in g})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"], choices=["dev", "test", "unseen"])
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-reference", action="store_true", help="det-3 · hybrid-3 행을 계산하지 않는다 (LLM 만 돌린다)")
    args = ap.parse_args()

    from app.api.v1.extract import select_concepts, to_concepts
    from app.cli import load_extractor
    from app.core.config import EXTRACT_PROFILES
    from app.main import role_llm_config
    from app.services.concepts.filter_llm import ConceptFilter
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.llm import LLMError, make_llm

    data = sets_available(args.sets)
    cfg = role_llm_config("relation", args.effort)
    if cfg is None:
        sys.exit("LLM 설정이 필요하다 (ANTHROPIC_API_KEY 등)")
    llm = make_llm(cfg)
    cache = ROOT / "data/private/cache/llm-only"
    cache.mkdir(parents=True, exist_ok=True)

    def ask(text: str):
        key = hashlib.sha256(json.dumps([llm.provider, llm.model, args.effort, PROMPT_VERSION, SYSTEM, text], ensure_ascii=False).encode()).hexdigest()
        p = cache / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        for attempt in range(5):
            try:
                res = llm.complete_json(SYSTEM, "<노트>\n" + text + "\n</노트>", SCHEMA, "concepts").data
                break
            except LLMError as e:
                if e.kind == "refusal":                    # 안전 분류기 거절 (보안 실습 노트 등): 의견 없음 = 개념 없음. 기록해 둔다
                    res = {"concepts": [], "refused": True}
                    break
                if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res

    ex = None if args.no_reference else load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=ROOT / "data/private/concepts/cache"))
    prof = EXTRACT_PROFILES["extract-det-3"]
    cfilter = None if args.no_reference else ConceptFilter(make_llm(role_llm_config("relation", "low")), ROOT / "data/private/cache/relations/concepts")
    rng = np.random.default_rng(0)
    pct = lambda x: f"{x:.1%}"
    print(f"모델 {llm.model} · effort {args.effort} · {PROMPT_VERSION}\n")
    for name, (gold, texts) in data.items():
        notes = sorted(gold)
        with ThreadPoolExecutor(args.workers) as pool:
            raw = dict(zip(notes, pool.map(lambda n: ask(texts[n]), notes)))
        per = {"LLM only": {}, "extract-det-3": {}, "extract-hybrid-3 (LLM filter)": {}}
        stats = {"proposed": 0, "dropped": 0, "refused": 0}
        for n in notes:
            items = parse(raw[n])
            grounded, dropped = ground(items, texts[n], loose)
            stats["proposed"] += len(items)
            stats["dropped"] += dropped
            stats["refused"] += bool(raw[n].get("refused"))
            per["LLM only"][n] = score(gold[n], [Extracted(f, k, 1.0) for f, k in grounded])
            if ex is not None:
                cs, _ = to_concepts(select_concepts(ex.extract(n, texts[n]), prof), texts[n])
                full = [Extracted(c.label, frozenset({loose(c.label), *(loose(m.quote) for m in c.mentions)}), c.confidence) for c in cs]
                per["extract-det-3"][n] = score(gold[n], full)
                keep = cfilter.keep(texts[n], [c.label for c in cs])
                per["extract-hybrid-3 (LLM filter)"][n] = score(gold[n], [e for e in full if e.label in keep])
        print(f"== {name}: 노트 {len(notes)}개 · core {sum(s.core for s in per['LLM only'].values())} · LLM 이 낸 개념 {stats['proposed']} 중 원문에 없어 버림 {stats['dropped']} · 거절 {stats['refused']}")
        print(f"   {'방법':32s} 재현율(core)  정밀도   추출 수")
        for method, rows in per.items():
            if not rows:
                continue
            tot = lambda f, s: sum(f(rows[n]) for n in s)
            agg = lambda s: (tot(lambda r: r.core_hit, s) / max(1, tot(lambda r: r.core, s)), tot(lambda r: r.extracted_hit, s) / max(1, tot(lambda r: r.extracted, s)))
            r, p = agg(notes)
            boots = [agg(list(rng.choice(notes, len(notes)))) for _ in range(2000)]
            lo = lambda i: np.percentile([b[i] for b in boots], 2.5)
            hi = lambda i: np.percentile([b[i] for b in boots], 97.5)
            print(f"   {method:32s} {pct(r)} [{lo(0):.0%}–{hi(0):.0%}]  {pct(p)} [{lo(1):.0%}–{hi(1):.0%}]  {tot(lambda r: r.extracted, notes)}")
        print()


if __name__ == "__main__":
    main()
