"""실제 노트(data/private/notes)에서 계산용 사실을 LLM 으로 뽑고 R1·R2 를 돌린다. 사람이 정한 기대 결과가 없는 탐색용이다.

  uv run python experiments/derivation/real_notes.py [--sound] [--all-notes] [--workers 6]

복잡도를 말하는 노트("O(", "복잡도")만 고른다 (--all-notes 면 전부). LLM 을 노트마다 한 번 부른다 (캐시).
보고서: data/private/eval/derivations-<시각>.json (노트 내용이 들어 있어 저장소에 넣지 않는다)
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.cli import read_notes  # noqa: E402
from app.main import role_llm_config  # noqa: E402
from app.services.derive.cases import note_uuid  # noqa: E402
from app.services.derive.extract_llm import FactExtractor  # noqa: E402
from app.services.derive.rules import derive  # noqa: E402
from app.services.llm import LLMError, make_llm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", type=Path, default=ROOT / "data/private/notes")
    ap.add_argument("--all-notes", action="store_true")
    ap.add_argument("--sound", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    import hashlib
    texts = read_notes(args.notes)
    picked = {n: t for n, t in texts.items() if args.all_notes or "O(" in t or "복잡도" in t}
    print(f"노트 {len(texts)}개 중 {len(picked)}개", flush=True)
    ex = FactExtractor(make_llm(role_llm_config("facts", "medium")), ROOT / "data/private/cache/relations/facts")

    def one(item):
        name, body = item
        for attempt in range(4):                       # 평가 스크립트만 일시 오류를 다시 시도한다 (서버는 0회)
            try:
                return name, ex.extract(note_uuid(name), 1, body, hashlib.sha256(body.encode()).hexdigest())
            except LLMError as e:
                if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 3:
                    print(f"  ✗ {name}: {e.kind}", flush=True)
                    return name, None
                time.sleep(5 * (attempt + 1))

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        got = dict(pool.map(one, sorted(picked.items())))
    facts = [f for g in got.values() if g for f in g.facts]
    dropped = {}
    for g in got.values():
        for k, v in (g.dropped.items() if g else ()):
            dropped[k] = dropped.get(k, 0) + v
    kinds = {}
    for f in facts:
        kinds[f.kind] = kinds.get(f.kind, 0) + 1
    print(f"사실 {len(facts)}개 ({time.time() - t0:.0f}초) · 종류 {kinds} · 버림 {dropped}", flush=True)
    import re
    titles = {note_uuid(n): re.sub(r"\s+[0-9a-f]{32}$", "", n) for n in picked}      # Notion 내보내기 id 는 뺀다
    ds = derive(facts, titles)
    if args.sound and any(d.status == "conclusion" for d in ds):
        from app.services.derive.sound_llm import Premise, SoundIn, SoundnessJudge, apply
        judge = SoundnessJudge(make_llm(role_llm_config("sound", "medium")), ROOT / "data/private/cache/verifications/soundness")
        by_id = {f.fact_id: f for f in facts}
        todo = [d for d in ds if d.status == "conclusion"]
        verdicts, _, _ = judge.judge([SoundIn(d.derivation_id, d.conclusion, tuple(c for c, _ in d.conditions),
                                              tuple(s.text for s in d.steps),
                                              tuple(Premise(x, by_id[x].text(), by_id[x].evidence.quote) for x in d.premises))
                                      for d in todo])
        apply(ds, {v.ref: v for v in verdicts}, by_id)
    count = {}
    for d in ds:
        count[(d.rule, d.status)] = count.get((d.rule, d.status), 0) + 1
    print("도출 " + " · ".join(f"{r} {s} {n}" for (r, s), n in sorted(count.items())))
    for d in ds:
        if d.status == "conclusion" or (d.rule == "R2" and d.status != "conclusion"):
            print(f"\n[{d.rule} {d.status}] {d.conclusion}")
            print("  노트: " + " | ".join(titles.get(n, n)[:40] for n in d.notes))
            if d.status != "conclusion":
                print("  " + " / ".join(d.reasons))
    out = ROOT / f"data/private/eval/derivations-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"notes": len(picked), "facts": len(facts), "kinds": kinds, "dropped": dropped,
                               "derivations": [{"rule": d.rule, "status": d.status, "conclusion": d.conclusion,
                                                "result": d.result, "reasons": d.reasons, "missing": d.missing,
                                                "notes": [titles.get(n, n) for n in d.notes],
                                                "explanation": d.explanation} for d in ds]},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n보고서 → {out}")


if __name__ == "__main__":
    main()
