"""교사 라벨: LLM 만으로 뽑은 개념(experiments/llm_only/baseline.py 와 같은 프롬프트·캐시)을 정답 없는 노트에도 만든다.

  uv run --env-file .env python experiments/distill/teacher.py [--workers 4]

정답(concepts.csv)이 있는 노트와 notes-unseen 은 건너뛴다 (학생 모델이 평가 노트를 학습에 쓰지 않게). 노트 내용이 LLM 제공자로 나간다.
"""

import argparse
import hashlib
import json
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.llm_only.baseline import PROMPT_VERSION, SCHEMA, SYSTEM  # noqa: E402

nfc = lambda s: unicodedata.normalize("NFC", s)


def teacher_notes() -> dict[str, str]:
    """정답이 없는 과목 노트 (이름 → 원문)."""
    from app.services.concepts.gold import load_gold
    gold = set(load_gold(ROOT / "data/private/gold/concepts.csv", include_draft=True))
    out = {}
    for p in sorted((ROOT / "data/private/notes").glob("*.md")):
        name = nfc(p.stem)
        if name not in gold:
            out[name] = nfc(p.read_text(encoding="utf-8"))
    return out


def asker(effort: str = "medium"):
    from app.main import role_llm_config
    from app.services.llm import LLMError, make_llm
    llm = make_llm(role_llm_config("relation", effort))
    cache = ROOT / "data/private/cache/llm-only"
    cache.mkdir(parents=True, exist_ok=True)

    def ask(text: str):
        key = hashlib.sha256(json.dumps([llm.provider, llm.model, effort, PROMPT_VERSION, SYSTEM, text], ensure_ascii=False).encode()).hexdigest()
        p = cache / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        for attempt in range(5):
            try:
                res = llm.complete_json(SYSTEM, "<노트>\n" + text + "\n</노트>", SCHEMA, "concepts").data
                break
            except LLMError as e:
                if e.kind == "refusal":
                    res = {"concepts": [], "refused": True}
                    break
                if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="medium")
    args = ap.parse_args()
    notes = teacher_notes()
    ask = asker(args.effort)
    done = refused = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for n, res in zip(notes, pool.map(lambda n: ask(notes[n]), notes)):
            done += 1
            refused += bool(res.get("refused"))
            print(f"[{done}/{len(notes)}] {n[:30]}: {len(res.get('concepts', []))} concepts{' (refused)' if res.get('refused') else ''}", flush=True)
    print(f"done: {done} notes, {refused} refused")


if __name__ == "__main__":
    main()
