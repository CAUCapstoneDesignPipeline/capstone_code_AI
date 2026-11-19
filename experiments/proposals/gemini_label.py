"""표(pairing-human.csv)를 다른 계열 모델(Gemini)로 눈가림 표시한다: Claude 계열 라벨러의 편향을 피하려고.

  uv run --env-file .env python experiments/proposals/gemini_label.py [--model gemini-pro-latest]

문장과 키워드만 보낸다 (AI 라벨·열쇠는 보내지 않는다). 프롬프트는 pair_screen.py 의 다섯 라벨 안내와 같다 (pairscreen-llm-1).
결과는 data/private/gold/pairing-human-gemini.csv (id,label,conclusion). 표의 human 칸은 건드리지 않는다. 노트 문장이 Google 로 간다.
"""

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
GOLD = ROOT / "data/private/gold"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-pro-latest")
    ap.add_argument("--out", type=Path, default=GOLD / "pairing-human-gemini.csv")
    args = ap.parse_args()
    from app.services.links.pair_screen import PairScreen
    from app.services.llm import LLMConfig, make_llm
    cfg = LLMConfig("google", args.model, float(os.environ.get("AI_LLM_TIMEOUT_SECONDS", "120")))
    llm = make_llm(replace(cfg, effort="medium"))
    sheet = list(csv.DictReader(open(GOLD / "pairing-human.csv", encoding="utf-8")))
    pairs = [{"id": r["id"], "a": {"keyword": r["A_keyword"], "text": r["A_text"]}, "b": {"keyword": r["B_keyword"], "text": r["B_text"]}} for r in sheet]
    out = PairScreen(llm, None, batch=10).screen(pairs)
    miss = [p["id"] for p in pairs if out.get(p["id"], {}).get("label") in (None, "unsure")]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "conclusion"])
        for p in pairs:
            o = out.get(p["id"], {})
            w.writerow([p["id"], o.get("label", "unsure"), o.get("conclusion", "")])
    print(f"{args.model}: {dict(Counter(o['label'] for o in out.values()))} · 빠진/unsure {len(miss)}개 {miss[:10]} → {args.out}")


if __name__ == "__main__":
    main()
