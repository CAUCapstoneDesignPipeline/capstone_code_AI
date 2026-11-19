"""출고하는 extract-student-decisions-1 을 정답 노트에서 실제로 불러 잰다 (캐시 없이 OpenAI 를 부른다: 서비스와 같은 StudentJudge 코드·고정 임계값 0.70).

  uv run --env-file .env python experiments/decisions/eval_profile.py [--sets dev test unseen]

노트를 한 번에 하나씩 보내 노트 하나의 처음부터 끝까지(후보·특징·두 요청·섞기·위치) 지연을 잰다. 노트 원문이 OpenAI 로 나간다.
재현율은 core, 정밀도는 정답의 어느 개념과 맞는 비율 (AI 초안 정답, eval_profiles.py 와 같은 채점).
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"])
    args = ap.parse_args()
    import openai

    from app.api.v1.extract import to_concepts
    from app.cli import load_extractor
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.judge import DEFAULT_MODEL, THRESHOLD, WEIGHT, DecisionsJudge, StudentJudge
    from app.services.concepts.student import Student
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    judge = StudentJudge(Student.load(), DecisionsJudge(openai.OpenAI(max_retries=1, timeout=30.0), DEFAULT_MODEL))
    data = sets_available(args.sets)
    print(f"{judge.version} · {judge.model} · blend {1 - WEIGHT:.1f}/{WEIGHT:.1f} · threshold {THRESHOLD}\n")
    print(f"{'set':8s} core recall   precision   F1     concepts   latency median / p90 / max (s)")
    lat_all = []
    for name, (gold, texts) in data.items():
        sc, lat = {}, []
        for n in sorted(gold):
            t0 = time.perf_counter()
            sel = judge.select(ex, n, 1, texts[n])
            lat.append(time.perf_counter() - t0)
            cs, _ = to_concepts(sel, texts[n])
            sc[n] = score(gold[n], [Extracted(c.label, frozenset({loose(c.label), *(loose(m.quote) for m in c.mentions)}), c.confidence) for c in cs])
        R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
        P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
        lat_all += lat
        print(f"{name:8s} {R:10.1%} {P:11.1%}  {2 * R * P / (R + P + 1e-9):.3f} {sum(s.extracted for s in sc.values()):9d}   "
              f"{statistics.median(lat):.2f} / {np.percentile(lat, 90):.2f} / {max(lat):.2f}", flush=True)
    print(f"\nall {len(lat_all)} notes: median {statistics.median(lat_all):.2f}s, p90 {np.percentile(lat_all, 90):.2f}s, max {max(lat_all):.2f}s")


if __name__ == "__main__":
    main()
