"""출고하는 두 profile 을 정답 노트에서 잰다: extract-student-1 (학습한 모델 그대로) 과 extract-llm-1 (서비스의 개념 뽑기·위치 찾기 코드, LLM 출력은 캐시에서 읽는다 — 새로 부르지 않는다).

  uv run --env-file .env python experiments/distill/eval_profiles.py

LLM 출력 캐시는 experiments/llm_only/baseline.py 가 만든 것이다. 재현율은 core, 정밀도는 정답의 어느 개념과 맞는 비율 (AI 초안 정답, `ai concepts eval` 과 같은 채점).
학생 지연은 이 컴퓨터에서 노트 하나를 처음부터 끝까지(후보·특징·모델·위치) 고르는 시간이다.
"""

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class CachedLLM:
    """baseline.py 가 캐시한 LLM 출력을 돌려주는 어댑터 (없으면 오류: 여기서는 호출하지 않는다)."""
    provider, model = "anthropic", "cached"

    def __init__(self, real, originals, effort="medium"):
        self.real, self.effort, self.originals = real, effort, originals      # originals: NFC 로 바꾼 글 → baseline 이 캐시한 원래 글

    def complete_json(self, system, user, schema, name):
        from app.services.concepts.llm_concepts import PROMPT_VERSION, SYSTEM
        text = user[len("<노트>\n"):-len("\n</노트>")]
        text = self.originals.get(text, text)
        key = hashlib.sha256(json.dumps([self.real.provider, self.real.model, self.effort, PROMPT_VERSION, SYSTEM, text], ensure_ascii=False).encode()).hexdigest()
        p = ROOT / "data/private/cache/llm-only" / f"{key}.json"
        if not p.exists():
            raise RuntimeError("LLM 출력 캐시에 없다")
        return SimpleNamespace(data=json.loads(p.read_text(encoding="utf-8")))


def main():
    argparse.ArgumentParser().parse_args()
    from app.api.v1.extract import to_concepts
    from app.cli import load_extractor
    from app.main import role_llm_config
    from app.services.concepts.gold import Extracted, loose, score
    from app.services.concepts.llm_concepts import ConceptLLM
    from app.services.concepts.student import Student
    from app.services.llm import make_llm
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    student = Student.load()
    data = sets_available(["dev", "test", "unseen"])
    import unicodedata
    originals = {unicodedata.normalize("NFC", t): t for _, (_, texts) in data.items() for t in texts.values()}
    llm = ConceptLLM(CachedLLM(make_llm(role_llm_config("relation", "medium")), originals))
    print(f"student {student.version} (threshold {student.threshold}, {student.trained.get('notes')} training notes)\n")
    lat = []
    print(f"{'set':8s} {'profile':18s} core recall   precision   concepts")
    for name, (gold, texts) in data.items():
        for label in ("extract-student-1", "extract-llm-1"):
            sc = {}
            for n in sorted(gold):
                t0 = time.perf_counter()
                sel = student.select(ex, n, 1, texts[n]) if label == "extract-student-1" else llm.select(n, 1, texts[n])
                if label == "extract-student-1":
                    lat.append((time.perf_counter() - t0) * 1000)
                cs, _ = to_concepts(sel, texts[n])
                ext = [Extracted(c.label, frozenset({loose(c.label), *(loose(m.quote) for m in c.mentions)}), c.confidence) for c in cs]
                sc[n] = score(gold[n], ext)
            R = sum(s.core_hit for s in sc.values()) / sum(s.core for s in sc.values())
            P = sum(s.extracted_hit for s in sc.values()) / max(1, sum(s.extracted for s in sc.values()))
            print(f"{name:8s} {label:18s} {R:10.1%} {P:11.1%} {sum(s.extracted for s in sc.values()):9d}")
    print(f"\nstudent selection latency per note: median {statistics.median(lat):.0f} ms, p90 {np.percentile(lat, 90):.0f} ms, max {max(lat):.0f} ms (n={len(lat)})")


if __name__ == "__main__":
    main()
