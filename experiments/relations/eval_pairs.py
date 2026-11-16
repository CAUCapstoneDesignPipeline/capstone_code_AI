"""출고하는 pairs-1 (PairRelations) 을 정답 노트에서 실제로 불러 지연을 잰다 (캐시 없이 OpenAI 를 부른다: 서비스와 같은 코드·고정 임계값).

  uv run --env-file .env python experiments/relations/eval_pairs.py [--sets dev test unseen]

개념은 정답 개념으로 고정하고 노트를 한 번에 하나씩 보내 관계 단계의 처음부터 끝까지(후보 쌍·특징·요청·분류·근거) 지연을 잰다. 노트 원문이 OpenAI 로 나간다.
이 노트들은 모델의 학습 노트라 재현율·정밀도는 낙관적이다 — 일반화 성능은 pair_model.py 의 노트 단위 교차검증 값(docs/eval/relations.md)을 본다. 여기서는 지연과 출력 크기를 잰다.
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

    from app.dev.router import _policy_types
    from app.services.concepts.judge import DEFAULT_MODEL
    from app.services.relations import Limits
    from app.services.relations.candidates import candidate_pairs
    from app.services.relations.pairs import DecisionsPairs, PairModel, PairRelations
    from experiments.relations import harness
    from experiments.relations.pool import canon

    pr = PairRelations(DecisionsPairs(openai.OpenAI(max_retries=1, timeout=30.0), DEFAULT_MODEL), PairModel.load())
    policy = list(_policy_types())
    data = harness.load(args.sets)
    print(f"{pr.version} · {pr.decisions.model} · 임계값 {pr.model.threshold}\n")
    print(f"{'set':8s} 노트  쌍(평균)  낸 관계(평균)  근거(평균)  지연 중앙값 / p90 / 최대 (s)   [학습 노트라 재현율은 낙관적: 정답 관계 재현율]")
    lat_all = []
    for name, notes in data.items():
        lat, n_rel, n_ev, n_pairs, hit, gold = [], [], [], [], 0, 0
        for note in notes.values():
            ms = harness.mentions_for(note)
            n_pairs.append(len(candidate_pairs(note.text, ms)))
            t0 = time.perf_counter()
            ev, rels = pr.extract(note.text, ms, policy, Limits())
            lat.append(time.perf_counter() - t0)
            n_rel.append(len(rels)); n_ev.append(len(ev))
            got = {canon((r.source, r.target, r.code)) for r in rels}
            want = {canon((g["source"], g["target"], g["type"])) for g in note.gold}
            hit += len(got & want); gold += len(want)
        lat_all += lat
        print(f"{name:8s} {len(notes):3d} {np.mean(n_pairs):8.0f} {np.mean(n_rel):12.1f} {np.mean(n_ev):10.1f}   {statistics.median(lat):.2f} / {np.percentile(lat, 90):.2f} / {max(lat):.2f}   {hit / gold:.0%}", flush=True)
    print(f"\n전체 {len(lat_all)}개 노트: 지연 중앙값 {statistics.median(lat_all):.2f}s, p90 {np.percentile(lat_all, 90):.2f}s, 최대 {max(lat_all):.2f}s")


if __name__ == "__main__":
    main()
