"""정답 노트 쌍 늘리기: 판정할 후보 쌍 풀 만들기 (정보 검색 평가의 pooling).

질의 노트마다 다른 주제의 노트 중 여러 방법의 상위와 무작위를 합친 후보를 만든다. 사람(또는 초안 작성자)이 쌍마다
'이어진다/아니다' 를 판정하면 이어진 쌍이 정답 쌍이 된다. 한 방법의 상위만 판정하면 그 방법에 유리한 정답만 생기므로
  S (Adamic-Adar) 상위 3 + TF-IDF 코사인 상위 3 + 무작위 2 (seed 고정)
를 합친다. 그래도 두 방법이 모두 못 찾는 쌍은 무작위로만 들어오므로, 이 정답으로 잰 재현율은 실제보다 조금 높다.

  uv run python experiments/gold_pairs/pool.py --queries 24 > data/private/gold/pair-pool.csv
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from app.api.v1.extract import select_concepts
    from app.cli import load_extractor, read_notes
    from app.core.config import EXTRACT_PROFILE, EXTRACT_PROFILES
    from app.services.concepts.markdown import nfc
    from app.services.links import build_corpus, related_notes
    from app.services.links.evaluate import _cos

    texts = read_notes(ROOT / "data/private/notes")
    topics = {nfc(r["note_id"]).strip(): r["topic"]
              for r in csv.DictReader((ROOT / "data/private/gold/topics.csv").open(encoding="utf-8-sig"))}
    a = argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=ROOT / "data/private/concepts/cache")
    ex = load_extractor(a)
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    types = [(c, c, c) for c in ("is_a", "part_of", "requires", "causes", "used_for", "contrasts_with", "example_of")]
    corpus = build_corpus(texts, ex, lambda ms: select_concepts(ms, profile), types, profile.limits)

    usable = sorted(n for n in texts if topics.get(n) not in (None, "목차"))
    by_topic = {}
    for n in sorted(usable, key=lambda n: hashlib.sha256(f"{args.seed}:{n}".encode()).hexdigest()):
        by_topic.setdefault(topics[n], []).append(n)
    queries, i = [], 0                               # 주제마다 돌아가며 골라 주제가 고르게 들어가게
    while len(queries) < args.queries and any(by_topic.values()):
        t = sorted(by_topic)[i % len(by_topic)]
        if by_topic[t]:
            queries.append(by_topic[t].pop(0))
        i += 1

    rng = random.Random(args.seed)
    w = csv.writer(sys.stdout)
    w.writerow(["query", "candidate", "query_topic", "candidate_topic", "sources"])
    for q in queries:
        other = [n for n in usable if topics[n] != topics[q]]
        s = [l.note_b for l in related_notes(corpus, q, k=len(texts)) if topics.get(l.note_b) != topics[q]][:3]
        tf = sorted(other, key=lambda n: (-round(_cos(corpus.notes[q].weights, corpus.notes[n].weights), 6), n))[:3]
        rnd = rng.sample(other, 2)
        src = {}
        for name, ns in (("S", s), ("tfidf", tf), ("random", rnd)):
            for n in ns:
                src.setdefault(n, []).append(name)
        for n, names in sorted(src.items()):
            w.writerow([q, n, topics[q], topics[n], "+".join(names)])


if __name__ == "__main__":
    main()
