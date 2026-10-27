"""별칭 병합 전체 흐름 시험: /embeddings (bge-m3) 최근접 후보 → /merge-decisions (merge-det-1) → 정답 별칭과 비교.

BE 가 할 일을 그대로 흉내 낸다: 개념 표기마다 벡터를 만들고, cosine 최근접 k 개를 병합 후보 쌍으로 묶어
/merge-decisions 에 similarity 와 함께 보낸다. 'same' 만 합친다. 설계 7장 기준: 잘못 합침 5% 이하.

  uv run --extra embeddings python experiments/alias_embeddings/merge_pipeline.py --include-draft [--k 5]
"""

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TOKEN = "experiment"
PROFILE = "embeddings-bge-m3-1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-draft", action="store_true")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import csv

    import numpy as np
    from fastapi.testclient import TestClient

    from app.core.config import Settings
    from app.main import create_app
    from app.services.concepts import ConceptExtractor, ExtractConfig, Idf
    from app.services.concepts.markdown import mask, nfc
    from app.services.relations.sentences import sentence_spans
    from experiments.alias_embeddings.compare import build_items

    keys, groups, texts = build_items(args.include_draft)
    topics = {nfc(r["note_id"]).strip(): r["topic"]
              for r in csv.DictReader((ROOT / "data/private/gold/topics.csv").open(encoding="utf-8"))}
    # 표기가 처음 나온 노트·문장 (build_items 와 같은 규칙)과 그 문장 속 실제 표기
    from app.services.concepts.gold import loose
    raw = {}
    for row in csv.DictReader((ROOT / "data/private/gold/concepts.csv").open(encoding="utf-8-sig")):
        for n in nfc(row["concept"]).split("|"):
            raw.setdefault(loose(n), nfc(row["note_id"]).strip())
    bodies, items = {}, []
    for k in keys:
        name, _ = texts[k]
        note = raw[k]
        body = bodies.setdefault(note, nfc((ROOT / "data/private/notes" / f"{note}.md").read_text(encoding="utf-8")))
        masked = mask(body)
        span = None
        for s, e in sentence_spans(body):
            m = re.search(re.escape(name), masked[s:e], re.I)
            if m:
                span = (s, e, body[s + m.start():s + m.end()])
                break
        items.append((k, name, note, span))

    ids = {n: str(uuid.uuid5(uuid.NAMESPACE_URL, "note:" + n)) for n in bodies}
    tids = {t: str(uuid.uuid5(uuid.NAMESPACE_URL, "topic:" + t)) for t in set(topics.values())}
    u16 = lambda s, i: len(s[:i].encode("utf-16-le")) // 2
    snap = lambda n: {"noteId": ids[n], "noteVersion": 1, "inputRevision": 1, "title": n[:200], "body": bodies[n],
                      "bodySha256": hashlib.sha256(bodies[n].encode()).hexdigest(),
                      "topic": {"id": tids[topics[n]], "name": topics[n]}}

    def ctx(n):
        return {"analysisJobId": str(uuid.uuid4()), "stageExecutionId": str(uuid.uuid4()),
                "userId": "c49bdcd3-9f13-5186-93c2-28eccf386140", "noteId": ids[n], "noteVersion": 1,
                "inputRevision": 1, "graphRevision": 1, "pipelineVersion": "alias-experiment"}

    def post(c, path, body):
        raw = json.dumps(body, ensure_ascii=False).encode()
        r = c.post("/internal/v1" + path, content=raw, headers={
            "Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json", "X-Request-Id": str(uuid.uuid4()),
            "X-Attempt-Id": str(uuid.uuid4()), "Idempotency-Key": body["context"]["stageExecutionId"],
            "X-Input-SHA256": hashlib.sha256(raw).hexdigest()})
        if r.status_code != 200:
            sys.exit(f"{path} {r.status_code} {r.text[:500]}")
        return r.json()

    idf = Idf.load(ROOT / "data/private/concepts/idf.json")
    app = create_app(Settings(service_token=TOKEN, concept_idf=Path("x"), embedding_profiles=(PROFILE,)),
                     extractor=ConceptExtractor(ExtractConfig(), idf))
    with TestClient(app) as c:
        any_note = next(iter(bodies))
        vecs = []
        for i in range(0, len(items), 32):
            chunk = items[i:i + 32]
            r = post(c, "/embeddings", {"context": ctx(any_note), "profileVersion": PROFILE, "expectedDimensions": 1024,
                                         "items": [{"itemId": f"t{i + j}", "text": it[1]} for j, it in enumerate(chunk)]})
            got = {x["itemId"]: x["vector"] for x in r["items"]}
            vecs += [got[f"t{i + j}"] for j in range(len(chunk))]
        v = np.array(vecs)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        sims = v @ v.T
        np.fill_diagonal(sims, -9)
        cand = set()
        for i in range(len(items)):
            for j in np.argsort(-sims[i])[: args.k]:
                cand.add((min(i, int(j)), max(i, int(j))))
        cand = sorted(p for p in cand if items[p[0]][3] and items[p[1]][3])

        # 요청 상한(쌍 32, 문서 10, 근거 40) 안에서 차례로 묶는다. pairId 는 cand 의 번호
        batches, cur, notes = [], [], set()
        for n, (i, j) in enumerate(cand):
            need = notes | {items[i][2], items[j][2]}
            if cur and (len(cur) >= 16 or len(need) > 10):
                batches.append(cur)
                cur, need = [], {items[i][2], items[j][2]}
            cur.append(n)
            notes = need
        if cur:
            batches.append(cur)

        decisions, reasons = {}, {}
        for batch in batches:
            ev, pairs = {}, []
            for n in batch:
                occ = []
                for x in cand[n]:
                    k, name, note, (s, e, surface) = items[x]
                    eid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"ev:{note}:{s}:{e}"))
                    ev[eid] = {"evidenceId": eid, "noteId": ids[note], "noteVersion": 1,
                               "bodySha256": hashlib.sha256(bodies[note].encode()).hexdigest(),
                               "startOffset": u16(bodies[note], s), "endOffset": u16(bodies[note], e),
                               "quote": bodies[note][s:e]}
                    occ.append({"conceptId": str(uuid.uuid5(uuid.NAMESPACE_URL, f"c:{k}")), "label": name,
                                "topicId": tids[topics[note]], "evidenceId": eid, "surface": surface})
                i, j = cand[n]
                pairs.append({"pairId": f"p{n}", "left": occ[0], "right": occ[1], "similarity": round(float(sims[i, j]), 6)})
            notes = sorted({items[x][2] for n in batch for x in cand[n]})
            r = post(c, "/merge-decisions", {"context": ctx(notes[0]), "profileVersion": "merge-det-1",
                                              "documents": [snap(m) for m in notes], "evidence": list(ev.values()),
                                              "pairs": pairs})
            decisions.update({d["pairId"]: d["decision"] for d in r["decisions"]})
            reasons.update({d["pairId"]: d["reason"] for d in r["decisions"]})

    n_pos = sum(1 for i in range(len(items)) for j in range(i + 1, len(items)) if groups[i] == groups[j])
    in_cand = sum(1 for i, j in cand if groups[i] == groups[j])
    same = [(i, j) for n, (i, j) in enumerate(cand) if decisions.get(f"p{n}") == "same"]
    unc = [(i, j) for n, (i, j) in enumerate(cand) if decisions.get(f"p{n}") == "uncertain"]
    tp = sum(groups[i] == groups[j] for i, j in same)
    print(f"표기 {len(items)}개 · 같은 개념 쌍 {n_pos} · 최근접 {args.k} 후보 쌍 {len(cand)} (그중 같은 개념 {in_cand}, "
          f"후보 재현율 {in_cand / n_pos:.0%})")
    print(f"merge-det-1 same {len(same)} → 맞음 {tp} · 잘못 합침 {len(same) - tp} "
          f"({(len(same) - tp) / len(same) if same else 0:.0%}) · 전체 재현율 {tp / n_pos:.0%} · uncertain {len(unc)} "
          f"(그중 같은 개념 {sum(groups[i] == groups[j] for i, j in unc)})")
    for i, j in same:
        if groups[i] != groups[j]:
            print("  잘못 합침:", items[i][1], "↔", items[j][1])
    if "-v" in sys.argv[1:] or True:
        for n, (i, j) in enumerate(cand):
            if groups[i] == groups[j] and decisions.get(f"p{n}") != "same":
                print(f"  놓침 [{decisions.get(f'p{n}')}] {items[i][1]} ↔ {items[j][1]} ({items[i][2][:12]}/{items[j][2][:12]}): "
                      f"{reasons.get(f'p{n}', '')[:110]}")
    print("  합친 것:", ", ".join(f"{items[i][1]}↔{items[j][1]}" for i, j in same if groups[i] == groups[j]))


if __name__ == "__main__":
    main()
