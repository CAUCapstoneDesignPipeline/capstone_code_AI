"""한 번에 모든 노트의 쌍·특징·캐시 라벨을 모아 pickle 로 둔다 (다시 계산하지 않게).

  uv run --env-file .env python experiments/relations/dataset.py

개념은 extract-det-3 가 고른 것 (캐시된 LLM 라벨이 그 개념 쌍으로 만들어졌다). 과목 노트 141개 (정답 있는 41 + 없는 100).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.cache import scratch
OUT = scratch("rel_dataset.pkl")


def main():
    from app.api.v1.extract import select_concepts
    from app.cli import load_extractor
    from app.core.config import EXTRACT_PROFILES
    from app.dev.router import playground_policy
    from app.main import role_llm_config
    from app.services.concepts.markdown import nfc
    from app.services.llm import make_llm
    from app.services.relations.typer import RelationTyper
    from experiments.relations.pairs import CACHE, cached_label, candidate_pairs, raw_features

    pol = playground_policy()
    types = [(t["code"], t["label"], t["description"]) for t in pol["relationTypes"]]
    codes = {c for c, _, _ in types}
    sig = json.dumps(sorted(types), ensure_ascii=False)
    typer = RelationTyper(make_llm(role_llm_config("relation", "low")), CACHE)
    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    from app.services.concepts.gold import load_gold
    from app.services.evaluation import assign_splits
    import csv
    gp = ROOT / "data/private/gold/concepts.csv"
    gold = load_gold(gp, include_draft=True)
    rows_ = list(csv.DictReader(gp.open(encoding="utf-8-sig")))
    splits = assign_splits(gold, {nfc(r["note_id"]).strip(): (r.get("split") or "").strip() for r in rows_})
    out, n_pairs, n_lab = [], 0, 0
    for p in sorted((ROOT / "data/private/notes").glob("*.md")):
        name = nfc(p.stem)
        text = nfc(p.read_text(encoding="utf-8"))
        sel = select_concepts(ex.extract(name, text), EXTRACT_PROFILES["extract-det-3"])
        for pr in candidate_pairs(text, sel, codes):
            if pr["decided"]:
                continue
            ss, se = pr["span"]
            lab = cached_label(typer, sig, text[ss:se], pr["a"].surface, pr["b"].surface)
            raw = raw_features(ex.kiwi, text, pr)
            out.append(dict(note=name, split=splits.get(name, "none"), a=pr["a"].surface, b=pr["b"].surface, sentence=text[ss:se], raw=raw,
                            label=None if lab is None else (lab["code"], bool(lab["a_is_source"]), float(lab["confidence"]))))
            n_pairs += 1
            n_lab += lab is not None
    pickle.dump(out, open(OUT, "wb"))
    print(f"{len(out)} pairs from course notes, {n_lab} with cached LLM labels ({n_lab / max(1, len(out)):.0%}) → {OUT.name}")


if __name__ == "__main__":
    main()
