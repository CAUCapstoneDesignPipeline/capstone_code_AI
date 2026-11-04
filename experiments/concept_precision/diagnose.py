"""개념 추출 오답을 단어가 아니라 유형으로 나눠 본다 (dev 노트만). 처음 보는 노트에도 통하는 고침을 찾기 위해서다.

추출된 개념마다 특징을 붙이고, 특징 값마다 정답률(정밀도)과 개수를 낸다. 정답과 맞는지는 ai concepts eval 과 같은 규칙.
  uv run python experiments/concept_precision/diagnose.py [--split dev]
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def features(c, ms, text, idf, n_docs):
    label = c.label
    words = label.split()
    hangul, latin = bool(re.search(r"[가-힣]", label)), bool(re.search(r"[A-Za-z]", label))
    struct = max(m.score for m in ms) - 0  # 아래에서 구조 가중치를 따로 본다
    df = idf.df.get(ms[0].key, 0) if hasattr(idf, "df") else 0
    return {
        "script": "ko" if hangul and not latin else "en" if latin and not hangul else "mixed",
        "words": "1" if len(words) == 1 else "2" if len(words) == 2 else "3+",
        "source": "markdown" if any(m.method.value == "markdown" for m in ms) else "tfidf",
        "mentions": "1" if len(c.mentions) == 1 else "2-3" if len(c.mentions) <= 3 else "4+",
        "df": "rare(≤2)" if df <= 2 else "mid(3-10)" if df <= 10 else "common(>10)",
        "confidence": f"{min(4, int(c.confidence * 10)) / 10:.1f}+",
        "len": "short(≤3)" if len(label) <= 3 else "mid" if len(label) <= 12 else "long(>12)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    args = ap.parse_args()

    from app.api.v1.extract import select_concepts, to_concepts
    from app.cli import load_extractor, read_notes
    from app.core.config import EXTRACT_PROFILE, EXTRACT_PROFILES
    from app.services.concepts.gold import load_gold, loose
    from app.services.concepts.markdown import nfc
    from app.services.evaluation import assign_splits

    gold_path = ROOT / "data/private/gold/concepts.csv"
    gold = load_gold(gold_path, include_draft=True)
    rows = list(csv.DictReader(gold_path.open(encoding="utf-8-sig")))
    splits = assign_splits(gold, {nfc(r["note_id"]).strip(): (r.get("split") or "").strip() for r in rows})
    notes = sorted(n for n in gold if args.split == "all" or splits[n] == args.split)
    texts = read_notes(ROOT / "data/private/notes", notes)
    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json",
                                           cache=ROOT / "data/private/concepts/cache"))
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    table = defaultdict(lambda: [0, 0])          # (특징, 값) → [맞음, 전체]
    per_note = []
    for n in notes:
        known = set().union(*(g.aliases for g in gold[n]))
        sel = select_concepts(ex.extract(n, texts[n]), profile)
        concepts, _ = to_concepts(sel, texts[n])
        hit_n = 0
        for c, (k, ms) in zip(concepts, sel.items()):
            ok = bool({loose(c.label), *(loose(m.quote) for m in c.mentions)} & known)
            hit_n += ok
            for f, v in features(c, ms, texts[n], ex.idf, 0).items():
                table[(f, v)][0] += ok
                table[(f, v)][1] += 1
        per_note.append((len(concepts), len(gold[n]), len(texts[n]), hit_n))
    print(f"{args.split} 노트 {len(notes)}개 · 추출 {sum(p[0] for p in per_note)} · 정답 {sum(p[1] for p in per_note)}\n")
    print(f"{'특징':<12}{'값':<14}{'정밀도':>7}{'개수':>7}")
    for (f, v), (ok, tot) in sorted(table.items()):
        print(f"{f:<12}{v:<14}{ok / tot:7.0%}{tot:7d}")
    print("\n노트 길이 · 추출 수 · 정답 수 · 맞은 수")
    for (n_ex, n_gold, length, hit), n in sorted(zip(per_note, notes), key=lambda x: x[0][2]):
        print(f"  {length:6d}자  추출 {n_ex:3d}  정답 {n_gold:3d}  맞음 {hit:3d}  {n[:30]}")


if __name__ == "__main__":
    main()
