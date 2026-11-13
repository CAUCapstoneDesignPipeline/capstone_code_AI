"""학생(extract-student-1)의 오류를 유형으로 나눠 본다 (dev 정답 노트만: test·unseen 의 오답을 보면 그 노트가 평가 노트가 아니게 된다).

  uv run --env-file .env python experiments/distill/error_check.py [--examples 14]

잘못 고른 것(정답에 없는 채택)과 놓친 것(core 정답인데 채택 못 함)을 원인별로 센다. 교사 = LLM 이 뽑은 개념(캐시)이다.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=int, default=14)
    args = ap.parse_args()

    from app.cli import load_extractor
    from app.services.concepts.config import STOPWORDS
    from app.services.concepts.gold import loose
    from app.services.concepts.student import EN_STOP, Student, rows, suppress_nested
    from experiments.distill.features import teacher_concepts
    from experiments.llm_only.baseline import sets_available

    ex = load_extractor(argparse.Namespace(idf=ROOT / "data/private/concepts/idf.json", cache=None))
    s = Student.load()
    gold, texts = sets_available(["dev"])["dev"]
    fp, fn = Counter(), Counter()
    ex_fp, ex_fn = defaultdict(list), defaultdict(list)
    n_kept = n_fp = n_core = n_miss = 0
    for n in sorted(gold):
        text, rs = rows(ex, texts[n])
        p = np.round(s.probabilities(np.array([r["x"] for r in rs], float)), 6)
        kept = suppress_nested(rs, p, s.threshold)
        kept_set = set(kept)
        forms_of = lambda r: {loose(r["surface"]), *(loose(text[a:b]) for a, b in r["spans"])}
        gold_all = set().union(*(g.aliases for g in gold[n]))
        teach = set().union(*[k for _, k in teacher_concepts(texts[n])]) if teacher_concepts(texts[n]) else set()
        # --- 잘못 고른 것
        for i in kept:
            r = rs[i]
            n_kept += 1
            if forms_of(r) & gold_all:
                continue
            n_fp += 1
            lab, words = r["surface"], [w for w in lab.lower().replace("-", " ").split() if w] if (lab := r["surface"]) else []
            if forms_of(r) & teach:
                kind = "teacher keeps it too (gold gap or label disagreement)"
            elif any(loose(lab) != g and loose(lab) in g or g in loose(lab) and len(g) >= 4 for g in gold_all):
                kind = "variant of a gold concept (longer/shorter form)"
            elif len(words) == 1 and (words[0] in STOPWORDS or words[0] in EN_STOP or len(lab) <= 3):
                kind = "generic single word"
            elif any(ch.isdigit() for ch in lab) or any(not ch.isalnum() and ch not in " -/+.'" for ch in lab):
                kind = "digits/symbols"
            elif words and (words[0] in EN_STOP or words[-1] in EN_STOP or words[0] in STOPWORDS or words[-1] in STOPWORDS):
                kind = "phrase with a generic/function word at an edge"
            elif len(words) == 1:
                kind = "other single word"
            else:
                kind = "other multi-word phrase"
            fp[kind] += 1
            ex_fp[kind].append(f"{lab} ({p[i]:.2f})")
        # --- 놓친 core
        cand = {}
        for i, r in enumerate(rs):
            for f in forms_of(r):
                cand.setdefault(f, i)
        for g in gold[n]:
            if g.level != "core":
                continue
            n_core += 1
            if any(i in kept_set and forms_of(rs[i]) & g.aliases for i in range(len(rs)) if forms_of(rs[i]) & g.aliases):
                continue
            n_miss += 1
            hit = [cand[a] for a in g.aliases if a in cand]
            if not hit:
                kind = "never a candidate (generator ceiling)"
            else:
                best = max(hit, key=lambda i: p[i])
                kind = "candidate dropped as nested inside a longer kept one" if p[best] >= s.threshold else ("probability just under the threshold (0.25-0.45)" if p[best] >= 0.25 else "probability far under the threshold (<0.25)")
                ex_fn[kind].append(f"{g.label} ({p[best]:.2f})")
                fn[kind] += 1
                continue
            fn[kind] += 1
            ex_fn[kind].append(g.label)
    print(f"dev: kept {n_kept}, of which not in gold {n_fp} ({n_fp / n_kept:.0%}); core gold {n_core}, missed {n_miss} ({n_miss / n_core:.0%})\n")
    print("== wrongly kept (not in gold), by type ==")
    for k, v in fp.most_common():
        print(f"  {v:4d} ({v / n_fp:4.0%})  {k}\n        e.g. " + "; ".join(ex_fp[k][:args.examples]))
    print("\n== missed core concepts, by cause ==")
    for k, v in fn.most_common():
        print(f"  {v:4d} ({v / n_miss:4.0%})  {k}\n        e.g. " + "; ".join(ex_fn[k][:args.examples]))


if __name__ == "__main__":
    main()
