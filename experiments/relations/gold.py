"""관계 정답 만들기 (AI 초안 + 다른 모델의 검토): 정답 개념이 있는 노트 59개(과목 dev 23 · test 18, 논문 리뷰 18)에서 개념 사이 관계를 적는다.

  uv run --env-file .env python experiments/relations/gold.py [--sets dev test unseen] [--workers 4]

1. 초안 (Opus 5.5): 노트 원문 + 그 노트의 정답 개념 목록 → 노트가 말하는 관계 (source, target, 유형 8개 중 하나, 원문에서 그대로 옮긴 근거, 근거 방식).
2. 검토 (Sonnet 5.5, 초안과 다른 모델): 초안의 관계마다 받아들일지·유형을 고칠지·뺄지, 빠뜨린 관계는 없는지.
3. 코드 검증: 두 끝이 정답 개념이어야 하고, 근거는 원문에 있어야 하며, 자기 자신·중복은 버린다. 대칭 유형(contrasts_with·analogous_to)은 두 끝의 순서를 고정한다.
결과는 data/private/gold-relations/<집합>/<노트>.json (git 에 올라가지 않는다). 노트 내용이 LLM 제공자로 나간다. 정답은 AI 가 만든 것이라 사람이 검토하기 전의 수치다.
관계 유형과 설명은 서비스 policy(app/dev/router.py 의 _policy_types)와 같다. 판정 기준은 "노트가 그렇게 말하는가"다 (상식으로 만들지 않는다).
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data/private/gold-relations"
CACHE = ROOT / "data/private/cache/relations-gold"
SYMMETRIC = {"contrasts_with", "analogous_to"}
TYPES = ["is_a", "part_of", "causes", "requires", "used_for", "contrasts_with", "example_of", "analogous_to"]
DRAFT_MODEL, REVIEW_MODEL = "claude-opus-5-5", "claude-sonnet-5-5"

RULES = """판정 기준은 "노트가 그렇게 말하는가"다. 상식이나 노트 밖 지식으로 관계를 만들지 않는다. 두 개념이 같이 나오거나 나열되기만 한 것은 관계가 아니다.
- source·target 은 아래 개념 목록의 표기 그대로 쓴다. 목록에 없는 것은 쓰지 않는다.
- type 은 관계 유형 중 하나다. 한 쌍에 서로 다른 유형이 모두 성립하면 각각 적는다. 방향이 있는 유형은 주어가 source, 목적어가 target 이다.
  contrasts_with·analogous_to 는 방향이 없다 (순서는 아무거나).
- evidence 는 그 관계를 말하는 노트 원문의 한 문장(또는 한 줄)을 **글자 하나 바꾸지 않고 그대로** 옮긴 것이다.
- basis: sentence = 한 문장 안에서 두 개념의 관계가 서술된다. structure = 문장이 아니라 제목·목록·표 구조가 관계를 드러낸다
  (예: "## TCP" 아래 목록의 항목들이 TCP 의 구성 요소, 비교 표의 두 열이 서로 대비). structure 일 때 evidence 는 구조를 보여 주는 줄이다.

관계 유형:
{types}"""
TYPE_LINES = "\n".join([
    "- is_a: 주어가 목적어의 한 종류다 (예: TCP는 전송 프로토콜이다)", "- part_of: 주어가 목적어의 구성 요소다",
    "- causes: 주어가 목적어를 일으키거나 결과로 낳는다", "- requires: 주어가 성립하거나 동작하려면 목적어가 필요하다",
    "- used_for: 주어가 목적어를 위해 쓰인다", "- contrasts_with: 노트가 주어와 목적어를 대비하거나 비교한다",
    "- example_of: 주어가 목적어의 구체적 사례다", "- analogous_to: 두 개념이 비슷한 원리로 동작한다"])

REL = {"type": "object", "additionalProperties": False, "required": ["source", "target", "type", "evidence", "basis"],
       "properties": {"source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string", "enum": TYPES},
                      "evidence": {"type": "string"}, "basis": {"type": "string", "enum": ["sentence", "structure"]}}}
DRAFT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["relations"], "properties": {"relations": {"type": "array", "items": REL}}}
REVIEW_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["reviews", "missed"], "properties": {
    "reviews": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "verdict", "type", "swap"], "properties": {
        "id": {"type": "integer"}, "verdict": {"type": "string", "enum": ["accept", "reject"]},
        "type": {"type": "string", "enum": TYPES}, "swap": {"type": "boolean"}}}},
    "missed": {"type": "array", "items": REL}}}

DRAFT_SYSTEM = ("너는 공부 노트를 읽고 개념 사이 관계를 정답으로 표시하는 사람이다. 노트가 말하는 관계를 빠짐없이, 그러나 노트가 말하지 않는 것은 하나도 넣지 않고 적는다.\n"
                + RULES.format(types=TYPE_LINES))
REVIEW_SYSTEM = ("너는 다른 사람이 노트에서 뽑은 개념 관계 목록을 검토한다. 각 관계가 노트에 실제로 쓰여 있고 유형이 맞는지 본다.\n"
                 "- verdict accept: 노트가 그 관계를 말하고 유형이 맞다 (type 에 쓰인 유형을 그대로 돌려준다). 유형이 틀렸지만 다른 유형으로는 성립하면 accept 하고 type 에 맞는 유형을 적는다.\n"
                 "- reject: 노트가 그 관계를 말하지 않는다 (상식이나 나열만으로 만든 것, 근거가 그 관계를 말하지 않는 것, 어느 유형으로도 성립하지 않는 것).\n"
                 "- swap: 방향이 거꾸로면 true (방향이 있는 유형만). 판정은 엄격하게 한다 — 확신이 없으면 reject 다.\n"
                 "- missed: 목록에 없는데 노트가 분명히 말하는 관계만 적는다 (적는 형식은 아래와 같다).\n" + RULES.format(types=TYPE_LINES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev", "test", "unseen"], choices=["dev", "test", "unseen"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    from dataclasses import replace

    from app.main import role_llm_config
    from app.services.concepts.gold import loose
    from app.services.llm import LLMError, make_llm
    from experiments.llm_only.baseline import sets_available

    def llm_for(model):
        cfg = replace(role_llm_config("relation", "medium"), provider="anthropic", model=model, effort="medium")
        return make_llm(cfg)

    drafter, reviewer = llm_for(DRAFT_MODEL), llm_for(REVIEW_MODEL)
    CACHE.mkdir(parents=True, exist_ok=True)

    def ask(llm, system, user, schema, name):
        key = hashlib.sha256(json.dumps([llm.model, system, user], ensure_ascii=False).encode()).hexdigest()
        p = CACHE / f"{key}.json"
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        for attempt in range(5):
            try:
                res = llm.complete_json(system, user, schema, name).data
                break
            except LLMError as e:
                if e.kind == "refusal":
                    res = {"relations": [], "reviews": [], "missed": [], "refused": True}
                    break
                if e.kind not in ("timeout", "provider", "rate_limit") or attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        p.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        return res

    def concept_list(concepts):
        return "\n".join(f"- {c.label}" for c in concepts)

    def process(item):
        name, note, text, concepts = item
        base = f"<개념 목록>\n{concept_list(concepts)}\n</개념 목록>\n\n<노트>\n{text}\n</노트>"
        draft = ask(drafter, DRAFT_SYSTEM, base, DRAFT_SCHEMA, "relations")
        rels = draft.get("relations", [])
        listing = "\n".join(f"[{i}] {r['source']} --{r['type']}--> {r['target']} | 근거: {r['evidence']}" for i, r in enumerate(rels))
        review = ask(reviewer, REVIEW_SYSTEM, base + f"\n\n<검토할 관계>\n{listing}\n</검토할 관계>", REVIEW_SCHEMA, "review") if rels or True else {}
        return name, note, text, concepts, rels, review

    data = sets_available(args.sets)
    items = [(name, n, texts[n], gold[n]) for name, (gold, texts) in data.items() for n in sorted(gold)]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(process, items))

    flat = lambda s: loose(s)
    summary = {}
    for name, note, text, concepts, rels, review in results:
        by_alias = {a: c for c in concepts for a in c.aliases}
        ftext = flat(text)
        verdicts = {r["id"]: r for r in review.get("reviews", [])}
        final, stats = {}, summary.setdefault(name, dict(notes=0, drafted=0, accepted=0, retyped=0, swapped=0, rejected=0, added=0, ungrounded=0, bad_end=0))
        stats["notes"] += 1

        def add(r, origin):
            a, b = by_alias.get(flat(r["source"])), by_alias.get(flat(r["target"]))
            if a is None or b is None or a is b:
                stats["bad_end"] += 1
                return False
            if flat(r["evidence"]) not in ftext:
                stats["ungrounded"] += 1
                return False
            src, tgt = (a, b)
            if r["type"] in SYMMETRIC and src.label > tgt.label:
                src, tgt = tgt, src
            key = (src.label, tgt.label, r["type"])
            if key in final:
                return False
            final[key] = {"source": src.label, "target": tgt.label, "type": r["type"], "evidence": r["evidence"], "basis": r["basis"], "origin": origin}
            return True

        for i, r in enumerate(rels):
            stats["drafted"] += 1
            v = verdicts.get(i)
            if v is None or v["verdict"] == "reject":
                stats["rejected"] += 1
                continue
            r = dict(r)
            if v["type"] != r["type"]:
                stats["retyped"] += 1
                r["type"] = v["type"]
            if v["swap"] and r["type"] not in SYMMETRIC:
                stats["swapped"] += 1
                r["source"], r["target"] = r["target"], r["source"]
            if add(r, "draft"):
                stats["accepted"] += 1
        for r in review.get("missed", []):
            if add(r, "added"):
                stats["added"] += 1
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{note}.json").write_text(json.dumps({"note": note, "concepts": [c.label for c in concepts], "relations": list(final.values())}, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"초안 {DRAFT_MODEL} → 검토 {REVIEW_MODEL}\n")
    for name, s in summary.items():
        total = s["accepted"] + s["added"]
        print(f"{name:7s} 노트 {s['notes']:3d} · 초안 {s['drafted']:4d} → 받아들임 {s['accepted']:4d} (유형 고침 {s['retyped']}, 방향 바꿈 {s['swapped']}) · 뺌 {s['rejected']:4d} · 검토자가 더함 {s['added']:3d} · "
              f"원문에 없는 근거 {s['ungrounded']} · 개념 목록 밖 {s['bad_end']} → 정답 관계 {total}개 (노트당 {total / s['notes']:.1f})")


if __name__ == "__main__":
    main()
