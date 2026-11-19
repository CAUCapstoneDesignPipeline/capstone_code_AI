"""관계 평가 공통: 정답 읽기, 개념 언급 찾기, 채점, 후보 쌍이 닿는 범위.

정답은 experiments/relations/gold.py 가 만든 data/private/gold-relations/<집합>/<노트>.json (AI 초안 + 다른 모델의 검토, 사람 검토 전)이다.
관계를 평가할 때는 개념을 정답 개념으로 고정한다 (개념 추출의 오차가 섞이지 않게). 개념의 표기는 정답 csv 의 원래 표기(`|` 로 이은 별칭)다.
채점 (한 집합의 모든 노트를 합친 micro):
  strict    쌍·유형·방향이 모두 맞아야 한다 (contrasts_with·analogous_to 는 두 끝의 순서를 따지지 않는다)
  type      쌍·유형이 맞으면 된다 (방향 무시)
  pair      두 개념 사이에 관계가 있다고만 맞히면 된다 (유형 무시)
재현율 = 정답 관계 중 맞힌 비율, 정밀도 = 낸 관계 중 맞은 비율. "draft" 재현율은 검토자가 더한 관계를 뺀, 초안을 검토자가 받아들인 정답만으로 센 값이다.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/private/gold-relations"
SYMMETRIC = {"contrasts_with", "analogous_to"}
TYPES = ["is_a", "part_of", "causes", "requires", "used_for", "contrasts_with", "example_of", "analogous_to"]


@dataclass
class Note:
    name: str
    text: str
    concepts: dict[str, list[str]]                  # 라벨 → 원래 표기들 (첫 번째가 라벨)
    gold: list[dict] = field(default_factory=list)  # source, target, type, evidence, basis, origin


def _raw_forms():
    """노트 → 라벨(loose) → 표기들. 과목 csv 와 논문 리뷰 csv 에서."""
    from app.services.concepts.gold import loose
    from app.services.concepts.markdown import nfc
    out = {}
    files = [ROOT / "data/private/gold/concepts.csv", *sorted((ROOT / "data/private/gold-unseen").glob("final-*.csv"))]
    for f in files:
        for r in csv.DictReader(f.open(encoding="utf-8-sig")):
            names = [n.strip() for n in nfc(r["concept"]).split("|") if loose(n)]
            if names:
                out.setdefault(nfc(r["note_id"]).strip(), {}).setdefault(loose(names[0]), names)
    return out


def load(sets=("dev", "test", "unseen")) -> dict[str, dict[str, Note]]:
    from app.services.concepts.gold import loose
    from experiments.llm_only.baseline import sets_available
    raw = _raw_forms()
    data = sets_available(list(sets))
    out = {}
    for name, (gold_concepts, texts) in data.items():
        out[name] = {}
        for n in sorted(gold_concepts):
            path = GOLD / name / f"{n}.json"
            if not path.exists():
                continue
            g = json.loads(path.read_text(encoding="utf-8"))
            forms = {c: raw.get(n, {}).get(loose(c), [c]) for c in g["concepts"]}
            out[name][n] = Note(n, texts[n], forms, g["relations"])
    return out


def mentions_for(note: Note, version: int = 1):
    """정답 개념의 언급 (원문에서 표기를 찾는다): 라벨 → [Mention]. 서비스의 selected 와 같은 모양."""
    from app.services.concepts.llm_concepts import _forms_regex
    from app.services.concepts.markdown import Utf16Index
    from app.services.concepts.schema import Mention, Method, Span
    u16 = Utf16Index(note.text)
    out = {}
    for label, forms in note.concepts.items():
        spans = set()
        for f in forms:
            rx = _forms_regex(f)
            if rx is not None:
                spans |= {m.span() for m in rx.finditer(note.text)}
        # 서로 겹치는 언급은 긴 것만 둔다 ("Hand Pose Estimation" 안의 "Pose")
        kept = []
        for s, e in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
            if not kept or s >= kept[-1][1]:
                kept.append((s, e))
        if kept:
            out[label] = [Mention(span=Span(note.name, version, u16(s), u16(e)), surface=note.text[s:e], key=label, method=Method.LLM,
                                  method_version="gold", score=2.0) for s, e in kept]
    return out


def _key(r):
    s, t, ty = r
    return (frozenset((s, t)) if ty in SYMMETRIC else (s, t)), ty


def score_note(gold: list[dict], pred: list[tuple[str, str, str]]):
    """(strict, type, pair) 각각의 (맞힌 정답 수, 맞은 예측 수) 와 draft 정답 수·맞힌 draft 정답 수."""
    pred = list(dict.fromkeys(pred))
    g = [(r["source"], r["target"], r["type"]) for r in gold]
    origin = [r["origin"] for r in gold]
    strict_g = {_key(r) for r in g}
    strict_p = {_key(r) for r in pred}
    type_key = lambda r: (frozenset((r[0], r[1])), r[2])
    pair_key = lambda r: frozenset((r[0], r[1]))
    res = {}
    for name, fn in (("strict", _key), ("type", type_key), ("pair", pair_key)):
        gk, pk = {fn(r) for r in g}, {fn(r) for r in pred}
        res[name] = dict(gold=len(gk), pred=len(pk), hit_gold=len(gk & pk), hit_pred=len(pk & gk))
    draft = [r for r, o in zip(g, origin) if o == "draft"]
    dk = {_key(r) for r in draft}
    res["draft"] = dict(gold=len(dk), hit_gold=len(dk & strict_p))
    return res


def aggregate(per_note: list[dict]):
    tot = {}
    for res in per_note:
        for k, v in res.items():
            t = tot.setdefault(k, {})
            for kk, vv in v.items():
                t[kk] = t.get(kk, 0) + vv
    out = {}
    for k in ("strict", "type", "pair"):
        t = tot[k]
        r, p = t["hit_gold"] / max(1, t["gold"]), t["hit_pred"] / max(1, t["pred"])
        out[k] = (r, p, 2 * r * p / (r + p + 1e-9))
    out["draft_recall"] = tot["draft"]["hit_gold"] / max(1, tot["draft"]["gold"])
    out["n_pred"], out["n_gold"] = tot["strict"]["pred"], tot["strict"]["gold"]
    return out


def reach(note: Note):
    """정답 관계 중 후보 쌍 규칙(같은 문장·언급 3칸 안·사이 40자 이하)이 닿는 것, 문장 하나에 둘 다 나오는 것, 마크다운 구조가 닿는 것 (정답 인덱스 집합)."""
    from app.services.relations.extract import Limits, _py_offsets
    from app.services.relations.sentences import sentence_spans
    from app.services.relations.structure import structural_candidates
    ms = mentions_for(note)
    occs = _py_offsets(note.text, ms)
    sentences = sentence_spans(note.text)
    pool, same_sentence = set(), set()
    lim = Limits()
    for ss, se in sentences:
        inside, last = [], -1
        for o in occs:
            if ss <= o.start and o.end <= se and o.start >= last:
                inside.append(o)
                last = o.end
        for i, a in enumerate(inside):
            for j in range(i + 1, len(inside)):
                b = inside[j]
                if a.key == b.key:
                    continue
                same_sentence.add(frozenset((a.key, b.key)))
                if j < i + 1 + lim.max_pair_distance and len(note.text[a.end:b.start]) <= lim.max_between_chars:
                    pool.add(frozenset((a.key, b.key)))
    structure = {frozenset((c.source, c.target)) for c in structural_candidates(note.text, occs)}
    pairs = [frozenset((r["source"], r["target"])) for r in note.gold]
    return dict(pool={i for i, p in enumerate(pairs) if p in pool}, sentence={i for i, p in enumerate(pairs) if p in same_sentence},
                structure={i for i, p in enumerate(pairs) if p in structure}, unseen_mention={i for i, r in enumerate(note.gold) if r["source"] not in ms or r["target"] not in ms})
