"""관계 학습 자료: 한 문장 안의 개념 쌍(`extract_relations` 가 LLM typer 에 묻는 쌍과 같은 쌍)과 그 특징, 캐시된 LLM 라벨.

쌍을 고르는 규칙은 app/services/relations/extract.py 와 같다 (같은 문장, 언급 순서로 3칸 안, 사이 40자 이하, 규칙이 정하지 못한 쌍).
특징은 문장 속 모양만 본다: 사이 글자·낱말, 두 개념 앞뒤 조사, 사이 형태소의 품사·어간, 강조·제목·목록 여부. 개념이 무엇인지는 보지 않는다.
라벨은 이미 캐시에 있는 LLM typer 결과(`relation-typing-1`)를 같은 key 로 찾는다 (LLM 을 새로 부르지 않는다). 캐시에 없는 쌍은 라벨이 없다.
"""

import json
import re
from collections import Counter
from pathlib import Path

from app.services.relations.extract import Limits, _py_offsets
from app.services.relations.rules import classify, code_for
from app.services.relations.sentences import sentence_spans
from app.services.relations.typer import PairItem

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/private/cache/relations"
TYPES = ["is_a", "part_of", "causes", "requires", "used_for", "contrasts_with", "example_of", "analogous_to"]
SYMMETRIC = {"contrasts_with", "analogous_to"}


def candidate_pairs(text: str, mentions: dict, codes: set[str], limits: Limits = Limits()):
    """(문장 번호, 문장 범위, a, b, 규칙이 정했는가) — extract_relations 와 같은 쌍."""
    occs = _py_offsets(text, mentions)
    out = []
    for s_idx, (ss, se) in enumerate(sentence_spans(text)):
        inside, last_end = [], -1
        for o in occs:
            if ss <= o.start and o.end <= se and o.start >= last_end:
                inside.append(o)
                last_end = o.end
        for i, a in enumerate(inside):
            for j in range(i + 1, min(len(inside), i + 1 + limits.max_pair_distance)):
                b = inside[j]
                if a.key == b.key or len(text[a.end:b.start]) > limits.max_between_chars:
                    continue
                rule = classify(text[a.end:b.start], text[b.end:min(se, b.end + 20)])
                decided = bool(rule and code_for(rule.semantic, codes))
                out.append(dict(s_idx=s_idx, span=(ss, se), a=a, b=b, i=i, j=j, n_inside=len(inside), decided=decided))
    return out


def cached_label(typer, policy_sig: str, sentence: str, a: str, b: str):
    """캐시된 LLM 판정 {"code", "a_is_source", "confidence"} 또는 None."""
    return typer._cache_get(typer._key(policy_sig, PairItem("x", sentence, a, b)))


def kiwi_tokens(kiwi, s: str) -> list[tuple[str, str]]:
    return [(t.form, t.tag) for t in kiwi.tokenize(s)] if s.strip() else []


def raw_features(kiwi, text: str, p: dict) -> dict:
    """한 쌍의 날 특징 (낱말 집합은 학습 때 어휘를 정해 숫자로 바꾼다)."""
    ss, se = p["span"]
    a, b = p["a"], p["b"]
    between = text[a.end:b.start]
    after_a = kiwi_tokens(kiwi, text[a.end:min(b.start, a.end + 12)])[:2]
    before_b = kiwi_tokens(kiwi, text[max(a.end, b.start - 12):b.start])[-2:]
    after_b = kiwi_tokens(kiwi, text[b.end:min(se, b.end + 14)])[:3]
    btoks = kiwi_tokens(kiwi, between)
    line_start = text.rfind("\n", 0, a.start) + 1
    line = text[line_start:a.start]
    sentence = text[ss:se]
    return dict(
        num=[len(between), len(between.split()), p["j"] - p["i"], len(sentence), ss / max(1, len(text)), p["n_inside"], len(a.surface), len(b.surface),
             sum(t.startswith("V") for _, t in btoks), sum(t.startswith("N") for _, t in btoks), sum(t.startswith("J") for _, t in btoks),
             int("," in between), int("(" in between or ")" in between), int(bool(re.match(r"\s*(#|[-*•]|\d+[.)])", text[line_start:line_start + 4]))),
             int(text[max(0, a.start - 2):a.start] == "**"), int(text[max(0, b.start - 2):b.start] == "**"),
             int(bool(re.search(r"[가-힣]", a.surface))), int(bool(re.search(r"[가-힣]", b.surface))), int(a.surface.isascii()), int(b.surface.isascii()),
             int(sentence.rstrip().endswith("?")), int(bool(re.search(r"\d", between)))],
        between=[f"{f}/{t}" for f, t in btoks],
        after_a=[f"{f}/{t}" for f, t in after_a], before_b=[f"{f}/{t}" for f, t in before_b], after_b=[f"{f}/{t}" for f, t in after_b])


class Vocab:
    """학습 쌍에서 자주 나온 형태소(어간/품사)에 번호를 붙인다. 이진 특징 두 묶음(사이, 앞뒤)으로 쓴다."""

    def __init__(self, raws, top_between=250, top_around=80, min_count=5):
        c_b, c_a = Counter(), Counter()
        for r in raws:
            c_b.update(set(r["between"]))
            c_a.update({f"a:{x}" for x in r["after_a"]} | {f"p:{x}" for x in r["before_b"]} | {f"z:{x}" for x in r["after_b"]})
        self.between = [k for k, v in c_b.most_common(top_between) if v >= min_count]
        self.around = [k for k, v in c_a.most_common(top_around * 3) if v >= min_count]
        self.index_b = {k: i for i, k in enumerate(self.between)}
        self.index_a = {k: i for i, k in enumerate(self.around)}

    @property
    def names(self):
        return NUM_NAMES + [f"btw:{k}" for k in self.between] + [f"ctx:{k}" for k in self.around]

    def vector(self, r):
        v = list(r["num"]) + [0] * (len(self.between) + len(self.around))
        base = len(r["num"])
        for k in set(r["between"]):
            if k in self.index_b:
                v[base + self.index_b[k]] = 1
        off = base + len(self.between)
        for k in {f"a:{x}" for x in r["after_a"]} | {f"p:{x}" for x in r["before_b"]} | {f"z:{x}" for x in r["after_b"]}:
            if k in self.index_a:
                v[off + self.index_a[k]] = 1
        return v


NUM_NAMES = ["between_chars", "between_words", "mention_gap", "sentence_chars", "sentence_pos", "mentions_in_sentence", "a_len", "b_len", "btw_verbs", "btw_nouns",
             "btw_particles", "comma", "paren", "list_or_heading", "a_bold", "b_bold", "a_hangul", "b_hangul", "a_ascii", "b_ascii", "question", "btw_digit"]
