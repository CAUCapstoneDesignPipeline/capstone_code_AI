"""관계 후보 쌍: 같은 문장 ∪ 같은 절(가장 안쪽 제목 아래) ∪ 제목에 나온 개념 ↔ 그 제목 절 안의 개념.

정답 관계의 쌍을 98% 담으면서 가능한 모든 쌍의 36~39% 만 남긴다 (과목 노트 노트당 260~360쌍, 논문 리뷰 65쌍). 쌍은 첫 언급이 앞선 개념이 a 다.
종류(kinds): sent = 같은 문장, section = 같은 가장 안쪽 절, head = 한쪽이 제목 줄에 나오고 다른 쪽이 그 절 안에 있다.
"""

import itertools
import re

from app.services.relations.extract import _py_offsets
from app.services.relations.sentences import sentence_spans


def _sections(text: str):
    heads, pos = [], 0
    for line in text.split("\n"):
        m = re.match(r"^(#{1,6})\s+", line)
        if m:
            heads.append([pos, len(m.group(1)), None, len(line)])
        pos += len(line) + 1
    for i, h in enumerate(heads):
        h[2] = next((g[0] for g in heads[i + 1:] if g[1] <= h[1]), len(text))
    return heads


def candidate_pairs(text: str, mentions: dict):
    """→ [(a, b, kinds)] a·b 는 개념 key, a 의 첫 언급이 b 보다 앞선다."""
    occ = _py_offsets(text, mentions)
    P = {}
    for o in occ:
        P.setdefault(o.key, []).append((o.start, o.end))
    heads, sents = _sections(text), sentence_spans(text)
    deepest = lambda p: max((i for i, h in enumerate(heads) if h[0] <= p < h[2]), default=-1, key=lambda i: heads[i][1])
    inside = lambda p: {i for i, h in enumerate(heads) if h[0] <= p < h[2]}
    in_head = lambda p: any(h[0] <= p < h[0] + h[3] for h in heads)
    sent_of = lambda p: next((i for i, (s, e) in enumerate(sents) if s <= p < e), -1)
    S = {k: {sent_of(s) for s, _ in v} for k, v in P.items()}
    INNER = {k: {deepest(s) for s, _ in v} for k, v in P.items()}
    SEC = {k: set().union(*(inside(s) for s, _ in v)) for k, v in P.items()}
    HEAD = {k: {deepest(s) for s, _ in v if in_head(s)} for k, v in P.items()}
    first = {k: min(s for s, _ in v) for k, v in P.items()}
    out = []
    for a, b in itertools.combinations(sorted(P, key=lambda k: first[k]), 2):
        kinds = set()
        if S[a] & S[b]:
            kinds.add("sent")
        if INNER[a] & INNER[b]:
            kinds.add("section")
        if (HEAD[a] & SEC[b]) or (HEAD[b] & SEC[a]):
            kinds.add("head")
        if kinds:
            out.append((a, b, frozenset(kinds)))
    return out
