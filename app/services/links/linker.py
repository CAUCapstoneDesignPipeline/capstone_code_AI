"""노트 쌍 점수와 연결 설명."""

import math
from collections import defaultdict
from dataclasses import dataclass

from .corpus import Corpus


@dataclass(frozen=True)
class Bridge:
    concept_a: str                              # 노트 A 의 개념 id
    concept_b: str                              # 노트 B 의 개념 id (별칭으로 이어졌으면 다를 수 있다)
    label_a: str
    label_b: str
    contribution: float                         # 점수 기여도
    evidence_a: tuple[int, int] | None          # 노트 A 에서 이 개념이 나오는 문장 (Python 문자)
    evidence_b: tuple[int, int] | None
    paths: tuple[tuple[str, str, str, str, str], ...]   # (A, code1, B, code2, C): A–B 는 노트 A, B–C 는 노트 B 의 관계


@dataclass(frozen=True)
class Link:
    note_a: str
    note_b: str
    score: float
    bridges: tuple[Bridge, ...]


MIN_HUB_NOTES = 5      # 노트가 적을 때 2~3개 노트에 나온 개념까지 '흔하다' 고 버리지 않게


def related_notes(corpus: Corpus, note_id: str, k: int = 10, hub_ratio: float = 0.15,
                  max_bridges: int = 5, min_confidence: float = 0.0) -> list[Link]:
    """note_id 와 가장 많이, 드문 개념으로 이어진 노트 k 개. 같은 입력이면 같은 순서 (동점은 노트 id 순).
    점수 = Σ (A 에서의 중요도 × B 에서의 중요도 / log(1 + 그 개념이 나온 노트 수)), 허브 개념은 뺀다.
    min_confidence: 양쪽 노트 모두에서 신뢰도가 이 값 이상인 개념만 쓴다."""
    valid = lambda n, c: n.confidence.get(c, 1.0) >= min_confidence
    a = corpus.notes[note_id]
    hub = max(MIN_HUB_NOTES, int(len(corpus.notes) * hub_ratio))
    scored = []
    for other_id in sorted(corpus.notes):
        if other_id == note_id:
            continue
        b = corpus.notes[other_id]
        inverse = defaultdict(list)                 # id → 노트 B 의 개념들
        for d, ids in b.ids.items():
            if not valid(b, d):
                continue
            for i in ids:
                inverse[i].append(d)
        contribs = []
        for c in sorted(a.weights):
            if not valid(a, c):
                continue
            best = None
            for i in sorted(a.ids[c]):
                for d in inverse.get(i, []):
                    shared = a.ids[c] & b.ids[d]
                    df = min(corpus.df[x] for x in shared)
                    if df > hub:
                        continue
                    w = round(a.weights[c] * b.weights[d] / math.log(1 + df), 6)
                    if best is None or (w, d) > (best[0], best[1]):
                        best = (w, d)
            if best:
                contribs.append((best[0], c, best[1]))
        # 같은 개념을 두 번 세지 않는다: '교착 상태' 와 별칭 'Deadlock' 이 B 의 같은 개념으로 이어지면 큰 쪽 하나만
        contribs.sort(key=lambda x: (-x[0], x[1]))
        used_a, used_b, kept = set(), set(), []
        for w, c, d in contribs:
            if c in used_a or d in used_b:
                continue
            kept.append((w, c, d))
            used_a |= a.ids[c]
            used_b |= b.ids[d]
        contribs = kept
        if not contribs:
            continue
        scored.append((round(sum(x[0] for x in contribs), 4), other_id, contribs))
    scored.sort(key=lambda t: (-t[0], t[1]))

    return [Link(note_id, other_id, score, tuple(_bridge(corpus, note_id, other_id, c, d, w)
                                                 for w, c, d in contribs[:max_bridges]))
            for score, other_id, contribs in scored[:k]]


def _bridge(corpus: Corpus, a_id: str, b_id: str, c: str, d: str, contribution: float) -> Bridge:
    a, b = corpus.notes[a_id], corpus.notes[b_id]
    # A–B–C: 노트 A 에서 공유 개념과 이어진 개념, 노트 B 에서 공유 개념과 이어진 개념
    left = [(y if x == c else x, code) for x, code, y in a.relations if c in (x, y)]
    right = [(y if x == d else x, code) for x, code, y in b.relations if d in (x, y)]
    paths = sorted({(a.labels.get(l, l), c1, a.labels[c], c2, b.labels.get(r, r))
                    for l, c1 in left for r, c2 in right if l != r})[:5]
    return Bridge(c, d, a.labels[c], b.labels[d], contribution, corpus.sentence_with(a_id, c),
                  corpus.sentence_with(b_id, d), tuple(paths))


@dataclass(frozen=True)
class PathCandidate:
    """노트 사이 A —(노트 1 관계)→ B —(노트 2 관계)→ C. B 는 두 노트가 공유하는 개념이다."""
    note1: str
    rel1: int                   # corpus.notes[note1].relations 의 번호
    note2: str
    rel2: int
    proposed: str | None        # 합성 규칙 결과. 규칙이 없으면 None (가설)


def directed_paths(corpus: Corpus, a_id: str, b_id: str, c: str, d: str, rules: dict[tuple[str, str], str],
                   limit: int = 5) -> list[PathCandidate]:
    """노트 a 의 개념 c 와 노트 b 의 개념 d 가 같은 매개 개념일 때, 방향이 맞는 두 단계 경로.
    양쪽 방향 모두 본다: (a 에서 X→c, b 에서 d→Y) 와 (b 에서 X→d, a 에서 c→Y). A 와 C 가 같은 개념이면 뺀다.
    합성 규칙이 있는 경로를 먼저, 그다음 노트·관계 순서."""
    out = []
    for n1, b1, n2, b2 in ((a_id, c, b_id, d), (b_id, d, a_id, c)):
        e1, e2 = corpus.notes[n1], corpus.notes[n2]
        ins = [(i, x, code) for i, (x, code, y) in enumerate(e1.relations) if y == b1]
        outs = [(j, y, code) for j, (x, code, y) in enumerate(e2.relations) if x == b2]
        for i, x, c1 in ins:
            for j, y, c2 in outs:
                if e1.ids.get(x, frozenset({x})) & e2.ids.get(y, frozenset({y})):
                    continue
                out.append(PathCandidate(n1, i, n2, j, rules.get((c1, c2))))
    out.sort(key=lambda p: (p.proposed is None, p.note1, p.rel1, p.rel2))
    return out[:limit]
