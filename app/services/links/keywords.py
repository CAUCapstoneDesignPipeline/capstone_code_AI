"""노트 색인에서 키워드(노드)와 개념을 만든다 (결정적). 정의는 docs/knowledge-graph.md, 스키마는 app/schemas/knowledge.py.

키워드는 같은 뜻의 개념 묶음이다. 여기서는 노트 사이 연결과 같은 표기·별칭 기준으로 묶는다 (한 노트 안의 확실한 별칭,
느슨한 표기 키). 이 기준은 후보일 뿐이다: 표기가 같아도 뜻이 다른 것 ("DP" = Dynamic Programming / Decision Problem) 은
구분하지 못한다. 같은 뜻인지 확정하는 일은 /merge-decisions 몫이라 이 묶음에는 들어 있지 않다.
같은 노트의 두 개념이 서로 별칭이 아니면 하나로 묶지 않는다 (한 노트가 같은 말을 서로 다른 뜻으로 쓰지는 않는다).
개념의 설명은 노트에서 그 개념을 풀어 쓴 문장 하나(Describer)이고, 소스는 그 문장의 위치다.
"""

from collections import Counter, defaultdict
from collections.abc import Callable

from app.schemas.common import Span
from app.schemas.knowledge import Concept, Keyword, PageSource
from app.services.concepts.describe import Describer
from app.services.concepts.markdown import Utf16Index

from .corpus import Corpus

MAX_ALIASES = 50
MAX_EXPLANATION = 2000
MAX_OFFSET = 100000          # Span 의 위치 한도


class _Groups:
    """(노트, 개념 id) 를 노드로 하는 합집합 찾기. 같은 노트의 서로 별칭이 아닌 두 개념이 한 묶음에 들어가는 합치기는 거절한다."""

    def __init__(self):
        self.parent: dict[tuple[str, str], tuple[str, str]] = {}
        self.members: dict[tuple[str, str], set[tuple[str, str]]] = {}

    def add(self, x):
        self.parent.setdefault(x, x)
        self.members.setdefault(x, {x})

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y, same_note_ok: Callable[[tuple[str, str], tuple[str, str]], bool]) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return True
        for a in self.members[rx]:
            for b in self.members[ry]:
                if a[0] == b[0] and not same_note_ok(a, b):
                    return False
        self.parent[ry] = rx
        self.members[rx] |= self.members.pop(ry)
        return True


def build_keywords(corpus: Corpus, note_uuid: Callable[[str], str], uuid_of: Callable[..., str],
                   min_confidence: float = 0.0) -> tuple[list[Keyword], dict[str, dict]]:
    """(키워드들, conceptId → {note: 노트 이름, confidence}). 키워드는 노트 수, 개념 수 순으로 정렬한다.
    note_uuid: 노트 이름 → 노트 id (UUID). uuid_of(*parts): 안정적인 UUID (app.client.uid).
    min_confidence: 이 신뢰도 이상인 개념만 쓴다."""
    notes = corpus.notes
    groups = _Groups()
    by_id: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for nid in sorted(notes):
        e = notes[nid]
        for c in sorted(e.weights):
            if e.confidence.get(c, 1.0) < min_confidence:
                continue
            groups.add((nid, c))
            for i in sorted(e.ids[c]):
                by_id[i].append((nid, c))

    def same_note_ok(a, b) -> bool:
        ids = notes[a[0]].ids
        return b[1] in ids[a[1]]                            # 같은 노트 안에서 서로 별칭이면 한 키워드에 둘 수 있다

    for i in sorted(by_id):
        first, *rest = by_id[i]
        for other in rest:
            groups.union(first, other, same_note_ok)

    clusters: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for x in sorted(groups.parent):
        clusters[groups.find(x)].append(x)

    indexes: dict[str, Utf16Index] = {}
    describers: dict[str, Describer] = {}
    out, meta = [], {}
    for members in clusters.values():
        concepts, labels = [], Counter()
        seen_spans: set[tuple[str, int, int]] = set()
        for nid, c in sorted(members, key=lambda m: (-notes[m[0]].weights[m[1]], m)):
            e = notes[nid]
            if nid not in describers:
                describers[nid] = Describer(e.text)
            span = describers[nid].of(e.spans.get(c, []))
            if span is None:
                continue
            raw = e.text[span[0]:span[1]]
            text = raw.strip()
            if not text or (nid, *span) in seen_spans:        # 같은 문장은 한 번만 (별칭이 같은 문장을 가리킨다)
                continue
            u16 = indexes.setdefault(nid, Utf16Index(e.text))
            start = u16(span[0] + len(raw) - len(raw.lstrip()))
            end = start + len(text.encode("utf-16-le")) // 2
            if end > MAX_OFFSET:
                continue
            seen_spans.add((nid, *span))
            cid = uuid_of("concept", nid, c)
            concepts.append(Concept(
                conceptId=cid, origin="extracted", explanation=text[:MAX_EXPLANATION],
                sources=[PageSource(noteId=note_uuid(nid), noteVersion=1,
                                    span=Span(startOffset=start, endOffset=end, quote=text))]))
            meta[cid] = {"note": nid, "confidence": e.confidence.get(c)}
            labels[e.labels[c]] += 1
        if not concepts:
            continue
        name = sorted(labels, key=lambda s: (-labels[s], len(s), s))[0]
        aliases = [s for s in sorted(labels, key=lambda s: (-labels[s], s)) if s != name][:MAX_ALIASES]
        out.append(Keyword(keywordId=uuid_of("keyword", *sorted(members)[0]), name=name, aliases=aliases,
                           concepts=concepts[:500]))
    out.sort(key=lambda k: (-len(k.note_ids()), -len(k.concepts), k.name))
    return out, meta
