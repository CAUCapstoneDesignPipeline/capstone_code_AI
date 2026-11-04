"""노트 색인에서 키워드(노드)와 개념을 만든다 (결정적). 정의는 docs/knowledge-graph.md, 스키마는 app/schemas/knowledge.py.

키워드는 같은 뜻의 개념 묶음이다. 여기서는 노트 사이 연결과 같은 표기·별칭 기준으로 묶는다 (한 노트 안의 확실한 별칭,
느슨한 표기 키). 이 기준은 후보일 뿐이다: 표기가 같아도 뜻이 다른 것 ("DP" = Dynamic Programming / Decision Problem) 은
구분하지 못한다. 같은 뜻인지 확정하는 일은 /merge-decisions 몫이라 이 묶음에는 들어 있지 않다.
같은 노트의 두 개념이 서로 별칭이 아니면 하나로 묶지 않는다 (한 노트가 같은 말을 서로 다른 뜻으로 쓰지는 않는다).
개념의 설명은 노트에서 그 개념을 풀어 쓴 문장 하나(Describer)이고, 소스는 그 문장의 위치다.
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations

from app.schemas.common import Span
from app.schemas.knowledge import Concept, Keyword, PageSource
from app.services.concepts.describe import Describer
from app.services.concepts.markdown import Utf16Index

from .corpus import Corpus
from .layout import Layout

MAX_ALIASES = 50
MAX_EXPLANATION = 2000
MAX_OFFSET = 100000          # Span 의 위치 한도
MAX_KEYWORDS_PER_SENTENCE = 8
MAX_KEYWORDS_PER_PARAGRAPH = 10
MAX_BODY_KEYWORDS_PER_SECTION = 10
MAX_EVIDENCE_PER_LEVEL = 4
W_HEADING = 0.5
W_PARAGRAPH = 0.3
LEVELS = ("sentence", "heading", "paragraph")      # 센 순서


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
    out, meta, _, _ = _build(corpus, note_uuid, uuid_of, min_confidence)
    return out, meta


def build_keyword_graph(corpus: Corpus, note_uuid: Callable[[str], str], uuid_of: Callable[..., str],
                        min_confidence: float = 0.0) -> tuple[list[Keyword], dict[str, dict], dict[tuple[str, str], dict]]:
    """build_keywords + 키워드 사이 연결 (증거 수준별). 연결 증거는 세 가지다 (센 순서):
      sentence   같은 문장에 두 키워드가 같이 나온다. 점수 1/(그 문장의 키워드 수 - 1), 8개 넘는 나열은 뺀다
      heading    한 키워드가 제목에 있고 다른 키워드가 그 절 본문에 나온다 (제목 키워드가 그 절의 주제). 점수 0.5. 제목에 같이 나온 두 키워드도 여기
      paragraph  같은 문단(빈 줄·제목으로 끊긴 줄 묶음)에 나온다. 점수 0.3/(문단의 키워드 수 - 1), 10개 넘는 문단은 뺀다
    같은 노트에서 이미 같은 문장으로 이어진 쌍은 제목·문단 증거를 더하지 않는다.
    edges: (keywordId, keywordId) 정렬 쌍 → {scores: {수준: 점수}, counts: {수준: 증거 수}, notes: {수준: {노트}},
    evidence: {수준: [{note, text}] 수준마다 4개까지}}."""
    out, meta, member_of, describers = _build(corpus, note_uuid, uuid_of, min_confidence)
    edges: dict[tuple[str, str], dict] = {}

    def add(x: str, y: str, level: str, w: float, nid: str, text: str):
        e = edges.setdefault((x, y), {"scores": {}, "counts": {}, "notes": {}, "evidence": {}})
        e["scores"][level] = e["scores"].get(level, 0.0) + w
        e["counts"][level] = e["counts"].get(level, 0) + 1
        e["notes"].setdefault(level, set()).add(nid)
        ev = e["evidence"].setdefault(level, [])
        if len(ev) < MAX_EVIDENCE_PER_LEVEL:
            ev.append({"note": nid, "text": text})

    by_note: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (nid, c), kid in member_of.items():
        by_note[nid].append((c, kid))
    for nid, items in sorted(by_note.items()):
        text = corpus.notes[nid].text
        d = describers[nid] if nid in describers else describers.setdefault(nid, Describer(text))
        lay = Layout(text)
        sent_kids: dict[int, set[str]] = defaultdict(set)
        para_kids: dict[int, dict[str, str]] = defaultdict(dict)
        head_kids: dict[int, set[str]] = defaultdict(set)
        body_kids: dict[int, dict[str, str]] = defaultdict(dict)
        for c, kid in sorted(items):
            for a, b in corpus.notes[nid].spans.get(c, []):
                i = d.containing(a, b)
                snippet = text[d.sents[i][0]:d.sents[i][1]].strip() if i is not None else lay.line_text(a)
                if i is not None:
                    sent_kids[i].add(kid)
                if lay.in_heading(a):
                    head_kids[lay.section(a)].add(kid)
                else:
                    if lay.paragraph(a) >= 0:
                        para_kids[lay.paragraph(a)].setdefault(kid, snippet)
                    body_kids[lay.section(a)].setdefault(kid, snippet)
        linked: set[tuple[str, str]] = set()
        for i, kids in sorted(sent_kids.items()):
            if 2 <= len(kids) <= MAX_KEYWORDS_PER_SENTENCE:
                sent = text[d.sents[i][0]:d.sents[i][1]].strip()
                for x, y in combinations(sorted(kids), 2):
                    add(x, y, "sentence", 1 / (len(kids) - 1), nid, sent)
                    linked.add((x, y))
        for sec, heads in sorted(head_kids.items()):
            head = lay.heading_text(sec)
            for x, y in combinations(sorted(heads), 2):
                add(x, y, "heading", W_HEADING, nid, head)
            body = body_kids.get(sec, {})
            if not 1 <= len(body) <= MAX_BODY_KEYWORDS_PER_SECTION:
                continue
            for h in sorted(heads):
                for k, snippet in sorted(body.items()):
                    pair = (h, k) if h < k else (k, h)
                    if k != h and pair not in linked:
                        add(*pair, "heading", W_HEADING, nid, f"{head}\n{snippet}")
        for kd in (para_kids[p] for p in sorted(para_kids)):
            if not 2 <= len(kd) <= MAX_KEYWORDS_PER_PARAGRAPH:
                continue
            for x, y in combinations(sorted(kd), 2):
                if (x, y) not in linked:
                    add(x, y, "paragraph", W_PARAGRAPH / (len(kd) - 1), nid, kd[x] if kd[x] == kd[y] else f"{kd[x]}\n{kd[y]}")
    for e in edges.values():
        e["scores"] = {k: round(v, 4) for k, v in e["scores"].items()}
    return out, meta, edges


def _build(corpus: Corpus, note_uuid: Callable[[str], str], uuid_of: Callable[..., str], min_confidence: float):
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
    out, meta, member_of = [], {}, {}
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
            if nid not in indexes:                                   # 노트마다 한 번만 (개념마다 만들면 글 길이 × 개념 수가 든다)
                indexes[nid] = Utf16Index(e.text)
            u16 = indexes[nid]
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
        kid = uuid_of("keyword", *sorted(members)[0])
        out.append(Keyword(keywordId=kid, name=name, aliases=aliases, concepts=concepts[:500]))
        for m in members:                                        # 설명이 없어 개념으로 못 넣은 언급도 이 키워드의 것이다
            member_of[m] = kid
    out.sort(key=lambda k: (-len(k.note_ids()), -len(k.concepts), k.name))
    return out, meta, member_of, describers
