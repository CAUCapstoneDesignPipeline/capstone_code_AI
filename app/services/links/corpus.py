"""노트 묶음의 색인: 노트별 개념·중요도·언급 위치·관계·별칭.

개념 id 는 느슨한 표기 키다 (공백·하이픈 제거, 소문자). 별칭은 **그 노트 안에서만** 쓴다.
노트 A 의 개념과 노트 B 의 개념은 각자 자기 노트의 별칭으로 한 번 넓힌 id 집합이 겹칠 때 같다고 본다.
전역으로 이어 붙이면 "DP = Dynamic Programming" (노트 1) 과 "DP = Decision Problem" (노트 2) 처럼
다른 개념이 약어를 타고 하나로 뭉친다 (실제 노트에서 806 묶음 중 다수가 이렇게 틀렸다).
"""

import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict
import unicodedata
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, field

from app.services.concepts import ConceptExtractor, Mention
from app.services.llm import LLMError
from app.services.relations import extract_relations
from app.services.relations.extract import _py_offsets
from app.services.relations.sentences import sentence_spans
from app.services.relations.structure import _is_acronym, _usable

log = logging.getLogger(__name__)
_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")
_PAREN_OPEN = re.compile(r"^\s*\*{0,2}\s*\(\s*\*{0,2}\s*$")
_PAREN_CLOSE = re.compile(r"^\s*\*{0,2}\s*\)")
_DEFINITIONAL = re.compile(r"(?:은|는|이란|란)\s.*(?:이다|다|입니다)[.!]?\s*$")


def loose_key(key: str) -> str:
    """노트 사이에서 같은 개념으로 볼 정도로 느슨한 키: 공백·하이픈·밑줄·마침표 제거, 소문자."""
    return re.sub(r"[\s\-_./·]+", "", unicodedata.normalize("NFC", key).lower())


def _translation_pair(a: str, b: str) -> bool:
    """한쪽은 한글만, 다른 쪽은 영어만: '교착 상태 (Deadlock)' 처럼 같은 개념을 두 언어로 쓴 것."""
    ha, la, hb, lb = bool(_HANGUL.search(a)), bool(_LATIN.search(a)), bool(_HANGUL.search(b)), bool(_LATIN.search(b))
    return (ha and not la and lb and not hb) or (hb and not lb and la and not ha)


def note_aliases(text: str, occs) -> set[tuple[str, str]]:
    """노트 안의 확실한 별칭 쌍 (원래 key 기준).
    - 'X (Y)' 로 바로 붙어 있고 한쪽이 한글, 다른 쪽이 영어 (교착 상태 (Deadlock))
      'X: Y' 는 쓰지 않는다. 콜론 뒤는 번역이 아니라 설명인 경우가 많다 ("회복: Deadlock 을 …")
    - 약어와 풀어 쓴 말이 둘 다 이 노트의 개념 (WAL ↔ Write Ahead Logging)"""
    pairs = set()
    usable = [o for o in occs if _usable(o)]
    for a, b in zip(usable, usable[1:]):
        between = text[a.end:b.start]
        bracketed = _PAREN_OPEN.match(between) and _PAREN_CLOSE.match(text[b.end:b.end + 4])
        if bracketed and _translation_pair(a.surface, b.surface):
            pairs.add(tuple(sorted((a.key, b.key))))
    by_key = {o.key: o.surface for o in usable}
    keys = sorted(by_key)
    for x in keys:
        for y in keys:
            if x != y and _is_acronym(by_key[x], by_key[y]):
                pairs.add(tuple(sorted((x, y))))
    return pairs


@dataclass
class NoteEntry:
    note_id: str
    text: str
    weights: dict[str, float]                       # 개념 id → 이 노트에서의 중요도 (0~1)
    ids: dict[str, frozenset[str]]                  # 개념 id → 이 노트의 별칭으로 넓힌 id 집합 (자기 포함)
    spans: dict[str, list[tuple[int, int]]]         # 개념 id → 언급 위치 (Python 문자)
    labels: dict[str, str]                          # 개념 id → 이 노트에서 가장 많이 쓴 표기
    relations: list[tuple[str, str, str]]           # (개념 id, code, 개념 id)
    confidence: dict[str, float] = field(default_factory=dict)   # 개념 id → API 응답과 같은 신뢰도 (0~1)
    # relations[i] 의 근거: (UTF-16 start, end, quote) 들. extract 응답의 evidence 와 같은 구간이다
    relation_evidence: list[tuple[tuple[int, int, str], ...]] = field(default_factory=list)


@dataclass
class Corpus:
    notes: dict[str, NoteEntry]
    df: Counter                                     # id → 그 id 를 (별칭 포함) 가진 노트 수
    version: str = ""

    def sentence_with(self, note_id: str, cid: str) -> tuple[int, int] | None:
        """그 개념이 나오는 문장. 정의문(… 는 … 다)을 먼저, 그다음 앞쪽 문장."""
        note = self.notes[note_id]
        spans = note.spans.get(cid, [])
        sents = [(s, e) for s, e in sentence_spans(note.text) if any(s <= a and b <= e for a, b in spans)]
        if not sents:
            return spans[0] if spans else None
        return min(sents, key=lambda se: (not _DEFINITIONAL.search(note.text[se[0]:se[1]]), se[0]))


    def with_note(self, entry: "NoteEntry") -> "Corpus":
        """노트 하나를 더하거나 바꾼 새 색인 (원래 색인은 그대로). 붙여넣은 글·편집한 노트를 기존 노트와 잇는 데 쓴다."""
        df = Counter(self.df)
        old = self.notes.get(entry.note_id)
        if old is not None:
            df.subtract(_all_ids(old))
        df.update(_all_ids(entry))
        return Corpus({**self.notes, entry.note_id: entry}, +df, self.version)


def _all_ids(entry: "NoteEntry") -> set[str]:
    return set().union(*entry.ids.values()) if entry.ids else set()


def build_entry(note_id: str, text: str, extractor: ConceptExtractor, select, policy_types, limits,
                typer=None) -> NoteEntry:
    """노트 하나의 색인 항목. select: (mentions) → {key: [Mention]} (API 와 같은 profile 선택 규칙)."""
    text = unicodedata.normalize("NFC", text)
    selected: dict[str, list[Mention]] = select(extractor.extract(note_id, text))
    occs = _py_offsets(text, selected)
    # typer 가 있으면 규칙·구조가 못 정한 문장 속 개념 쌍을 LLM 이 유형을 정한다 (extract-hybrid-2 와 같다).
    # 모델이 이 노트를 거절하면(보안 실습 노트 등) 이 노트만 규칙·구조 관계로 둔다. 다른 모델로 바꾸지 않는다.
    # 시간 초과·제공자 오류는 그대로 올린다 (재시도는 호출한 쪽이 한다)
    try:
        evs, rels = extract_relations(text, selected, policy_types, typer, limits, structure=True)
    except LLMError as e:
        if e.kind != "refusal":
            raise
        log.warning("관계 LLM 이 노트 %s 를 거절해 규칙 관계만 쓴다 (%s)", note_id, e)
        evs, rels = extract_relations(text, selected, policy_types, None, limits, structure=True)

    weights, conf, spans, surfaces = {}, {}, defaultdict(list), defaultdict(Counter)
    top = max((ms[0].score for ms in selected.values()), default=1.0) or 1.0
    for key, ms in selected.items():
        c = loose_key(key)
        weights[c] = max(weights.get(c, 0.0), round(ms[0].score / top, 4))
        conf[c] = max(conf.get(c, 0.0), round(min(ms[0].score, 2.0) / 2.0, 4))   # api/v1/extract.to_concepts 와 같다
        for m in ms:
            surfaces[c][m.surface] += 1
    for o in occs:
        spans[loose_key(o.key)].append((o.start, o.end))
    ids = {c: {c} for c in weights}
    for a, b in note_aliases(text, occs):             # 이 노트 안에서만 한 단계 넓힌다
        la, lb = loose_key(a), loose_key(b)
        if la in ids and lb in ids:
            ids[la].add(lb)
            ids[lb].add(la)
    labels = {c: sorted(cnt, key=lambda s: (-cnt[s], s))[0] for c, cnt in surfaces.items()}
    kept = [r for r in rels if loose_key(r.source) != loose_key(r.target)]
    relations = [(loose_key(r.source), r.code, loose_key(r.target)) for r in kept]
    relation_evidence = [tuple((evs[n].start16, evs[n].end16, evs[n].quote) for n in r.evidence) for r in kept]
    return NoteEntry(note_id, text, weights, {c: frozenset(v) for c, v in ids.items()}, dict(spans), labels, relations,
                     conf, relation_evidence)


class EntryMemo:
    """노트 항목의 프로세스 안 LRU 캐시 (스레드 안전). 결과를 바꾸지 않는 성능용이다."""

    def __init__(self, size: int = 2000):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        return None

    def put(self, key, entry):
        with self._lock:
            self._d[key] = entry
            self._d.move_to_end(key)
            while len(self._d) > self.size:
                self._d.popitem(last=False)


def build_corpus(texts: dict[str, str], extractor: ConceptExtractor, select, policy_types, limits,
                 version: str = "", lock=None, typer=None, memo: "EntryMemo | None" = None) -> Corpus:
    """lock: 형태소 분석기를 같이 쓰는 다른 요청과 나눌 잠금. 노트마다 잡고 놓아서 그 사이 요청이 끼어들 수 있다.
    memo: 노트 항목 재사용 (원문·policy·방법 버전·관계 모델이 같으면 결과도 같으므로 다시 만들지 않는다)."""
    notes, df = {}, Counter()
    sig = json.dumps([version, sorted(policy_types), typer.model if typer else None], ensure_ascii=False)
    for note_id in sorted(texts):
        key = (note_id, hashlib.sha256(texts[note_id].encode()).hexdigest(), sig)
        entry = memo.get(key) if memo is not None else None
        if entry is None:
            with lock or nullcontext():
                entry = build_entry(note_id, texts[note_id], extractor, select, policy_types, limits, typer)
            if memo is not None:
                memo.put(key, entry)
        notes[note_id] = entry
        df.update(_all_ids(entry))
    return Corpus(notes, df, version)
