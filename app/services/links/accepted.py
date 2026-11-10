"""'이해함' 으로 받아들인 새 지식을 지식 그래프의 generated 개념으로 두고, 다음 제안의 재료로 쓴다 (회의록 2026-11-04: "이해함" 은 시스템 그래프에 반영하고,
추천으로 들어온 지식도 기존 지식과 똑같이 처리한다. 문서에는 사용자가 허용할 때만 넣으므로 노트 원문은 건드리지 않는다).

흐름
  1. 새 지식 카드(두 요소 A·B + 결론 + 외부 자료)에서 '이해함' → AcceptedStore.add. 카드가 기댄 노트 구간(quote)들을 함께 저장한다.
  2. 제안 색인을 만들 때 merge_into 가 저장된 지식을 키워드·개념(origin=generated)으로 넣는다.
       - 결론이 이름 또는 별칭이 느슨하게 같은 기존 키워드가 있으면 그 키워드의 새 개념으로, 없으면 새 키워드 (기존 키워드의 새 개념 / 새 키워드)
       - 요소가 되어 다른 노트의 요소와 이어진다. 기댄 노트와는 이어지지 않는다 (자기가 나온 노트와 다시 이으면 제자리 돌기라서). 기댄 노트가 트리거일 때는 그 노트의 요소로 센다.
       - 받아들인 지식도 요소라 다시 '이해함' 하면 그 위에 쌓인다: 새 개념의 소스는 두 요소가 기댄 페이지 전부다.
  3. 기댄 노트 구간이 노트에서 사라지면(evidence_lost) 그 지식은 재료로 쓰지 않는다. 기록은 남기고 상태만 바꾼다 (구간이 다시 생기면 되살아난다).
     구간이 밀렸을 뿐(같은 문장이 다른 위치)이면 위치를 새로 맞춘다.

AI 서버는 상태가 없다. 이 저장소는 playground 용이고, 운영에서는 BE 가 그래프를 저장하고 받아들인 지식을 요청에 실어 보낸다 (merge_into 는 그대로 쓸 수 있다).
"""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from pydantic import ValidationError

from app.schemas.common import Span
from app.schemas.knowledge import Concept, ExternalSource, Keyword, PageSource

from .corpus import loose_key
from .generate import merge_external, side_sources
from .pairing import GENERATED_MAX_CHARS, MIN_CHARS

STORE_VERSION = 1
LIVE, LOST = "live", "evidence_lost"


class AcceptError(ValueError):
    """받아들일 수 없는 카드 (이유가 메시지)."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _u16(text: str, i: int) -> int:
    return len(text[:i].encode("utf-16-le")) // 2


def build_record(a: dict, b: dict, new_keyword: str, conclusion: str, external: list[dict], concept_id: str) -> dict:
    """카드 → 저장할 기록. a·b: {keyword, conceptId, note, noteId, text, start, end, quote, (origin, sources, external)}.
    소스는 두 요소가 기댄 페이지 구간들이고 서로 다른 노트가 둘 이상이어야 한다 (generated 개념의 조건)."""
    keyword, conclusion = new_keyword.strip(), conclusion.strip()
    if not keyword or len(keyword) > 100:
        raise AcceptError("새 키워드는 1~100자여야 한다")
    if not conclusion or len(conclusion) > 2000:
        raise AcceptError("결론은 1~2000자여야 한다")
    sources = side_sources(a, b)
    if len({s["noteId"] for s in sources}) < 2:
        raise AcceptError("서로 다른 두 노트 이상이 기댄 지식만 받아들인다")
    if len(sources) > 20:
        raise AcceptError("기댄 구간이 너무 많다 (최대 20)")
    now = _now()
    ext = [{"url": e["url"], "title": (e.get("title") or "")[:100] or None, "quote": e.get("quote") or None,
            "retrievedAt": e.get("retrievedAt") or now} for e in merge_external(external, a, b)]
    rec = {"conceptId": concept_id, "keyword": keyword, "conclusion": conclusion, "external": ext, "sources": sources,
           "from": [{"keyword": x["keyword"], "conceptId": x["conceptId"], "note": x["note"]} for x in (a, b)], "createdAt": now}
    try:                                                      # 개념으로 만들 수 있는 모양인지 (URL·구간 길이 등) 받아들일 때 알아낸다
        _concept(rec, [_page_source(s, s) for s in sources])
    except (ValidationError, ValueError) as e:
        raise AcceptError(f"개념 형식을 만들지 못했다: {e.errors()[0]['msg'] if isinstance(e, ValidationError) else e}")
    return rec


def _page_source(src: dict, at: dict) -> PageSource:
    """저장한 구간(src) + 지금 노트에서의 위치(at) → 페이지 소스."""
    return PageSource(noteId=src["noteId"], noteVersion=1, span=Span(startOffset=at["start"], endOffset=at["end"], quote=src["quote"]))


def _concept(rec: dict, sources: list[PageSource]) -> Concept:
    return Concept(conceptId=rec["conceptId"], origin="generated", explanation=rec["conclusion"], sources=sources,
                   external=[ExternalSource(url=e["url"], title=e.get("title"), retrievedAt=e["retrievedAt"], quote=e.get("quote"))
                             for e in rec["external"]])


def locate(rec: dict, texts: dict[str, str]) -> list[dict] | None:
    """기댄 구간이 지금 노트에 모두 있으면 구간마다 현재 위치 [{noteId, note, start, end, quote}], 하나라도 없으면 None.
    quote 가 그대로 있는 위치를 찾는다 (같은 문장이 여럿이면 저장된 위치에 가까운 것)."""
    out = []
    for s in rec["sources"]:
        text = texts.get(s["note"])
        if text is None or not s["quote"]:
            return None
        best, i = None, text.find(s["quote"])
        while i >= 0:
            start = _u16(text, i)
            if best is None or abs(start - s["start"]) < abs(best - s["start"]):
                best = start
            i = text.find(s["quote"], i + 1)
        if best is None:
            return None
        out.append({**s, "start": best, "end": best + len(s["quote"].encode("utf-16-le")) // 2})
    return out


def usable_as_element(rec: dict) -> bool:
    """요소로 쓸 수 있는 길이인가 (제안 색인의 gate 와 같다: 혼자 읽을 수 있는 설명)."""
    return MIN_CHARS <= len(rec["conclusion"]) <= GENERATED_MAX_CHARS


def status_of(rec: dict, texts: dict[str, str]) -> dict:
    live = locate(rec, texts) is not None
    return {"conceptId": rec["conceptId"], "keyword": rec["keyword"], "conclusion": rec["conclusion"], "status": LIVE if live else LOST,
            "usable": usable_as_element(rec), "from": rec["from"], "createdAt": rec["createdAt"],
            "notes": sorted({s["note"] for s in rec["sources"]}), "sources": len(rec["sources"]), "external": len(rec["external"])}


def merge_into(keywords: list[Keyword], meta: dict[str, dict], records: list[dict], texts: dict[str, str],
               uuid_of: Callable[..., str]) -> tuple[list[Keyword], dict[str, dict]]:
    """색인용 (키워드들, conceptId → meta) 에 받아들인 지식을 더해 새 목록으로 돌려준다 (들어온 것은 바꾸지 않는다).
    근거가 사라진 지식(evidence_lost)은 넣지 않는다. meta 의 note 는 화면에 보일 이름, notes 는 기댄 노트들이다."""
    by_name: dict[str, int] = {}
    out = list(keywords)
    for i, k in enumerate(out):
        for name in (k.name, *k.aliases):
            by_name.setdefault(loose_key(name), i)
    meta = dict(meta)
    for rec in sorted(records, key=lambda r: (r["createdAt"], r["conceptId"])):
        at = locate(rec, texts)
        if at is None:
            continue
        try:
            concept = _concept(rec, [_page_source(s, p) for s, p in zip(rec["sources"], at)])
        except (ValidationError, ValueError):
            continue
        key = loose_key(rec["keyword"])
        if key in by_name:                                                  # 기존 키워드(또는 앞서 받아들인 키워드)의 새 개념
            i = by_name[key]
            if any(c.conceptId == concept.conceptId for c in out[i].concepts):
                continue
            out[i] = out[i].model_copy(update={"concepts": [*out[i].concepts, concept]})
        else:                                                               # 새 키워드
            out.append(Keyword(keywordId=uuid_of("keyword", "generated", key), name=rec["keyword"], concepts=[concept]))
            by_name[key] = len(out) - 1
        notes = sorted({s["note"] for s in rec["sources"]})
        meta[concept.conceptId] = {"note": "새 지식 · " + " · ".join(notes), "confidence": None, "origin": "generated", "notes": notes,
                                   "sources": [dict(p) for p in at],
                                   "external": [dict(e) for e in rec["external"]]}
    return out, meta


class AcceptedStore:
    """받아들인 지식 기록 (JSON 파일 하나, 원자적으로 쓴다). version 은 바뀔 때마다 오르고 색인 캐시의 열쇠가 된다."""

    def __init__(self, path: Path):
        self.path, self._lock = Path(path), threading.Lock()
        self._items = self._load()
        self.version = 0

    def _load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return []
        return [r for r in data.get("items", []) if isinstance(r, dict) and r.get("conceptId")] if isinstance(data, dict) else []

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps({"version": STORE_VERSION, "items": self._items}, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self.version += 1

    def all(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._items]

    def get(self, concept_id: str) -> dict | None:
        with self._lock:
            return next((dict(r) for r in self._items if r["conceptId"] == concept_id), None)

    def add(self, rec: dict) -> dict:
        """같은 conceptId 가 있으면 내용을 바꾸고 처음 받아들인 때(createdAt)는 지킨다 (같은 카드를 다시 눌러도 하나다)."""
        with self._lock:
            old = next((r for r in self._items if r["conceptId"] == rec["conceptId"]), None)
            if old is not None:
                rec = {**rec, "createdAt": old["createdAt"]}
                self._items = [rec if r["conceptId"] == rec["conceptId"] else r for r in self._items]
            else:
                self._items.append(rec)
            self._save()
            return dict(rec)

    def remove(self, concept_id: str) -> bool:
        with self._lock:
            kept = [r for r in self._items if r["conceptId"] != concept_id]
            if len(kept) == len(self._items):
                return False
            self._items = kept
            self._save()
            return True
