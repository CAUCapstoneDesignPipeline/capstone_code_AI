"""개발용 playground (AI_DEV_UI=1 또는 `ai serve --dev` 일 때만 붙는다).

  GET /dev/              playground 화면
  GET /dev/config        화면 초기값 (profile 목록, 팀 관계 유형 초안, 로컬 접속이면 서비스 토큰)
  GET /dev/notes         노트 폴더(AI_NOTES_DIR, 기본 data/private/notes)의 노트 id 목록
  GET /dev/notes/{id}    노트 원문 (NFC)
  GET /dev/index         노트 전체 색인을 미리 만든다 (노트 사이 연결용)
  POST /dev/extract      {noteId?, text?} 노트 하나의 키워드와 키워드마다 설명 (결정적)
  POST /dev/keyword-graph {noteId?, text?, scope, limit, minConfidence} 키워드 노드와 같은 문장에 같이 나온 연결 (결정적)
  POST /dev/keywords     {noteId?, text?, q, minPages, limit, minConfidence} 키워드(노드)와 그 안의 개념 (결정적)
  POST /dev/links        {noteId?, text?, k, minConfidence} 와 이어진 다른 노트 (결정적). text 를 주면 그 글로 잇는다
  GET /dev/links/{id}    저장된 노트로 같은 일
  GET/POST/DELETE /dev/llm   관계 판정 LLM (extract-hybrid-1) 을 실행 중에 고르거나 끈다.
                             API 키는 서버 메모리에만 두고, 저장·로그·응답에 넣지 않는다. 로컬 접속만 받는다

배포용이 아니다. 로컬 노트를 읽어 보여 주므로 127.0.0.1 에서만 띄운다.
"""

import json
import os
import threading
import time
import unicodedata
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from app.core.config import API_PREFIX, EMBEDDING_PROFILES, EXTRACT_PROFILE, EXTRACT_PROFILES
from app.services.llm import DEFAULT_MODELS, PROVIDERS, LLMConfig, make_llm
from app.services.relations import RelationTyper

router = APIRouter(prefix="/dev", include_in_schema=False)

NOTES_DIR = Path("data/private/notes")      # AI_NOTES_DIR 로 바꾼다 (run_playground.py --notes)
RELATION_TYPES = Path("experiments/extract_v0/relation_types.json")
PAGE = Path(__file__).with_name("playground.html")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _notes_dir() -> Path:
    return Path(os.environ["AI_NOTES_DIR"]) if os.environ.get("AI_NOTES_DIR") else NOTES_DIR


def _note_paths() -> dict[str, Path]:
    return {_nfc(p.stem): p for p in sorted(_notes_dir().glob("*")) if p.suffix in (".md", ".txt")}


@router.get("/", response_class=HTMLResponse)
async def page():
    return PAGE.read_text(encoding="utf-8")


@router.get("/config")
async def config(request: Request):
    types = []
    if RELATION_TYPES.exists():
        types = [{"code": t["name"], "label": t["name"], "description": t["description"]}
                 for t in json.loads(RELATION_TYPES.read_text(encoding="utf-8"))["types"]]
    loopback = _loopback(request)
    st = request.app.state
    return {
        "llm": _llm_status(st),
        "llmProviders": [{"id": p, "label": LLM_LABELS[p], "defaultModel": DEFAULT_MODELS.get(p, ""),
                          "keyEnv": KEY_ENV[p]} for p in PROVIDERS],
        "apiPrefix": API_PREFIX,
        # 로컬 접속일 때만 토큰을 채워 준다 (직접 붙여넣지 않아도 되게)
        "token": st.settings.service_token if loopback else None,
        "extractProfiles": [{"id": p.id, "relations": p.relations, "llm": p.uses_llm, "default": p.id == EXTRACT_PROFILE}
                            for p in EXTRACT_PROFILES.values()],
        "embeddingProfiles": [{"id": p.id, "dims": p.dims, "loaded": p.id in st.embedders}
                              for p in EMBEDDING_PROFILES.values()],
        "relationTypes": types,
    }


@router.get("/notes")
async def notes():
    return sorted(_note_paths())


@router.get("/notes/{note_id:path}")
async def note(note_id: str):
    path = _note_paths().get(_nfc(note_id))
    if path is None:
        raise HTTPException(404)
    return {"id": _nfc(note_id), "body": _nfc(path.read_text(encoding="utf-8"))}


# ---- 관계 판정 LLM 고르기 ----

LLM_LABELS = {"anthropic": "Anthropic · Claude", "openai": "OpenAI", "google": "Google · Gemini"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}


def _loopback(request: Request) -> bool:
    return bool(request.client and request.client.host in ("127.0.0.1", "::1"))


def _llm_status(state) -> dict | None:
    t = state.typer
    return {"provider": t.provider, "model": t.model} if t is not None else None


class LlmChoice(BaseModel):
    provider: str
    model: str = ""
    apiKey: str = ""        # 비우면 서버 환경 변수 (ANTHROPIC_API_KEY 등) 를 쓴다


@router.get("/llm")
async def get_llm(request: Request):
    return {"llm": _llm_status(request.app.state)}


@router.post("/llm")
async def set_llm(choice: LlmChoice, request: Request):
    if not _loopback(request):
        raise HTTPException(403, "로컬에서만 바꿀 수 있다")
    provider = choice.provider.strip().lower()
    if provider not in PROVIDERS:
        raise HTTPException(400, f"제공자는 {', '.join(PROVIDERS)} 중 하나")
    model = choice.model.strip() or DEFAULT_MODELS.get(provider, "")
    if not model:
        raise HTTPException(400, f"{LLM_LABELS[provider]} 은 모델 id 를 직접 적어야 한다")
    cfg = LLMConfig(provider, model, api_key=choice.apiKey.strip() or None)
    try:   # client 를 만들 때 키가 없으면 SDK 가 바로 오류를 낸다 (openai, google)
        llm = await run_in_threadpool(make_llm, cfg)
    except Exception as e:
        raise HTTPException(400, f"{LLM_LABELS[provider]} client 를 만들지 못했다: {type(e).__name__}. "
                                 f"API 키 또는 {KEY_ENV[provider]} 를 확인한다")
    st = request.app.state
    st.typer = RelationTyper(llm, st.settings.relation_cache)
    return {"llm": _llm_status(st)}


@router.delete("/llm")
async def clear_llm(request: Request):
    if not _loopback(request):
        raise HTTPException(403, "로컬에서만 바꿀 수 있다")
    request.app.state.typer = None
    return {"llm": None}


# ---- 노트 사이 연결 ----

def _policy_types() -> list[tuple[str, str, str]]:
    """팀 관계 유형 초안 + analogous_to (playground 기본값과 같다)."""
    types = []
    if RELATION_TYPES.exists():
        types = [(t["name"], t["name"], t["description"])
                 for t in json.loads(RELATION_TYPES.read_text(encoding="utf-8"))["types"]]
    if not any(c == "analogous_to" for c, _, _ in types):
        types.append(("analogous_to", "유사", "두 개념이 비슷한 원리로 동작한다"))
    return types


_BUILD_LOCK = threading.Lock()


def _links_corpus(state):
    """노트 폴더·추출 설정이 바뀌지 않았으면 만들어 둔 색인을 다시 쓴다."""
    with _BUILD_LOCK:            # 화면의 미리 만들기와 첫 연결 요청이 겹쳐도 한 번만 만든다
        return _links_corpus_locked(state)


def _links_corpus_locked(state):
    from app.api.v1.extract import select_concepts
    from app.services.links import build_corpus

    paths = _note_paths()
    typer = getattr(state, "typer", None)
    signature = (state.extractor.method_version, typer.model if typer else None,
                 tuple(sorted((k, p.stat().st_mtime_ns, p.stat().st_size) for k, p in paths.items())))
    cached = getattr(state, "links_cache", None)
    if cached and cached[0] == signature:
        return cached[1], cached[2]
    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    t0 = time.monotonic()
    texts = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
    corpus = build_corpus(texts, state.extractor, lambda ms: select_concepts(ms, profile), _policy_types(),
                          profile.limits, version=state.extractor.method_version, lock=state.kiwi_lock, typer=typer)
    built_ms = int((time.monotonic() - t0) * 1000)
    state.links_cache = (signature, corpus, built_ms)
    return corpus, built_ms


FREE_ID = "(직접 입력)"


class LinksQuery(BaseModel):
    noteId: str | None = None   # 노트 폴더의 노트. text 가 없으면 저장된 원문을 쓴다
    text: str | None = None     # 붙여넣은 글이나 편집한 원문. 이 글로 색인 항목을 새로 만들어 잇는다
    k: int = 10
    minConfidence: float = 0.0  # 이 신뢰도 이상인 개념만 노트 사이 연결에 쓴다 (양쪽 노트 모두)


@router.get("/index")
async def index(request: Request):
    """노트 전체 색인을 미리 만든다 (화면을 열 때 부른다). 이미 있으면 바로 돌려준다."""
    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    corpus, built_ms = await run_in_threadpool(_links_corpus, state)
    return {"notes": len(corpus.notes), "buildMs": built_ms}


@router.get("/links/{note_id:path}")
async def links_get(note_id: str, request: Request, k: int = 10):
    return await links(LinksQuery(noteId=note_id, k=k), request)


@router.post("/links")
async def links(q: LinksQuery, request: Request):
    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    corpus, built_ms = await run_in_threadpool(_links_corpus, state)
    note_id = _nfc(q.noteId) if q.noteId else None
    if q.text is None:
        if note_id not in corpus.notes:
            raise HTTPException(404, "노트를 찾지 못했다")
    else:
        if len(q.text) > 50000:
            raise HTTPException(413, "글은 50,000자까지다")
        if note_id not in corpus.notes:
            note_id = FREE_ID
        text = _nfc(q.text)
        if note_id not in corpus.notes or corpus.notes[note_id].text != text:
            corpus = await run_in_threadpool(_with_text, state, corpus, note_id, text)

    from app.services.links import related_notes
    result = await run_in_threadpool(related_notes, corpus, note_id, max(1, min(q.k, 30)), 0.15, 5,
                                     max(0.0, min(q.minConfidence, 1.0)))

    def quote(nid, span):
        return corpus.notes[nid].text[span[0]:span[1]].strip() if span else None

    return {
        "note": note_id,
        "notes": len(corpus.notes),
        "buildMs": built_ms,
        "minConfidence": q.minConfidence,
        "links": [{
            "note": l.note_b, "score": l.score,
            "bridges": [{"labelA": b.label_a, "labelB": b.label_b, "contribution": round(b.contribution, 4),
                         "confidenceA": corpus.notes[l.note_a].confidence.get(b.concept_a),
                         "confidenceB": corpus.notes[l.note_b].confidence.get(b.concept_b),
                         "quoteA": quote(l.note_a, b.evidence_a), "quoteB": quote(l.note_b, b.evidence_b)}
                        for b in l.bridges],
        } for l in result],
    }


# ---- 키워드와 설명 추출 (노트 하나) ----

class ExtractQuery(BaseModel):
    noteId: str | None = None
    text: str | None = None


def _extract_note(state, note_id: str, text: str) -> list[dict]:
    from app.api.v1.extract import concept_label, select_concepts
    from app.services.concepts.describe import Describer
    from app.services.concepts.markdown import Utf16Index
    from app.services.relations.extract import _py_offsets

    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    with state.kiwi_lock:
        selected = select_concepts(state.extractor.extract(note_id, text), profile)
    py: dict[str, list[tuple[int, int]]] = {}
    for o in _py_offsets(text, selected):
        py.setdefault(o.key, []).append((o.start, o.end))
    describer, u16 = Describer(text), Utf16Index(text)
    out = []
    for key, ms in selected.items():
        span = describer.of(py.get(key, []))
        out.append({
            "ref": key, "label": concept_label(ms), "confidence": round(min(ms[0].score, 2.0) / 2.0, 4),   # /extract 와 같다
            "mentions": [{"startOffset": m.span.start, "endOffset": m.span.end} for m in ms],
            "description": None if span is None else {"text": text[span[0]:span[1]],
                                                      "startOffset": u16(span[0]), "endOffset": u16(span[1])},
        })
    return out


@router.post("/extract")
async def extract_keywords(q: ExtractQuery, request: Request):
    """노트 하나에서 키워드(개념 후보)와 키워드마다 설명(원문 문장 하나)을 뽑는다. 키워드·신뢰도는 /extract 와 같은 추출기·profile 이고
    (LLM 필터 없음), 설명은 app/services/concepts/describe.py 가 고른다. LLM 호출 없음."""
    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    if q.text is not None:
        if not q.text.strip() or len(q.text) > 50000:
            raise HTTPException(400, "글이 비었거나 50,000자를 넘는다")
        note_id, text = FREE_ID, _nfc(q.text)
    else:
        path = _note_paths().get(_nfc(q.noteId or ""))
        if path is None:
            raise HTTPException(404, "노트를 찾지 못했다")
        note_id, text = _nfc(q.noteId), _nfc(path.read_text(encoding="utf-8"))
    t0 = time.monotonic()
    keywords = await run_in_threadpool(_extract_note, state, note_id, text)
    return {"note": note_id, "keywords": keywords, "ms": int((time.monotonic() - t0) * 1000)}


# ---- 키워드 (노드) 와 개념 ----

class KeywordsQuery(BaseModel):
    noteId: str | None = None   # 이 노트에 개념을 넣은 키워드만. 없으면 전체
    text: str | None = None     # 붙여넣은 글이나 편집한 원문. 이 글을 노트로 더해서 만든다
    q: str = ""                 # 키워드 이름·별칭에 들어 있는 글자
    minPages: int = 1           # 이 수 이상의 페이지(노트)에 걸친 키워드만
    limit: int = 60
    minConfidence: float = 0.0


def _keyword_set(state, corpus, min_confidence: float):
    """(키워드들, conceptId → {note, confidence}, 키워드 사이 연결). 같은 색인·기준이면 만들어 둔 것을 다시 쓴다."""
    from app.client import uid
    from app.services.links import build_keyword_graph

    cached = getattr(state, "keywords_cache", None)
    if cached and cached[0] is corpus and cached[1] == min_confidence:
        return cached[2]
    built = build_keyword_graph(corpus, lambda n: uid("note", n), uid, min_confidence)
    state.keywords_cache = (corpus, min_confidence, built)
    return built


@router.post("/keywords")
async def keywords(q: KeywordsQuery, request: Request):
    """노드 = 키워드, 키워드 안에 개념(설명 + 소스 페이지) 여러 개 (docs/knowledge-graph.md). 응답의 키워드는
    app/schemas/knowledge.py 의 Keyword 로 검증된 것이고, 화면용으로 노트 이름·신뢰도를 덧붙였다. LLM 호출 없음."""
    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    t0 = time.monotonic()
    corpus, _ = await run_in_threadpool(_links_corpus, state)
    note_id = _nfc(q.noteId) if q.noteId else None
    if q.text is not None:
        if not q.text.strip() or len(q.text) > 50000:
            raise HTTPException(400, "글이 비었거나 50,000자를 넘는다")
        note_id = note_id if note_id in corpus.notes else FREE_ID
        text = _nfc(q.text)
        if note_id not in corpus.notes or corpus.notes[note_id].text != text:
            corpus = await run_in_threadpool(_with_text, state, corpus, note_id, text)
    elif note_id is not None and note_id not in corpus.notes:
        raise HTTPException(404, "노트를 찾지 못했다")
    mc = max(0.0, min(q.minConfidence, 1.0))
    all_keywords, meta, _ = await run_in_threadpool(_keyword_set, state, corpus, mc)
    needle = _nfc(q.q).strip().lower()

    def wanted(k) -> bool:
        if len(k.note_ids()) < max(1, q.minPages):
            return False
        if needle and not any(needle in s.lower() for s in (k.name, *k.aliases)):
            return False
        return note_id is None or any(meta[c.conceptId]["note"] == note_id for c in k.concepts)

    picked = [k for k in all_keywords if wanted(k)]
    return {
        "note": note_id, "notes": len(corpus.notes), "total": len(picked), "all": len(all_keywords),
        "ms": int((time.monotonic() - t0) * 1000),
        "keywords": [{
            **k.model_dump(exclude={"concepts"}), "pages": len(k.note_ids()),
            "concepts": [{"conceptId": c.conceptId, "origin": c.origin, "note": meta[c.conceptId]["note"],
                          "confidence": meta[c.conceptId]["confidence"], "explanation": c.explanation,
                          "startOffset": c.sources[0].span.startOffset, "endOffset": c.sources[0].span.endOffset}
                         for c in k.concepts],
        } for k in picked[:max(1, min(q.limit, 200))]],
    }


class KeywordGraphQuery(BaseModel):
    noteId: str | None = None
    text: str | None = None
    scope: str = "note"         # note: 이 노트의 키워드와 그 이웃 · all: 전체에서 연결이 센 키워드
    limit: int = 0              # 가져올 키워드 수 (0 = note 14 / all 200)
    minConfidence: float = 0.0


@router.post("/keyword-graph")
async def keyword_graph(q: KeywordGraphQuery, request: Request):
    """키워드를 노드로, 같은 문장에 같이 나온 키워드끼리 선으로 (app/services/links/keywords.py build_keyword_graph).
    선마다 그 문장(근거)을, 노드마다 개념(설명 + 페이지)을 담는다. LLM 호출 없음."""
    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    if q.scope not in ("note", "all"):
        raise HTTPException(400, "scope 는 note 또는 all")
    t0 = time.monotonic()
    corpus, _ = await run_in_threadpool(_links_corpus, state)
    note_id = _nfc(q.noteId) if q.noteId else None
    if q.text is not None:
        if not q.text.strip() or len(q.text) > 50000:
            raise HTTPException(400, "글이 비었거나 50,000자를 넘는다")
        note_id = note_id if note_id in corpus.notes else FREE_ID
        text = _nfc(q.text)
        if note_id not in corpus.notes or corpus.notes[note_id].text != text:
            corpus = await run_in_threadpool(_with_text, state, corpus, note_id, text)
    elif note_id is not None and note_id not in corpus.notes:
        raise HTTPException(404, "노트를 찾지 못했다")
    if q.scope == "note" and note_id is None:
        raise HTTPException(400, "note 범위에는 noteId 또는 text 가 필요하다")
    mc = max(0.0, min(q.minConfidence, 1.0))
    kws, meta, edges = await run_in_threadpool(_keyword_set, state, corpus, mc)
    by_id = {k.keywordId: k for k in kws}
    mine = {k.keywordId for k in kws if note_id and any(meta[c.conceptId]["note"] == note_id for c in k.concepts)}
    strength: dict[str, float] = {}
    adj: dict[str, dict[str, float]] = {}
    for (a, b), e in edges.items():
        for x, y in ((a, b), (b, a)):
            strength[x] = strength.get(x, 0.0) + e["score"]
            adj.setdefault(x, {})[y] = e["score"]
    if q.scope == "note":
        own = sorted((k for k in mine if k in strength), key=lambda k: (-strength[k], by_id[k].name))[:q.limit or 14]
        near: dict[str, float] = {}
        for k in own:
            for y, sc in adj[k].items():
                if y not in own:
                    near[y] = near.get(y, 0.0) + sc
        chosen = own + sorted(near, key=lambda k: (-near[k], by_id[k].name))[:max(6, len(own) * 3 // 4)]
    else:
        pool = [k.keywordId for k in kws if k.keywordId in strength and len(k.note_ids()) >= 2]
        chosen = sorted(pool, key=lambda k: (-strength[k], by_id[k].name))[:q.limit or 200]
    inside = set(chosen)
    quote = lambda nid, sent: corpus.notes[nid].text[sent[0]:sent[1]].strip()
    out_edges = []
    for (a, b), e in sorted(edges.items(), key=lambda kv: -kv[1]["score"]):
        if a in inside and b in inside:
            ev = sorted(e["evidence"], key=lambda x: (x[0] == note_id, x[0]))[::-1][:3] if note_id else e["evidence"][:3]
            out_edges.append({"a": a, "b": b, "score": e["score"], "pages": e["pages"], "sentences": len(e["evidence"]),
                              "evidence": [{"note": nid, "quote": quote(nid, sent)} for nid, sent in ev]})
    degree: dict[str, int] = {}
    for e in out_edges:
        degree[e["a"]] = degree.get(e["a"], 0) + 1
        degree[e["b"]] = degree.get(e["b"], 0) + 1
    nodes = [{
        "id": k, "title": by_id[k].name, "aliases": by_id[k].aliases[:4], "pages": len(by_id[k].note_ids()),
        "inNote": k in mine, "degree": degree.get(k, 0),
        "concepts": [{"note": meta[c.conceptId]["note"], "explanation": c.explanation} for c in by_id[k].concepts[:6]],
    } for k in chosen]
    return {"note": note_id, "scope": q.scope, "nodes": nodes, "edges": out_edges, "notes": len(corpus.notes),
            "keywords": len(kws), "linkedKeywords": len(strength), "mine": len(mine),
            "mineLinked": sum(1 for k in mine if k in strength), "ms": int((time.monotonic() - t0) * 1000)}


def _with_text(state, corpus, note_id: str, text: str):
    from app.api.v1.extract import select_concepts
    from app.services.links import build_entry

    profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
    with state.kiwi_lock:
        entry = build_entry(note_id, text, state.extractor, lambda ms: select_concepts(ms, profile),
                            _policy_types(), profile.limits, getattr(state, "typer", None))
    return corpus.with_note(entry)


# ---- 새 지식 제안 (전체 흐름) ----

# 예시 합성 규칙. 정책의 정본은 BE 다 (ai-api.md §3.5). 여기 규칙은 playground 에서 흐름을 보기 위한 초안이다
EXAMPLE_RULES = [("is_a", "is_a", "is_a"), ("part_of", "part_of", "part_of"), ("causes", "causes", "causes"),
                 ("requires", "requires", "requires"), ("is_a", "part_of", "part_of"), ("is_a", "used_for", "used_for"),
                 ("is_a", "requires", "requires"), ("is_a", "causes", "causes"), ("example_of", "is_a", "example_of")]


class SuggestQuery(BaseModel):
    noteId: str | None = None
    text: str | None = None
    k: int = 5
    minConfidence: float = 0.0


def playground_policy() -> dict:
    # 색인·관계 캐시와 같은 유형 목록이어야 한다 (관계 LLM 캐시 key 에 policy 의 label·설명이 들어간다)
    types = [{"code": c, "label": label, "description": d} for c, label, d in _policy_types()]
    codes = {t["code"] for t in types}
    rules = [{"firstType": a, "secondType": b, "resultType": r} for a, b, r in EXAMPLE_RULES
             if {a, b, r} <= codes]
    return {"version": "playground-1", "relationTypes": types, "compositionRules": rules}


@router.post("/suggest")
async def suggest(q: SuggestQuery, request: Request):
    """노트 하나에서 시작해 연결된 노트와 새 지식 제안을 낸다. 내부 API 를 BE 와 같은 방식(app.client)으로 부른다."""
    import httpx

    from app.client import AiCallError, AiClient, snapshot, suggest_knowledge, uid

    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    if state.explainer is None:          # 지식 제안은 항상 LLM 이 쓴다
        raise HTTPException(503, "설명 LLM 이 없다. .env 에 ANTHROPIC_API_KEY 를 넣거나 AI_EXPLAIN_LLM_PROVIDER 를 정한다")
    paths = _note_paths()
    names = {}
    snaps = []
    for name, p in paths.items():
        nid = uid("note", name)
        names[nid] = name
        snaps.append(snapshot(nid, name, _nfc(p.read_text(encoding="utf-8"))))
    if q.text is not None:
        if not q.text.strip() or len(q.text) > 50000:
            raise HTTPException(400, "글이 비었거나 50,000자를 넘는다")
        note = snapshot(uid("note", FREE_ID), FREE_ID, _nfc(q.text))
        names[note["noteId"]] = FREE_ID
        candidates = snaps
    else:
        nid = uid("note", _nfc(q.noteId or ""))
        note = next((s for s in snaps if s["noteId"] == nid), None)
        if note is None:
            raise HTTPException(404, "노트를 찾지 못했다")
        candidates = [s for s in snaps if s["noteId"] != nid]

    emb = next(((p, EMBEDDING_PROFILES[p].dims) for p in state.embedders), None)
    t0 = time.monotonic()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=request.app), base_url="http://ai",
                                 timeout=130) as http:
        # 서버의 동시 처리 한도(AI_MAX_INFLIGHT, 기본 8)보다 작게: 설명 6개를 한 번에 보낸다
        client = AiClient("http://ai", state.settings.service_token, http,
                          concurrency=max(1, min(6, state.settings.max_inflight - 1)))
        try:
            res = await suggest_knowledge(client, note, candidates, playground_policy(), k=max(1, min(q.k, 30)),
                                          min_confidence=max(0.0, min(q.minConfidence, 1.0)), embedding_profile=emb,
                                          links_profile="links-hybrid-1" if state.typer else "links-det-1",
                                          edges_profile="edges-llm-2" if state.edge_judge else "edges-det-1",
                                          verify_profile="explanation-hybrid-2" if state.explanation_judge
                                          else "explanation-det-3")
        except AiCallError as e:
            raise HTTPException(502, f"{e.path}: {e.status} {e.error.get('code', '')} {e.error.get('message', '')}")

    def name(x):
        return names.get(x, x)
    for l in res["links"]:
        l["note"] = name(l["noteId"])
    for s in res["suggestions"]:
        s["note"] = name(s["noteId"])
        for st in s["steps"]:
            st["note"] = name(st["noteId"])
    res["ms"] = int((time.monotonic() - t0) * 1000)
    res["rules"] = [f"{r['firstType']} + {r['secondType']} → {r['resultType']}" for r in playground_policy()["compositionRules"]]
    return res


# ---- 관계 연결 그래프 ----

class GraphQuery(BaseModel):
    noteId: str | None = None
    text: str | None = None
    k: int = 8                  # 가운데 노트와 함께 띄울 노트 수
    minConfidence: float = 0.0


@router.post("/graph")
async def graph(q: GraphQuery, request: Request):
    """가운데 노트 + 가장 많이 이어진 노트 k 개, 그리고 그 노트들 사이의 모든 연결 (선).
    선마다 공유 개념, 양쪽 노트의 근거 문장, 두 노트를 가로지르는 관계 사슬(A→B, B→C)을 담는다. LLM 호출 없음."""
    from app.services.links import related_notes

    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    corpus, built_ms = await run_in_threadpool(_links_corpus, state)
    note_id = _nfc(q.noteId) if q.noteId else None
    if q.text is not None:
        if not q.text.strip() or len(q.text) > 50000:
            raise HTTPException(400, "글이 비었거나 50,000자를 넘는다")
        note_id = note_id if note_id in corpus.notes else FREE_ID
        text = _nfc(q.text)
        if note_id not in corpus.notes or corpus.notes[note_id].text != text:
            corpus = await run_in_threadpool(_with_text, state, corpus, note_id, text)
    if note_id not in corpus.notes:
        raise HTTPException(404, "노트를 찾지 못했다")
    mc = max(0.0, min(q.minConfidence, 1.0))
    rules = _example_rules()

    def work():
        center = related_notes(corpus, note_id, max(1, min(q.k, 12)), min_confidence=mc)
        nodes = [note_id] + [l.note_b for l in center]
        inside, edges = set(nodes), {}
        for n in nodes:                                   # 띄운 노트끼리의 연결도 모두 그린다
            for l in related_notes(corpus, n, len(corpus.notes), min_confidence=mc):
                key = tuple(sorted((n, l.note_b)))
                if l.note_b in inside and key not in edges:
                    edges[key] = _describe(corpus, l, rules)
        return nodes, list(edges.values())

    nodes, edges = await run_in_threadpool(work)
    return {"center": note_id, "nodes": [{"id": n, "title": n} for n in nodes], "edges": edges,
            "notes": len(corpus.notes), "buildMs": built_ms}


def _example_rules() -> dict:
    codes = {c for c, _, _ in _policy_types()}
    return {(a, b): r for a, b, r in EXAMPLE_RULES if {a, b, r} <= codes}


def _describe(corpus, link, rules) -> dict:
    """선 하나의 설명: 공유 개념과 양쪽 근거 문장, 두 노트를 가로지르는 관계 사슬 (A→B 는 한 노트, B→C 는 다른 노트)."""
    from app.services.links.linker import directed_paths

    def quote(nid, span):
        return corpus.notes[nid].text[span[0]:span[1]].strip() if span else None

    def step(nid, i):
        e = corpus.notes[nid]
        x, code, y = e.relations[i]
        ev = e.relation_evidence[i] if i < len(e.relation_evidence) else ()
        return {"note": nid, "from": e.labels.get(x, x), "type": code, "to": e.labels.get(y, y),
                "quote": ev[0][2].strip() if ev else None}

    bridges, paths, seen = [], [], set()
    for b in link.bridges:
        bridges.append({"labelA": b.label_a, "labelB": b.label_b, "contribution": round(b.contribution, 4),
                        "confidenceA": corpus.notes[link.note_a].confidence.get(b.concept_a),
                        "confidenceB": corpus.notes[link.note_b].confidence.get(b.concept_b),
                        "quoteA": quote(link.note_a, b.evidence_a), "quoteB": quote(link.note_b, b.evidence_b)})
        for p in directed_paths(corpus, link.note_a, link.note_b, b.concept_a, b.concept_b, rules):
            key = (p.note1, p.rel1, p.note2, p.rel2)
            if key not in seen:
                seen.add(key)
                paths.append({"steps": [step(p.note1, p.rel1), step(p.note2, p.rel2)], "rule": p.proposed})
    return {"a": link.note_a, "b": link.note_b, "score": link.score, "bridges": bridges, "paths": paths[:6]}


class AllGraphQuery(BaseModel):
    k: int = 3                  # 노트마다 가장 강한 연결 몇 개를 선으로 그릴지
    minConfidence: float = 0.0


@router.post("/graph/all")
async def graph_all(q: AllGraphQuery, request: Request):
    """노트 폴더 전체: 노트마다 가장 강한 연결 k 개를 선으로 (겹치면 하나). 선 설명은 /graph/edge 로 따로 받는다."""
    from app.services.links import related_notes

    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    corpus, built_ms = await run_in_threadpool(_links_corpus, state)
    mc, k = max(0.0, min(q.minConfidence, 1.0)), max(1, min(q.k, 10))

    def work():
        edges = {}
        for n in sorted(corpus.notes):
            for l in related_notes(corpus, n, k, min_confidence=mc):
                key = tuple(sorted((n, l.note_b)))
                if key not in edges or edges[key]["score"] < l.score:
                    edges[key] = {"a": key[0], "b": key[1], "score": l.score,
                                  "top": l.bridges[0].label_a if l.bridges else ""}
        degree = {}
        for e in edges.values():
            for x in (e["a"], e["b"]):
                degree[x] = degree.get(x, 0) + 1
        return ([{"id": n, "title": n, "degree": degree.get(n, 0)} for n in sorted(corpus.notes)],
                sorted(edges.values(), key=lambda e: -e["score"]))

    nodes, edges = await run_in_threadpool(work)
    return {"nodes": nodes, "edges": edges, "notes": len(corpus.notes), "buildMs": built_ms}


class EdgeQuery(BaseModel):
    a: str
    b: str
    minConfidence: float = 0.0


@router.post("/graph/edge")
async def graph_edge(q: EdgeQuery, request: Request):
    """두 노트 사이 선 하나의 설명 (공유 개념·근거 문장·관계 사슬)."""
    from app.services.links import related_notes

    state = request.app.state
    if state.extractor is None:
        raise HTTPException(503, "추출기가 준비되지 않았다")
    corpus, _ = await run_in_threadpool(_links_corpus, state)
    a, b = _nfc(q.a), _nfc(q.b)
    if a not in corpus.notes or b not in corpus.notes:
        raise HTTPException(404, "노트를 찾지 못했다")
    mc = max(0.0, min(q.minConfidence, 1.0))

    def work():
        link = next((l for l in related_notes(corpus, a, len(corpus.notes), min_confidence=mc) if l.note_b == b), None)
        return _describe(corpus, link, _example_rules()) if link else None

    out = await run_in_threadpool(work)
    if out is None:
        raise HTTPException(404, "두 노트는 이어져 있지 않다")
    return out
