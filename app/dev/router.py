"""개발용 playground (AI_DEV_UI=1 또는 `ai serve --dev` 일 때만 붙는다).

  GET /dev/              playground 화면
  GET /dev/config        화면 초기값 (profile 목록, 팀 관계 유형 초안, 로컬 접속이면 서비스 토큰)
  GET /dev/notes         노트 폴더(AI_NOTES_DIR, 기본 data/private/notes)의 노트 id 목록
  GET /dev/notes/{id}    노트 원문 (NFC)
  GET /dev/index         노트 전체 색인을 미리 만든다 (노트 사이 연결용)
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
        # 서버의 동시 처리 한도(AI_MAX_INFLIGHT, playground 기본 8)보다 작게: 설명 6개를 한 번에 보낸다
        client = AiClient("http://ai", state.settings.service_token, http,
                          concurrency=max(1, min(6, state.settings.max_inflight - 1)))
        try:
            res = await suggest_knowledge(client, note, candidates, playground_policy(), k=max(1, min(q.k, 30)),
                                          min_confidence=max(0.0, min(q.minConfidence, 1.0)), embedding_profile=emb,
                                          links_profile="links-hybrid-1" if state.typer else "links-det-1",
                                          edges_profile="edges-llm-1" if state.edge_judge else "edges-det-1")
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
