"""BE 가 내부 AI API 를 부르는 방법의 참조 구현 (실행되는 예시). BE 는 Java 로 같은 일을 한다.

  AiClient           계약 헤더(Bearer, X-Request-Id, X-Attempt-Id, Idempotency-Key, X-Input-SHA256)를 붙여 호출
  suggest_knowledge  노트 하나에서 시작하는 전체 흐름:
      1. /note-links                연결된 노트와 A-B-C 경로 후보 (AI 제안 operation. 계약상 BE 가 직접 해도 된다).
                                    links-hybrid-1 은 관계를 LLM 으로 보탠다 (규칙만으로는 경로가 거의 없다)
      2. /verifications/bridges     매개 개념 B 가 두 노트에서 같은 개념인가
      3. /verifications/edges       A→B, B→C 간선이 근거에 정말 쓰여 있는가 (edges-llm-1: LLM 이 근거를 읽고 판정)
      4. /explanations              근거 기반 설명 = 새 지식 제안 (A —합성 유형→ C). 항상 LLM 이 쓴다 (explain-llm-8).
                                    단계 근거와 함께 문맥 근거(/note-links 단계의 context)를 보내 결론이 인용해 풀어 쓴다
      5. /verifications/explanations 설명의 주장별 판정 (explanation-hybrid-2: 근거 뒷받침·구체성·새로움을 LLM 이 본다)
      6. 판정 결합 (계약 §5 표): 하나라도 탈락이면 제외, 애매·가정·규칙 없음이면 가설, 아니면 후보.
         제외된 제안은 보여 주지 않는다 (screened.rejected 에 이유만 남긴다)

여기서 만드는 UUID 는 BE 의 저장 UUID 를 흉내 낸 것이다 (uuid5). 실제 BE 는 자기가 저장한
concept·relation·evidence UUID 를 쓴다. 경로 단계의 근거 구간은 /extract 응답의 evidence 와 같은 위치라서
BE 는 (noteId, startOffset, endOffset) 으로 자기 evidence UUID 를 찾으면 된다.
"""

import asyncio
import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field

import httpx

from app.services.links.corpus import loose_key

NS = uuid.UUID("5b8f0a52-8f0e-4d5c-9a51-7b0e0c6f3a10")
_STAMP_FIELDS = ("profileVersion", "provider", "model", "modelRevision", "promptVersion", "schemaVersion",
                 "preprocessingVersion")


class AiCallError(Exception):
    def __init__(self, path: str, status: int, error: dict | None):
        self.path, self.status, self.error = path, status, error or {}
        super().__init__(f"{path} → {status} {self.error.get('code', '')}")


def uid(*parts) -> str:
    return str(uuid.uuid5(NS, "\x1f".join(str(p) for p in parts)))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(note_id: str, title: str, body: str, topic_id: str | None = None, topic: str = "미분류",
             version: int = 1, revision: int = 1) -> dict:
    return {"noteId": note_id, "noteVersion": version, "inputRevision": revision, "title": title[:200] or "제목 없음",
            "body": body, "bodySha256": sha(body), "topic": {"id": topic_id, "name": topic}}


def stamp(meta: dict) -> dict:
    """응답 meta → 다음 검증 요청의 generationProvenance 항목 (durationMs 제외)."""
    return {k: meta[k] for k in _STAMP_FIELDS}


class AiClient:
    def __init__(self, base_url: str, token: str, http: httpx.AsyncClient | None = None, timeout: float = 130,
                 concurrency: int = 6):
        self.http = http or httpx.AsyncClient(base_url=base_url, timeout=timeout)
        self.token = token
        # 동시에 보내는 요청 수. AI 서버의 AI_MAX_INFLIGHT(기본 8)보다 작게 둔다 (넘으면 429 AI_BUSY)
        self.sem = asyncio.Semaphore(concurrency)

    def context(self, note: dict, user_id: str = "00000000-0000-4000-8000-000000000001") -> dict:
        return {"analysisJobId": str(uuid.uuid4()), "stageExecutionId": str(uuid.uuid4()), "userId": user_id,
                "noteId": note["noteId"], "noteVersion": note["noteVersion"], "inputRevision": note["inputRevision"],
                "graphRevision": 1, "pipelineVersion": "reference-client-1"}

    async def call(self, path: str, body: dict) -> dict:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")     # 해시는 실제로 보내는 바이트로 계산한다
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
                   "Accept": "application/json", "X-Request-Id": str(uuid.uuid4()),
                   "X-Attempt-Id": str(uuid.uuid4()), "Idempotency-Key": body["context"]["stageExecutionId"],
                   "X-Input-SHA256": hashlib.sha256(raw).hexdigest()}
        async with self.sem:
            r = await self.http.post("/internal/v1" + path, content=raw, headers=headers)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        if r.status_code != 200:
            raise AiCallError(path, r.status_code, (data or {}).get("error"))
        return data


def decide(bridge: str | None, edges: list[str], overall: str, proposed: str | None) -> tuple[str, list[str]]:
    """계약 §5 표의 BE 처리를 결합한다. (candidate | hypothesis | excluded, 이유들)"""
    why = []
    if bridge == "different":
        why.append("매개 개념이 두 노트에서 다른 개념입니다")
    if "contradicted" in edges:
        why.append("간선 근거가 관계를 부정합니다")
    if overall == "fail":
        why.append("설명 검증에 실패했습니다")
    if why:
        return "excluded", why
    if bridge in ("uncertain", None):
        why.append("매개 개념이 같은지 애매합니다" if bridge else "매개 개념을 검증할 근거가 없습니다")
    if "unknown" in edges:
        why.append("간선 근거가 관계를 분명히 말하지 않습니다")
    if overall == "uncertain":
        why.append("설명에 애매한 주장이나 빈 단계가 있습니다")
    if proposed is None:
        why.append("합성 규칙이 없습니다")
    return ("hypothesis", why) if why else ("candidate", ["모든 검증을 통과했습니다"])


# 분류 계층만 올라가는 경로(A 는 B 의 한 종류, B 는 C 의 한 종류)는 결론이 뻔하기 쉬워 같은 조건이면 뒤로 민다
TAXONOMY_CODES = frozenset({"is_a", "example_of"})


async def suggest_knowledge(client: AiClient, note: dict, candidates: list[dict], policy: dict, k: int = 5,
                            min_confidence: float = 0.0, max_paths: int = 6, pool: int = 12,
                            embedding_profile: tuple[str, int] | None = None,
                            explain_profile: str = "explain-llm-8", links_profile: str = "links-hybrid-1",
                            edges_profile: str = "edges-llm-2", verify_profile: str = "explanation-hybrid-2",
                            taxonomy_codes: frozenset[str] = TAXONOMY_CODES) -> dict:
    """note 와 연결된 노트·새 지식 제안. 단계를 나눠 싼 검사부터 하고 비싼 LLM 설명은 살아남은 경로에만 쓴다.

      1. /note-links 로 경로 후보를 넓게 (pool 개). 합성 규칙이 있는 경로 먼저, 연결 노트마다 돌아가며 고른다
      2. 매개 개념(/verifications/bridges)·간선(/verifications/edges) 검사를 묶어서 한꺼번에, 동시에
      3. 제외된 경로(매개가 다른 개념, 간선이 부정됨)는 버리고, 같은 A–C 경로는 하나로 합치고(근거는 모은다),
         남은 경로를 가치 순으로 고른다 (pick_paths)
      4. 고른 max_paths 개만 설명(/explanations)과 설명 검증을 동시에. 검증에서 제외된 제안은 버린다
    embedding_profile=(id, 차원) 이면 매개 개념 쌍의 cosine 을 /embeddings 로 구한다."""
    by_id = {n["noteId"]: n for n in (note, *candidates)}
    links = await client.call("/note-links", {"context": client.context(note), "profileVersion": links_profile,
                                              "note": note, "candidates": candidates, "policy": policy, "k": k,
                                              "minConfidence": min_confidence})
    links_stamp = stamp(links["meta"])
    labels = {t["code"]: t["label"] for t in policy["relationTypes"]}

    preps = [_prepare(note, by_id, path, labels, link_id, rank)
             for rank, (link_id, path) in enumerate(select_paths(links["links"], pool))]
    await asyncio.gather(_check_bridges(client, note, preps, links_stamp, embedding_profile),
                         _check_edges(client, note, preps, policy, links_stamp, edges_profile))

    excluded = [p for p in preps if _excluded(p)]
    chosen, merged = pick_paths([p for p in preps if not _excluded(p)], max_paths, taxonomy_codes)
    done = await asyncio.gather(*(_explain(client, note, p, policy, explain_profile, verify_profile) for p in chosen))
    order = {"candidate": 0, "hypothesis": 1}
    suggestions = sorted((s for s in done if s["status"] != "excluded"), key=lambda s: order[s["status"]])
    rejected = [{"source": s["source"], "bridge": s["bridge"], "target": s["target"], "knowledge": s["knowledge"],
                 "reasons": s["statusReasons"], "explanationVerdict": s["explanationVerdict"]}
                for s in done if s["status"] == "excluded"]
    return {"links": links["links"], "linksMeta": links["meta"], "suggestions": suggestions,
            "screened": {"paths": len(preps), "excluded": len(excluded), "merged": merged, "explained": len(chosen),
                         "rejected": rejected}}


def _excluded(p) -> bool:
    return (p.bridge or {}).get("decision") == "different" or any(v["outcome"] == "contradicted" for v in p.edge_verdicts)


def pick_paths(preps: list, max_paths: int, taxonomy_codes: frozenset[str] = TAXONOMY_CODES) -> tuple[list, int]:
    """설명할 경로 고르기. (고른 경로, 합친 경로 수)

      1. 같은 A–C (표기 대소문자·띄어쓰기 무시) 경로는 하나로: 가치가 가장 높은 것을 남기고, 다른 경로의 단계 근거 중
         같은 두 개념을 잇는 것을 그 단계 근거에 보탠다 (같은 지식을 두 번 제안하지 않는다, 기능 명세 1.4)
      2. 가치 순: 합성 규칙 있음 → 분류 계층만 올라가는 경로가 아님 → 근거가 확인된 간선 수 → 매개 같음 → 연결 순위
      3. 이미 고른 경로와 첫 단계(A→B)나 끝 단계(B→C)가 같은 경로(Add→EI→트랜잭션 기능 뒤의 Change→EI→트랜잭션 기능)는
         다른 경로를 먼저 고른 뒤에 채운다"""
    def value(p):
        codes = {s["typeCode"] for s in p.path["steps"]}
        return (p.path["proposedTypeCode"] is not None, not codes <= taxonomy_codes,
                sum(v["outcome"] == "entailed" for v in p.edge_verdicts), (p.bridge or {}).get("decision") == "same",
                -p.rank)

    groups: dict = {}
    for p in sorted(preps, key=value, reverse=True):
        s1, s2 = p.path["steps"]
        key = (loose_key(s1["fromLabel"]), loose_key(s2["toLabel"]))
        if key in groups:
            _absorb(groups[key], p)
        else:
            groups[key] = p
    ranked = list(groups.values())                         # 이미 가치 순
    out, used = [], set()
    for p in ranked:
        s1, s2 = p.path["steps"]
        ends = {("head", loose_key(s1["fromLabel"]), loose_key(s1["toLabel"])),
                ("tail", loose_key(s2["fromLabel"]), loose_key(s2["toLabel"]))}
        if not ends & used and len(out) < max_paths:
            out.append(p)
            used |= ends
    out += [p for p in ranked if all(p is not q for q in out)][:max_paths - len(out)]
    return sorted(out, key=value, reverse=True), len(preps) - len(ranked)


def _absorb(keep: "_Prep", dup: "_Prep", per_step: int = 8):
    """dup 의 단계 근거 중 keep 의 같은 단계와 같은 두 개념(표기 무시)을 잇는 것을 keep 의 그 단계 근거에 보탠다."""
    known = {e["evidenceId"] for e in keep.evidence}
    for i, (ks, ds) in enumerate(zip(keep.path["steps"], dup.path["steps"])):
        if (loose_key(ks["fromLabel"]), loose_key(ks["toLabel"])) != (loose_key(ds["fromLabel"]), loose_key(ds["toLabel"])):
            continue
        ids = keep.path_body["steps"][i]["evidenceIds"]
        for eid, span in zip(dup.path_body["steps"][i]["evidenceIds"], ds["evidence"]):
            if eid in ids or len(ids) >= per_step:
                continue
            ids.append(eid)
            ks.setdefault("evidence", []).append(span)
            if eid not in known:
                keep.evidence.append(next(e for e in dup.evidence if e["evidenceId"] == eid))
                known.add(eid)
            if span["noteId"] not in {d["noteId"] for d in keep.documents}:
                keep.documents.append(next(d for d in dup.documents if d["noteId"] == span["noteId"]))
    keep.context += [e for e in dup.context if e["evidenceId"] not in {x["evidenceId"] for x in keep.context}]


def select_paths(links: list[dict], pool: int) -> list[tuple[str, dict]]:
    """여러 연결 노트에서 고르게: 합성 규칙이 있는 경로를 먼저 연결 노트마다 돌아가며, 그다음 규칙 없는 경로를. 같은 경로는 한 번."""
    def key(p):
        return tuple((s["noteId"], s["fromLabel"], s["typeCode"], s["toLabel"]) for s in p["steps"])
    out, seen = [], set()
    for with_rule in (True, False):
        queues = [[p for p in l["paths"] if (p["proposedTypeCode"] is not None) == with_rule] for l in links]
        while any(queues) and len(out) < pool:
            for l, q in zip(links, queues):
                while q and key(q[0]) in seen:
                    q.pop(0)
                if q and len(out) < pool:
                    p = q.pop(0)
                    seen.add(key(p))
                    out.append((l["noteId"], p))
    return out


@dataclass
class _Prep:
    path: dict
    linked_note: str
    rank: int
    documents: list
    evidence: list
    path_body: dict
    edges: list
    pair: dict | None
    labels: dict
    context: list = field(default_factory=list)      # 문맥 근거 (설명에만 보낸다)
    bridge: dict | None = None
    edge_verdicts: list = field(default_factory=list)


def _prepare(note, by_id, path, labels, linked_note, rank) -> _Prep:
    """한 경로의 요청 재료 (호출 없음). UUID 는 BE 의 저장 UUID 를 흉내 낸 것이다 (uuid5)."""
    path = copy.deepcopy(path)                         # 같은 A–C 경로를 합칠 때 단계 근거를 보탠다
    s1, s2 = path["steps"]
    n1, n2 = by_id[s1["noteId"]], by_id[s2["noteId"]]
    evidence, step_ev = [], []
    for s in (s1, s2):
        ids = []
        for e in s["evidence"]:
            eid = uid("evidence", e["noteId"], e["startOffset"], e["endOffset"])
            if eid not in {x["evidenceId"] for x in evidence}:
                evidence.append({"evidenceId": eid, "noteId": e["noteId"], "noteVersion": by_id[e["noteId"]]["noteVersion"],
                                 "bodySha256": by_id[e["noteId"]]["bodySha256"], "startOffset": e["startOffset"],
                                 "endOffset": e["endOffset"], "quote": e["quote"]})
            ids.append(eid)
        step_ev.append(ids)
    quote = {e["evidenceId"]: e["quote"] for e in evidence}
    a_id, b_id, c_id = uid("concept", n1["noteId"], s1["fromLabel"]), uid("concept", loose_key(path["bridge"])), \
        uid("concept", n2["noteId"], s2["toLabel"])
    rel_ids = [uid("relation", s["noteId"], s["fromLabel"], s["typeCode"], s["toLabel"]) for s in (s1, s2)]
    path_body = {"pathId": uid("path", *rel_ids),
                 "concepts": [{"conceptId": a_id, "label": s1["fromLabel"]}, {"conceptId": b_id, "label": path["bridge"]},
                              {"conceptId": c_id, "label": s2["toLabel"]}],
                 "sourceConceptId": a_id, "bridgeConceptId": b_id, "targetConceptId": c_id,
                 "steps": [{"stepOrder": i + 1, "relationId": rel_ids[i], "fromConceptId": f, "toConceptId": t,
                            "typeCode": s["typeCode"], "evidenceIds": step_ev[i]}
                           for i, (s, f, t) in enumerate(((s1, a_id, b_id), (s2, b_id, c_id)))],
                 "proposedTypeCode": path["proposedTypeCode"]}
    edges = [{"edgeId": rel_ids[i], "source": {"conceptId": f, "label": s["fromLabel"]},
              "target": {"conceptId": t, "label": s["toLabel"]}, "typeCode": s["typeCode"], "evidenceIds": step_ev[i]}
             for i, (s, f, t) in enumerate(((s1, a_id, b_id), (s2, b_id, c_id)))]
    # 매개 개념: 단계 1 근거 속 B 와 단계 2 근거 속 B (표기가 근거에 그대로 있어야 한다)
    # 대소문자가 달라도 찾는다 (Page Table / page table). surface 는 근거에 실제로 적힌 그대로 보낸다
    def find(label, ids):
        for x in ids:
            m = re.search(re.escape(label), quote[x], re.I)
            if m:
                return x, m.group(0)
        return None, None
    left, lsurf = find(s1["toLabel"], step_ev[0])
    right, rsurf = find(s2["fromLabel"], step_ev[1])
    pair = None
    if left and right:
        pair = {"left": {"conceptId": b_id, "label": s1["toLabel"], "topicId": n1["topic"]["id"], "evidenceId": left,
                         "surface": lsurf},
                "right": {"conceptId": uid("concept", n2["noteId"], s2["fromLabel"]), "label": s2["fromLabel"],
                          "topicId": n2["topic"]["id"], "evidenceId": right, "surface": rsurf}}
    step_ids = {e["evidenceId"] for e in evidence}
    context = []
    for s in (s1, s2):
        for e in s.get("context", []):
            eid = uid("evidence", e["noteId"], e["startOffset"], e["endOffset"])
            if eid not in step_ids and eid not in {x["evidenceId"] for x in context}:
                context.append({"evidenceId": eid, "noteId": e["noteId"], "noteVersion": by_id[e["noteId"]]["noteVersion"],
                                "bodySha256": by_id[e["noteId"]]["bodySha256"], "startOffset": e["startOffset"],
                                "endOffset": e["endOffset"], "quote": e["quote"]})
    return _Prep(path, linked_note, rank, [n1, n2], evidence, path_body, edges, pair, labels, context=context)


# 계약의 요청 상한 (ai-api.md 필드 사전)
MAX_DOCS, MAX_EVIDENCE = 10, 40
EDGE_BATCH = 6


def _batches(items: list, docs_of, ev_of, max_items: int):
    """문서 10개·근거 40개·항목 max_items 개를 넘지 않게 묶는다. 각 묶음은 (항목들, 문서들, 근거들)."""
    out, cur, docs, ev = [], [], {}, {}
    for it in items:
        d = {x["noteId"]: x for x in docs_of(it)}
        e = {x["evidenceId"]: x for x in ev_of(it)}
        if cur and (len(cur) >= max_items or len(docs | d) > MAX_DOCS or len(ev | e) > MAX_EVIDENCE):
            out.append((cur, list(docs.values()), list(ev.values())))
            cur, docs, ev = [], {}, {}
        cur.append(it)
        docs.update(d)
        ev.update(e)
    if cur:
        out.append((cur, list(docs.values()), list(ev.values())))
    return out


async def _check_bridges(client, note, preps, links_stamp, embedding_profile):
    items = [p for p in preps if p.pair]
    if not items:
        return
    sim = {}
    if embedding_profile:                                      # 매개 표기 쌍의 cosine (한 번에, 32개씩)
        pid, dims = embedding_profile
        texts = sorted({t for p in items for t in (p.pair["left"]["label"], p.pair["right"]["label"])})
        vec = {}
        for i in range(0, len(texts), 32):
            chunk = texts[i:i + 32]
            r = await client.call("/embeddings", {"context": client.context(note), "profileVersion": pid,
                                                  "expectedDimensions": dims,
                                                  "items": [{"itemId": f"t{i + j}", "text": t} for j, t in enumerate(chunk)]})
            got = {it["itemId"]: it["vector"] for it in r["items"]}
            vec.update({t: got[f"t{i + j}"] for j, t in enumerate(chunk)})
        norm = lambda x: sum(t * t for t in x) ** 0.5 or 1.0
        for p in items:
            a, b = vec[p.pair["left"]["label"]], vec[p.pair["right"]["label"]]
            sim[id(p)] = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (norm(a) * norm(b))))

    def ev_of(p):
        want = {p.pair["left"]["evidenceId"], p.pair["right"]["evidenceId"]}
        return [e for e in p.evidence if e["evidenceId"] in want]

    async def run(batch):
        group, docs, ev = batch
        pairs = [{"pairId": f"p{n}", **g.pair, "similarity": round(sim.get(id(g), 0.0), 6)} for n, g in enumerate(group)]
        r = await client.call("/verifications/bridges", {"context": client.context(note), "profileVersion": "bridges-det-1",
                                                         "documents": docs, "evidence": ev, "pairs": pairs,
                                                         "generationProvenance": [links_stamp]})
        got = {v["pairId"]: v for v in r["verdicts"]}
        for n, g in enumerate(group):
            g.bridge = got[f"p{n}"]

    await asyncio.gather(*(run(b) for b in _batches(items, lambda p: p.documents, ev_of, 20)))


async def _check_edges(client, note, preps, policy, links_stamp, edges_profile):
    """같은 간선(relation)은 경로가 여러 개여도 한 번만 묻는다. 간선 EDGE_BATCH 개·문서 10개·근거 40개씩 묶어 동시에."""
    uniq, owner = {}, {}
    for p in preps:
        for e in p.edges:
            uniq.setdefault(e["edgeId"], e)
            owner.setdefault(e["edgeId"], p)

    def docs_of(e):
        p = owner[e["edgeId"]]
        return [d for d in p.documents if any(x["noteId"] == d["noteId"] for x in ev_of(e))]

    def ev_of(e):
        return [x for x in owner[e["edgeId"]].evidence if x["evidenceId"] in e["evidenceIds"]]

    verdicts = {}

    async def run(batch):
        group, docs, ev = batch
        r = await client.call("/verifications/edges", {"context": client.context(note), "profileVersion": edges_profile,
                                                       "documents": docs, "evidence": ev, "policy": policy,
                                                       "edges": group, "generationProvenance": [links_stamp]})
        verdicts.update({v["edgeId"]: v for v in r["verdicts"]})

    # 한 번에 20개를 묻으면 응답 하나를 오래 기다린다. 6개씩 나눠 동시에 묻는다 (간선별 캐시라 비용은 같다)
    await asyncio.gather(*(run(b) for b in _batches(list(uniq.values()), docs_of, ev_of, EDGE_BATCH)))
    for p in preps:
        p.edge_verdicts = [verdicts[e["edgeId"]] for e in p.edges]


async def _explain(client, note, p: _Prep, policy, explain_profile, verify_profile) -> dict:
    known = {e["evidenceId"] for e in p.evidence}
    context = [e for e in p.context if e["evidenceId"] not in known][:MAX_EVIDENCE - len(p.evidence)]
    base = lambda profile: {"context": client.context(note), "profileVersion": profile, "documents": p.documents,
                            "evidence": p.evidence + context}                # 문맥 근거도 같이: 설명이 인용해 풀어 쓴다
    s1, s2 = p.path["steps"]
    ex = await client.call("/explanations", {**base(explain_profile), "path": p.path_body, "policy": policy})
    ver = await client.call("/verifications/explanations", {
        **base(verify_profile), "path": p.path_body, "policy": policy, "explanation": ex["explanation"],
        "generationProvenance": [stamp(ex["meta"])]})
    status, why = decide(p.bridge["decision"] if p.bridge else None, [v["outcome"] for v in p.edge_verdicts],
                         ver["overall"], p.path["proposedTypeCode"])
    result = {"noteId": p.linked_note, "source": s1["fromLabel"], "bridge": p.path["bridge"], "target": s2["toLabel"],
              "steps": [s1, s2], "stepEvidenceIds": [st["evidenceIds"] for st in p.path_body["steps"]],
              "proposedTypeCode": p.path["proposedTypeCode"],
              "proposedLabel": p.labels.get(p.path["proposedTypeCode"]), "edgeVerdicts": p.edge_verdicts,
              "explanation": ex["explanation"],
              "explainedBy": {"profile": ex["meta"]["profileVersion"], "provider": ex["meta"]["provider"],
                              "model": ex["meta"]["model"]},
              "explanationVerdict": {"overall": ver["overall"], "claimVerdicts": ver["claimVerdicts"], "gaps": ver["gaps"]},
              "status": status, "statusReasons": why,
              "knowledge": next(c["text"] for c in ex["explanation"]["claims"] if c["kind"] == "conclusion")}
    if p.bridge:
        result["bridgeVerdict"] = p.bridge
    return result
