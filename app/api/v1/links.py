"""POST /internal/v1/note-links (findNoteLinks) — 노트 사이 연결과 A-B-C 경로 후보. 결정적.

⚠️ 계약 v0.1.0 에 없는 AI 쪽 제안 operation 이다. 계약상 2-hop 경로·순위는 BE 책임(F-PATH-01, F-RANK-01)이므로
BE 가 그대로 쓸지, 자기 그래프에서 같은 방법을 구현할지, 계약에 넣을지는 팀이 정한다.

무상태: BE 가 트리거 노트와 후보 노트(같은 사용자의 노트)를 보내면, 각 노트에서 개념·관계를 뽑아
  1. 노트 쌍 점수 (공유 개념의 드묾 가중 합, Adamic-Adar). minConfidence 이상인 개념만 쓴다 (양쪽 모두)
  2. 공유 개념(매개 B)마다 양쪽 노트의 근거 문장 (UTF-16 위치)
  3. A —(한 노트 관계)→ B —(다른 노트 관계)→ C 경로 후보와 합성 규칙 결과 (proposedTypeCode)
를 돌려준다. 경로의 각 단계 근거는 /extract 응답의 evidence 와 같은 구간이라 BE 가 저장한 relation·evidence UUID 로
대응시킨 뒤 /verifications/* 와 /explanations 를 부를 수 있다.
"""

from fastapi import APIRouter, Request

from app.api.v1.extract import select_concepts
from app.core.config import EXTRACT_PROFILE, EXTRACT_PROFILES
from app.core.errors import AiError
from app.schemas.reasoning import LinkBridge, LinkPath, LinkSpan, LinkStep, LinksRequest, LinksResponse, NoteLink
from app.services.concepts.markdown import Utf16Index
from app.services.links import build_corpus, related_notes
from app.services.links.corpus import EntryMemo
from app.services.links.linker import directed_paths

from .grounding import check_policy, check_snapshot
from .reasoning import build, meta, prepare, run

router = APIRouter()
MEMO = EntryMemo()     # 같은 노트·같은 설정이면 색인 항목을 다시 만들지 않는다 (BE 가 같은 노트를 반복해 보낸다)


@router.post("/note-links", response_model=LinksResponse, operation_id="findNoteLinks")
async def note_links(request: Request):
    t0, data, req, profile = await prepare(request, LinksRequest, "links")
    ctx, note = req.context, req.note
    bad = [f"context.{f}" for f in ("noteId", "noteVersion", "inputRevision")
           if str(getattr(ctx, f)).lower() != str(getattr(note, f)).lower()]
    if bad:
        raise AiError("INVALID_REFERENCE", [(f, "invalid") for f in bad])
    check_snapshot(note, "note")
    seen = {note.noteId.lower()}
    for i, d in enumerate(req.candidates):
        if d.noteId.lower() in seen:
            raise AiError("INVALID_REFERENCE", [(f"candidates.{i}.noteId", "duplicate")])
        seen.add(d.noteId.lower())
        check_snapshot(d, f"candidates.{i}")
    codes, rules = check_policy(req.policy)
    policy_types = [(t.code, t.label, t.description) for t in req.policy.relationTypes]
    snapshots = {d.noteId: d for d in (note, *req.candidates)}
    state = request.app.state

    def work():
        extract_profile = EXTRACT_PROFILES[EXTRACT_PROFILE]
        corpus = build_corpus({k: d.body for k, d in snapshots.items()}, state.extractor,
                              lambda ms: select_concepts(ms, extract_profile), policy_types, extract_profile.limits,
                              lock=state.kiwi_lock, typer=state.typer if "typer" in profile.requires else None,
                              version=f"{state.extractor.method_version}:{profile.revision}", memo=MEMO)
        links = related_notes(corpus, note.noteId, req.k, min_confidence=req.minConfidence)
        u16 = {k: Utf16Index(e.text) for k, e in corpus.notes.items()}

        def span(nid, s):
            if s is None:
                return None
            a, b = s
            return LinkSpan(noteId=nid, startOffset=u16[nid](a), endOffset=u16[nid](b), quote=corpus.notes[nid].text[a:b])

        def step(order, nid, i):
            e = corpus.notes[nid]
            x, code, y = e.relations[i]
            return LinkStep(stepOrder=order, noteId=nid, fromLabel=e.labels.get(x, x), toLabel=e.labels.get(y, y),
                            typeCode=code, evidence=[LinkSpan(noteId=nid, startOffset=s, endOffset=t, quote=q)
                                                     for s, t, q in e.relation_evidence[i]])

        out = []
        for l in links:
            bridges, paths = [], []
            for b in l.bridges:
                bridges.append(LinkBridge(
                    label=b.label_a, candidateLabel=b.label_b,
                    confidence=corpus.notes[l.note_a].confidence.get(b.concept_a, 0.0),
                    candidateConfidence=corpus.notes[l.note_b].confidence.get(b.concept_b, 0.0),
                    contribution=b.contribution, evidence=span(l.note_a, b.evidence_a),
                    candidateEvidence=span(l.note_b, b.evidence_b)))
                for p in directed_paths(corpus, l.note_a, l.note_b, b.concept_a, b.concept_b, rules):
                    s1, s2 = step(1, p.note1, p.rel1), step(2, p.note2, p.rel2)
                    paths.append(LinkPath(source=s1.fromLabel, bridge=s1.toLabel, target=s2.toLabel, steps=[s1, s2],
                                          proposedTypeCode=p.proposed))
            out.append(NoteLink(noteId=l.note_b, score=l.score, bridges=bridges, paths=paths[:10]))
        return out

    links = await run(request, profile, work)
    typed = "typer" in profile.requires
    model = f"{profile.model}:{state.typer.model}" if typed else None
    return build(LinksResponse, context=data["context"],
                 meta=meta(profile, t0, "local+" + state.typer.provider if typed else "local", model), links=links)
