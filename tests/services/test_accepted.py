"""'이해함' 으로 받아들인 새 지식: 기록 만들기, 저장, 기댄 구간 확인, 키워드·개념으로 합치기."""

import json

import pytest

from app.client import uid
from app.schemas.knowledge import Concept, Keyword, PageSource, Span
from app.services.links.accepted import (LOST, LIVE, AcceptedStore, AcceptError, build_record, locate, merge_into, status_of,
                                         usable_as_element)
from app.services.links.generate import KnowledgeGenerator, merge_external, side_sources

QA = "WAL 은 트랜잭션이 커밋되기 전에 변경 내용을 로그로 디스크에 먼저 기록하는 방식이다."
QB = "페이지 캐시는 디스크에서 읽은 데이터를 RAM 에 저장해 재사용하는 방식이다."
TEXTS = {"DB-WAL": f"# WAL\n\n{QA}\n", "OS-캐시": f"# 캐시\n\n{QB}\n"}
EXT = [{"url": "https://example.com/innodb", "title": "InnoDB", "quote": "O_DIRECT 는 OS 페이지 캐시를 거치지 않는다."}]
CONCLUSION = "DB 도 OS 도 같은 페이지를 메모리에 캐시해서 같은 데이터가 두 번 올라갈 수 있어요."


def side(keyword, note, text, cid):
    start = TEXTS[note].index(text)
    return {"keyword": keyword, "keywordId": f"k-{keyword}", "note": note, "noteId": uid("note", note), "conceptId": cid, "text": text,
            "start": start, "end": start + len(text), "quote": text}


A, B = side("WAL", "DB-WAL", QA, "ca"), side("페이지 캐시", "OS-캐시", QB, "cb")


def record(**over):
    args = dict(a=A, b=B, new_keyword="이중 버퍼링", conclusion=CONCLUSION, external=EXT, concept_id=uid("generated", "ca", "cb"))
    args.update(over)
    return build_record(args["a"], args["b"], args["new_keyword"], args["conclusion"], args["external"], args["concept_id"])


def extracted(name, note, text, cid):
    s = side(name, note, text, cid)
    return Keyword(keywordId=uid("keyword", name), name=name, concepts=[Concept(
        conceptId=uid("concept", cid), origin="extracted", explanation=text,
        sources=[PageSource(noteId=s["noteId"], noteVersion=1, span=Span(startOffset=s["start"], endOffset=s["end"], quote=text))])])


# ---- 기록 ----

def test_record_holds_both_sources_and_the_external_material():
    r = record()
    assert r["keyword"] == "이중 버퍼링" and r["conclusion"] == CONCLUSION and r["conceptId"] == uid("generated", "ca", "cb")
    assert [s["note"] for s in r["sources"]] == ["DB-WAL", "OS-캐시"] and r["sources"][0]["quote"] == QA
    assert r["external"][0]["url"] == EXT[0]["url"] and r["external"][0]["retrievedAt"].endswith("Z")
    assert [f["keyword"] for f in r["from"]] == ["WAL", "페이지 캐시"]


@pytest.mark.parametrize("over, why", [
    ({"new_keyword": "  "}, "새 키워드"), ({"new_keyword": "가" * 101}, "새 키워드"),
    ({"conclusion": ""}, "결론"), ({"conclusion": "가" * 2001}, "결론"),
    ({"external": [{"url": "javascript:alert(1)"}]}, "개념 형식"),          # http(s) 만 외부 자료가 된다
])
def test_record_rejects_what_cannot_become_a_concept(over, why):
    with pytest.raises(AcceptError, match=why):
        record(**over)


def test_record_needs_two_different_notes():
    other = {**B, "noteId": A["noteId"], "note": "DB-WAL"}
    with pytest.raises(AcceptError, match="서로 다른 두 노트"):
        record(b=other)


def test_a_card_built_on_accepted_knowledge_inherits_all_its_sources_and_external_material():
    third = "체크포인트는 장애가 났을 때 복구 시간을 줄이려고 쓰는 기록 지점이다."
    TEXTS["DB-체크"] = f"# 체크\n\n{third}\n"
    c = side("체크포인트", "DB-체크", third, "cc")
    g = {**A, "keyword": "이중 버퍼링", "conceptId": "cg", "note": "새 지식 · DB-WAL · OS-캐시", "origin": "generated", "text": CONCLUSION,
         "notes": ["DB-WAL", "OS-캐시"], "sources": record()["sources"], "external": record()["external"]}
    r = record(a=g, b=c, new_keyword="복구와 캐시", concept_id=uid("generated", "cg", "cc"),
               external=[{"url": "https://example.com/new", "title": "New", "quote": None}])
    assert [s["note"] for s in r["sources"]] == ["DB-WAL", "OS-캐시", "DB-체크"]            # 세 노트 전부
    assert [e["url"] for e in r["external"]] == ["https://example.com/new", EXT[0]["url"]]   # 새 자료 + 이어받은 자료


def test_generator_concept_lists_every_page_the_sides_rest_on():
    g = {**A, "origin": "generated", "sources": record()["sources"], "external": record()["external"]}
    third = {**B, "noteId": uid("note", "C"), "note": "C", "start": 3, "end": 9, "quote": "체크포인트"}
    got = KnowledgeGenerator._concept("11111111-1111-4111-8111-111111111111", g, third, CONCLUSION, [{"url": "https://example.com/new", "title": "", "quote": ""}])
    assert [s["noteId"] for s in got["sources"]] == [uid("note", "DB-WAL"), uid("note", "OS-캐시"), uid("note", "C")]
    assert [e["url"] for e in got["external"]] == ["https://example.com/new", EXT[0]["url"]]
    assert side_sources(A, {**A}) == side_sources(A)                    # 같은 구간은 한 번
    assert len(merge_external([{"url": f"https://x/{i}"} for i in range(15)])) == 10


# ---- 저장 ----

def test_store_persists_dedupes_and_removes(tmp_path):
    path = tmp_path / "sub" / "accepted.json"
    s = AcceptedStore(path)
    assert s.all() == [] and s.version == 0
    first = s.add(record())
    assert s.version == 1 and json.loads(path.read_text(encoding="utf-8"))["items"][0]["conceptId"] == first["conceptId"]
    again = s.add({**record(), "conclusion": "다시 눌러도 하나예요. " + CONCLUSION, "createdAt": "2099-01-01T00:00:00Z"})
    assert len(s.all()) == 1 and again["createdAt"] == first["createdAt"] and s.all()[0]["conclusion"].startswith("다시")   # 내용은 바뀌고 처음 받아들인 때는 그대로
    assert AcceptedStore(path).all() == s.all()                           # 다시 열어도 그대로
    assert s.remove(first["conceptId"]) and not s.remove(first["conceptId"]) and s.all() == [] and s.version == 3


def test_store_survives_a_missing_or_broken_file(tmp_path):
    p = tmp_path / "a.json"
    assert AcceptedStore(p).all() == []
    p.write_text("{ not json", encoding="utf-8")
    assert AcceptedStore(p).all() == []
    p.write_text(json.dumps({"items": [{"no": "id"}, record()]}), encoding="utf-8")
    assert len(AcceptedStore(p).all()) == 1


# ---- 기댄 구간 ----

def test_sources_are_tracked_through_edits_and_lost_when_the_sentence_goes():
    r = record()
    assert [s["start"] for s in locate(r, TEXTS)] == [TEXTS["DB-WAL"].index(QA), TEXTS["OS-캐시"].index(QB)]
    moved = {**TEXTS, "DB-WAL": "# WAL (고침)\n\n새로 쓴 문단이에요.\n\n" + TEXTS["DB-WAL"].split("\n\n", 1)[1]}
    at = locate(r, moved)
    assert at[0]["start"] == moved["DB-WAL"].index(QA) != r["sources"][0]["start"]            # 밀린 위치를 새로 맞춘다
    assert locate(r, {**TEXTS, "DB-WAL": "# WAL\n\n문장을 바꿔 썼어요.\n"}) is None              # 문장이 사라졌다
    assert locate(r, {"OS-캐시": TEXTS["OS-캐시"]}) is None                                      # 노트가 사라졌다
    assert status_of(r, TEXTS)["status"] == LIVE and status_of(r, {"OS-캐시": TEXTS["OS-캐시"]})["status"] == LOST
    emoji = {"DB-WAL": "😀😀 " + TEXTS["DB-WAL"], "OS-캐시": TEXTS["OS-캐시"]}
    assert locate(r, emoji)[0]["start"] == len("😀😀 ".encode("utf-16-le")) // 2 + TEXTS["DB-WAL"].index(QA)     # 위치는 UTF-16 이다


def test_usable_length_matches_the_pairing_gate():
    assert usable_as_element(record()) and not usable_as_element(record(conclusion="너무 짧다"))
    assert usable_as_element(record(conclusion="가 " * 150)) and not usable_as_element(record(conclusion="가 " * 301))       # 결론은 노트 문장(300자)보다 길어도 된다 (600자까지)


# ---- 키워드·개념으로 합치기 ----

def test_merge_adds_a_new_keyword_with_a_generated_concept_and_leaves_the_input_alone():
    base = [extracted("WAL", "DB-WAL", QA, "ca")]
    meta = {c.conceptId: {"note": "DB-WAL", "confidence": 0.9} for k in base for c in k.concepts}
    kws, m = merge_into(base, meta, [record()], TEXTS, uid)
    assert len(base) == 1 and len(meta) == 1                                               # 들어온 것은 그대로
    new = next(k for k in kws if k.name == "이중 버퍼링")
    c = new.concepts[0]
    assert c.origin == "generated" and c.explanation == CONCLUSION and len(c.sources) == 2 and c.external[0].url == EXT[0]["url"]
    assert new.note_ids() == {uid("note", "DB-WAL"), uid("note", "OS-캐시")}                  # 사용자용 그래프에서 두 페이지를 잇는다
    assert m[c.conceptId]["origin"] == "generated" and m[c.conceptId]["notes"] == ["DB-WAL", "OS-캐시"] and m[c.conceptId]["note"].startswith("새 지식")


def test_merge_attaches_to_an_existing_keyword_with_the_same_name_or_alias():
    k = extracted("이중 버퍼링", "DB-WAL", QA, "ca")
    k = k.model_copy(update={"aliases": ["Double Buffering"]})
    for name in ("이중-버퍼링", "double buffering"):                                       # 표기가 느슨하게 같으면 같은 키워드
        kws, _ = merge_into([k], {k.concepts[0].conceptId: {"note": "DB-WAL", "confidence": 0.9}}, [record(new_keyword=name)], TEXTS, uid)
        assert len(kws) == 1 and [c.origin for c in kws[0].concepts] == ["extracted", "generated"]


def test_two_accepted_cards_with_the_same_new_keyword_share_one_keyword():
    other = record(concept_id=uid("generated", "x", "y"), conclusion="둘째 결론이에요. " + CONCLUSION)
    kws, _ = merge_into([], {}, [record(), other], TEXTS, uid)
    assert len(kws) == 1 and len(kws[0].concepts) == 2


def test_merge_skips_knowledge_whose_evidence_is_lost_and_brings_it_back_when_it_returns():
    r = record()
    gone = {"OS-캐시": TEXTS["OS-캐시"]}
    kws, m = merge_into([], {}, [r], gone, uid)
    assert kws == [] and m == {}
    kws, _ = merge_into([], {}, [r], TEXTS, uid)
    assert len(kws) == 1
