"""참조 클라이언트의 경로 고르기와 요청 묶기 (호출 없음)."""

from app.client import MAX_DOCS, MAX_EVIDENCE, _batches, select_paths


def path(note, src, rule):
    return {"steps": [{"noteId": "a", "fromLabel": src, "typeCode": "is_a", "toLabel": "B"},
                      {"noteId": note, "fromLabel": "B", "typeCode": "is_a", "toLabel": "C"}],
            "proposedTypeCode": "is_a" if rule else None}


def test_select_paths_rule_first_and_spread_across_notes():
    links = [{"noteId": "n1", "paths": [path("n1", f"x{i}", False) for i in range(5)] + [path("n1", "r1", True)]},
             {"noteId": "n2", "paths": [path("n2", "y0", False), path("n2", "r2", True)]}]
    got = select_paths(links, 4)
    assert [p["steps"][0]["fromLabel"] for _, p in got] == ["r1", "r2", "x0", "y0"]   # 규칙 먼저, 노트를 번갈아
    assert len(select_paths(links + links, 50)) == 8                                    # 같은 경로는 한 번


def test_batches_respect_contract_limits():
    items = [{"d": [{"noteId": f"n{i}"}], "e": [{"evidenceId": f"e{i}-{j}"} for j in range(3)]} for i in range(30)]
    out = _batches(items, lambda x: x["d"], lambda x: x["e"], 20)
    assert sum(len(g) for g, _, _ in out) == 30
    assert all(len(d) <= MAX_DOCS and len(e) <= MAX_EVIDENCE and len(g) <= 20 for g, d, e in out)


def prep(src, codes, target="C", bridge="B", rule=True, rank=0, at=0, note2="n2"):
    """_prepare 로 만든 실제 경로 재료. at: 단계 근거 위치 (다른 경로의 같은 관계는 다른 문장)."""
    from app.client import _prepare, snapshot
    body = "가" * 200
    notes = {n: snapshot(n, n, body) for n in ("n1", note2)}
    span = lambda n, s: {"noteId": n, "startOffset": s, "endOffset": s + 3, "quote": "가가가"}
    path = {"source": src, "bridge": bridge, "target": target, "proposedTypeCode": "is_a" if rule else None,
            "steps": [{"stepOrder": 1, "noteId": "n1", "fromLabel": src, "toLabel": bridge, "typeCode": codes[0],
                       "evidence": [span("n1", at)], "context": [span("n1", 100 + at)]},
                      {"stepOrder": 2, "noteId": note2, "fromLabel": bridge, "toLabel": target, "typeCode": codes[1],
                       "evidence": [span(note2, 50)], "context": []}]}
    return _prepare(notes["n1"], notes, path, {}, note2, rank)


def test_pick_paths_merges_same_a_c_and_keeps_evidence():
    from app.client import pick_paths
    a = prep("Add", ("example_of", "is_a"), target="트랜잭션 기능", at=0)
    b = prep("add", ("is_a", "is_a"), target="트랜잭션 기능", rule=False, at=10, rank=1)   # 같은 A–C, 다른 문장
    out, merged = pick_paths([b, a], 6)
    assert merged == 1 and out == [a]                                   # 규칙 있는 쪽을 남긴다
    assert len(a.path_body["steps"][0]["evidenceIds"]) == 2 and len(a.path["steps"][0]["evidence"]) == 2
    assert len(a.evidence) == 3 and len(a.context) == 2                 # 근거와 문맥을 모은다


def test_pick_paths_prefers_non_taxonomy_and_spreads_tails():
    from app.client import pick_paths
    tax = prep("Add", ("is_a", "is_a"), target="트랜잭션 기능", bridge="EI")
    tax2 = prep("Change", ("is_a", "is_a"), target="트랜잭션 기능", bridge="EI", rank=1)
    other = prep("Add", ("is_a", "requires"), target="ILF", bridge="EI", rank=2)
    out, _ = pick_paths([tax, tax2, other], 2)
    assert out == [other, tax2]           # 분류 계층만 올라가는 경로는 뒤로, 이미 쓴 Add→EI·EI→트랜잭션 기능은 나중에
    out, _ = pick_paths([tax, tax2, other], 3)
    assert out == [other, tax, tax2]                   # 자리가 남으면 채운다
