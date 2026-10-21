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
