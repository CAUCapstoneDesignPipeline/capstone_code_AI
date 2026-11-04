"""⑤ 사람 판정 일치율 계산."""

from experiments.explanation_eval.human_eval import FIELDS, blind, join, score


def test_agreement_maps_system_status_to_human_labels():
    rows = [{"system": "candidate", "human": "valid"}, {"system": "candidate", "human": "invalid"},
            {"system": "hypothesis", "human": "unsure"}, {"system": "rejected", "human": "invalid"},
            {"system": "rejected", "human": ""}]
    s = score(rows)
    assert (s["labelled"], s["total"]) == (4, 5)
    assert s["agreement"] == 0.75 and s["candidate_precision"] == 0.5 and s["rejected_accuracy"] == 1.0
    assert s["confusion"]["valid→invalid"] == 1


def test_blind_sheet_hides_system_verdict():
    rows = [{"id": f"x{i:04d}", "note": "n", "linked_note": "m", "source": "A", "bridge": "B", "target": "C",
             "knowledge": "k", "evidence": "e", "human": "", "human_note": "", "system": st, "system_reasons": "r"}
            for i, st in enumerate(["candidate", "rejected", "hypothesis"])]
    sheet, key = blind(rows)
    assert all(set(r) == set(FIELDS) for r in sheet) and "system" not in FIELDS
    assert {r["id"] for r in sheet} == {r["id"] for r in key}
    assert [r["system"] for r in join(sheet, key)] == [next(k["system"] for k in key if k["id"] == r["id"]) for r in sheet]
