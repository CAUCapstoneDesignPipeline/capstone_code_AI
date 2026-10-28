"""⑤ 사람 판정 일치율 계산."""

from experiments.explanation_eval.human_eval import score


def test_agreement_maps_system_status_to_human_labels():
    rows = [{"system": "candidate", "human": "valid"}, {"system": "candidate", "human": "invalid"},
            {"system": "hypothesis", "human": "unsure"}, {"system": "rejected", "human": "invalid"},
            {"system": "rejected", "human": ""}]
    s = score(rows)
    assert (s["labelled"], s["total"]) == (4, 5)
    assert s["agreement"] == 0.75 and s["candidate_precision"] == 0.5 and s["rejected_accuracy"] == 1.0
    assert s["confusion"]["valid→invalid"] == 1
