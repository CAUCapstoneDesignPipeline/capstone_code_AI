from app.services.links.pair_screen import PairScreen
from app.services.llm import LLMResult


class Fake:
    provider, model = "anthropic", "fake"

    def __init__(self, drop=()):
        self.drop, self.calls = set(drop), []

    def complete_json(self, system, user, schema, name):
        ids = [l[4:] for l in user.splitlines() if l.startswith("id: ")]
        self.calls.append(ids)
        return LLMResult({"results": [{"id": i, "label": "useful" if i == "p1" else "vague", "conclusion": "결론" if i == "p1" else ""}
                                      for i in ids if i not in self.drop]}, self.provider, self.model)


def pair(i):
    return {"id": f"p{i}", "a": {"keyword": f"A{i}", "text": f"A 문장 {i}"}, "b": {"keyword": f"B{i}", "text": f"B 문장 {i}"}}


def test_labels_and_cache(tmp_path):
    fake = Fake()
    s = PairScreen(fake, tmp_path)
    r = s.screen([pair(1), pair(2)])
    assert (r["p1"]["label"], r["p1"]["conclusion"], r["p2"]["label"]) == ("useful", "결론", "vague")
    assert s.screen([pair(1), pair(2)])["p1"]["cached"] is True and len(fake.calls) == 1


def test_a_pair_the_model_skips_is_asked_once_more_in_a_small_batch_then_unsure():
    fake = Fake(drop=["p2"])
    r = PairScreen(fake, None).screen([pair(1), pair(2)])
    assert r["p2"]["label"] == "unsure" and fake.calls == [["p1", "p2"], ["p2"]]
