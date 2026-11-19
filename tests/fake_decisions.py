"""Decisions API 가짜 클라이언트 (tests/services/test_judge.py, tests/api/test_extract_decisions.py 가 함께 쓴다)."""

import re
from types import SimpleNamespace

from app.services.relations import pairs as P


class FakeDecisions:
    """질문 문구에서 후보 표기를 읽어 table 의 확률로 답한다 (없으면 0). 요청마다 기록한다."""

    def __init__(self, table=None, refuse=(), lock=None):
        self.table, self.refuse, self.lock = table or {}, set(refuse), lock
        self.calls, self.lock_free = [], []
        self.decisions = self

    def create(self, model, input, questions):
        self.calls.append((model, input, questions))
        if self.lock is not None:
            self.lock_free.append(not self.lock.locked())
        answers = []
        for q in questions:
            s = re.search(r'exact string "(.*?)" as written', q["instructions"]).group(1)
            if s in self.refuse:
                answers.append(SimpleNamespace(type="refusal"))
            elif q["type"] == "predicate":
                answers.append(SimpleNamespace(type="predicate", probability=self.table.get(s, 0.0)))
            else:
                p = self.table.get(s, 0.0)
                answers.append(SimpleNamespace(type="score", probabilities=[SimpleNamespace(label="not_concept", probability=1 - p),
                                                                           SimpleNamespace(label="mentioned_term", probability=0.0),
                                                                           SimpleNamespace(label="core_concept", probability=p)]))
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1))


class FakePairs:
    """관계 판정용 가짜 Decisions 클라이언트: 질문 문구에서 두 표기를 읽어 table[(a, b)] 의 선택지 확률로 답한다 (없으면 default, 기본은 '관계 없음')."""

    def __init__(self, table=None, default=None, refuse=()):
        self.table, self.default, self.refuse = table or {}, default or {"none": 1.0}, set(refuse)
        self.calls = []
        self.decisions = self

    def create(self, model, input, questions):
        self.calls.append((model, input, questions))
        answers = []
        for q in questions:
            a, b = re.search(r'two terms "(.*?)" \(A\) and "(.*?)" \(B\)', q["instructions"]).groups()
            if (a, b) in self.refuse:
                answers.append(SimpleNamespace(type="refusal"))
                continue
            probs = self.table.get((a, b), self.default)
            answers.append(SimpleNamespace(type="choice", probabilities=[SimpleNamespace(value=v, probability=probs.get(v, 0.0)) for v in (c["value"] for c in q["choices"])]))
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1))


def stump(feature, low, high):
    return {"feature": [feature, 0, 0], "threshold": [0.5, 0.0, 0.0], "left": [1, -1, -1], "right": [2, -1, -1], "value": [0.0, low, high]}


def tiny_model(threshold=0.5):
    """관계 확률 = 'Decisions 가 관계 있다고 본 정도' 가 0.5 를 넘는가, 유형 = is_a_ab / is_a_ba 중 확률이 높은 쪽."""
    d = P.N_STRUCT
    forest = lambda tree: {"baseline": 0.0, "trees": [tree], "max_depth": 1}
    return P.PairModel({"id": "pairs-test", "features": P.FEATURES, "classes": P.CLASSES, "threshold": threshold, "pair": forest(stump(d, -4.0, 4.0)),
                        "type": {"classes": [0, 1], "per_class": [forest(stump(d + 1, 0.0, 2.0)), forest(stump(d + 2, 0.0, 2.0))]}})
