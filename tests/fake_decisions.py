"""Decisions API 가짜 클라이언트 (tests/services/test_judge.py, tests/api/test_extract_decisions.py 가 함께 쓴다)."""

import re
from types import SimpleNamespace


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
