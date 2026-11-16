"""학생 + Decisions 판정 (`extract-student-decisions-1`): 학생이 낸 후보를 OpenAI Decisions API 가 한꺼번에 판정하고, 학생 확률과 섞어 개념을 고른다.

  1. 후보: 학생과 같은 토큰 n-gram (후보 생성 2판, 이름 속 기호를 잇는다) 가운데 학생 확률이 MIN_P 이상인 것을 확률 순으로 MAX_CANDIDATES 개까지.
  2. 판정: 노트 원문과 후보마다의 질문 두 개 (3단계 score 질문 · 표시 기준 predicate 질문) 를 각각 한 번의 요청으로 보낸다. 두 요청은 동시에 나가서 약 1초 걸린다.
     노트는 요청마다 한 번만 읽히고 후보 수에 지연이 거의 늘지 않는다. 답은 후보마다의 확률이다.
  3. 섞기: 확률 = (1 - WEIGHT) × 학생 + WEIGHT × (두 답의 평균). THRESHOLD 이상인 것을 고르고 더 긴 채택 후보 안에서만 나오는 짧은 후보는 뺀다 (학생과 같은 정리).

측정과 한계: docs/eval/concept-student.md. 질문 문구·유형·섞는 비율·임계값은 profile 의 일부라 바꾸면 새 profile id 를 만든다.
노트 원문이 OpenAI 로 나간다. 답이 오지 못하면 (한도·시간 초과·제공자 오류) LLMError 를 올려 계약 오류로 답하고, 호출한 쪽은 학생 결과를 그대로 쓴다.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np

from ..llm import LLMError
from .schema import Mention, Method, Span
from .student import MAX_CONCEPTS, MAX_MENTIONS, rows, suppress_nested
from .markdown import Utf16Index

PROMPT_VERSION = "decisions-1"
DEFAULT_MODEL = "gpt-6-luna"
WEIGHT, THRESHOLD, MIN_P, MAX_CANDIDATES = 0.7, 0.70, 0.01, 400

# 질문 문구는 약간만 바뀌어도 확률이 크게 달라진다 (같은 후보가 0.87 → 0.11). 문구를 바꾸면 PROMPT_VERSION 과 profile id 를 함께 올린다.
CRITERIA = (
    'Judge the exact string "{s}" as written in this study note. It is TRUE only if it is a CONCEPT a student would want as a node in a knowledge graph of this note: '
    'a field-specific technical term, or a named algorithm, protocol, data structure, technique, theorem, formula, model, architecture, dataset or metric, '
    'that the note explains, relates to other concepts, or mentions in passing as a standard term of its field (examples of TRUE: "Raft", "ACID", "backpropagation", '
    '"Page Table", "정규화", "Bellman 방정식", "Q-learning"). It is FALSE for: general words usable in any field (method, result, data, system, user, 방법, 문제, 성능); '
    'section or heading frames (Abstract, Step 1, 핵심 아이디어, 문제 상황); example values, variables, file names, page numbers (age = 30, x1, TID 10, p.20); '
    'people, company, venue or lecture names (Jack Edmonds, Google); and sentence fragments or strings that cut a longer term short or add extra words to it '
    '(a term as written in the note is the complete one, so "Interrupt vs" or half of a two-word term is FALSE).')
CHOICE = 'Classify the exact string "{s}" as written in this study note.'
LEVELS = [{"label": "not_concept", "description": "generic word, heading frame, example value, person/venue name, fragment, or a mis-bounded piece of a longer term"},
          {"label": "mentioned_term", "description": "a standard field-specific term that only appears in passing"},
          {"label": "core_concept", "description": "a field-specific concept the note explains, defines, or relates to other concepts"}]


def _score_question(s):
    return {"type": "score", "instructions": CHOICE.format(s=s), "levels": LEVELS}


def _criteria_question(s):
    return {"type": "predicate", "instructions": CRITERIA.format(s=s)}


def _score_probability(answer):
    return sum(x.probability for x in answer.probabilities if x.label != "not_concept")


def _criteria_probability(answer):
    return answer.probability


QUESTIONS = (("score", _score_question, _score_probability), ("criteria", _criteria_question, _criteria_probability))


def create_answers(client, model: str, text: str, questions: list[dict]) -> list:
    """Decisions 요청 하나를 보내 질문마다의 답을 돌려준다. 제공자 오류는 LLMError 로 바꾼다 (관계 판정(relations/pairs.py)도 쓴다)."""
    import openai
    try:
        d = client.decisions.create(model=model, input=text, questions=questions)
    except openai.RateLimitError as e:
        raise LLMError("rate_limit", str(e)) from e
    except openai.APITimeoutError as e:
        raise LLMError("timeout", str(e)) from e
    except (openai.AuthenticationError, openai.PermissionDeniedError) as e:
        raise LLMError("auth", str(e)) from e
    except openai.APIError as e:
        raise LLMError("provider", str(e)) from e
    if len(d.answers) != len(questions):
        raise LLMError("invalid_output", f"{len(d.answers)} answers for {len(questions)} questions")
    return d.answers


class DecisionsJudge:
    """후보 표기마다 '개념인가' 확률을 돌려준다. client 는 openai.OpenAI 처럼 decisions.create 가 있는 것."""

    provider = "openai"

    def __init__(self, client, model: str = DEFAULT_MODEL):
        self.client, self.model = client, model

    def _ask(self, question, reduce, text: str, surfaces: list[str]) -> np.ndarray:
        answers = create_answers(self.client, self.model, text, [question(s) for s in surfaces])
        # 답을 거절하거나 확률이 오지 않은 후보는 개념이 아니라고 본다
        return np.array([0.0 if a.type == "refusal" else float(reduce(a)) for a in answers])

    def probabilities(self, text: str, surfaces: list[str]) -> np.ndarray:
        """두 질문의 확률 평균. 질문마다 요청을 하나씩 동시에 보낸다."""
        if not surfaces:
            return np.zeros(0)
        with ThreadPoolExecutor(len(QUESTIONS)) as pool:
            futures = [pool.submit(self._ask, q, r, text, surfaces) for _, q, r in QUESTIONS]
            return np.mean([f.result() for f in futures], axis=0)


class StudentJudge:
    """학생(후보·확률) + 판정 (Decisions) 을 섞어 개념을 고른다. select 의 반환은 Student.select 와 같은 모양이다."""

    provider = "openai"

    def __init__(self, student, judge: DecisionsJudge):
        self.student, self.judge = student, judge

    @property
    def model(self) -> str:
        return self.judge.model

    @property
    def version(self) -> str:
        return f"{self.student.version}+{PROMPT_VERSION}"

    def select(self, extractor, note_id: str, note_version: int, text: str, lock=None) -> dict[str, list[Mention]]:
        """개념 key → 언급들 (섞은 확률이 높은 순, 같으면 key 순; 점수 = 2 × 확률). lock 은 형태소 분석기용 — 후보를 만드는 동안만 잡고 네트워크 호출 중에는 풀어 둔다."""
        with lock or nullcontext():
            text, rows_ = rows(extractor, text, symbols=True)
            if not rows_:
                return {}
            p = np.round(self.student.probabilities(np.array([r["x"] for r in rows_], dtype=np.float64)), 6)
        cand = [int(i) for i in np.argsort(-p, kind="stable") if p[i] >= MIN_P][:MAX_CANDIDATES]
        if not cand:
            return {}
        judged = self.judge.probabilities(text, [rows_[i]["surface"] for i in cand])
        q = np.zeros(len(rows_))
        q[cand] = np.round((1 - WEIGHT) * p[cand] + WEIGHT * judged, 4)
        idx = sorted(suppress_nested(rows_, q, THRESHOLD), key=lambda i: (-q[i], rows_[i]["key"]))[:MAX_CONCEPTS]
        u16 = Utf16Index(text)
        out = {}
        for i in idx:
            r = rows_[i]
            spans = sorted(set(r["spans"]))[:MAX_MENTIONS]
            out[r["key"]] = [Mention(span=Span(note_id, note_version, u16(s), u16(e)), surface=text[s:e], key=r["key"], method=Method.LLM,
                                     method_version=f"{self.version}+{self.model}", score=round(2.0 * float(q[i]), 4)) for s, e in spans]
        return out
