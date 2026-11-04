"""설명에 붙이는 쉬운 풀이 (결정적). 노트 내용이 아니라 용어와 표기 O(...) 자체의 뜻만 풀어 준다.

여기 있는 문장은 결론의 근거가 아니다 (전제는 항상 노트 사실이다). 노트가 어떤 비용을 적었는지는 바꾸지 않고, 읽는 사람이
'삽입', 'O(log n)' 같은 말을 몰라도 따라올 수 있게 돕기만 한다. 어휘 밖의 연산·역할은 풀이 없이 이름만 쓴다.
"""

from . import bigo

ROLE_GLOSS = {
    "priority_queue": "우선순위가 가장 높은(보통 키가 가장 작은) 원소부터 꺼내는 자료 구조",
    "queue": "먼저 넣은 원소를 먼저 꺼내는 줄",
    "stack": "나중에 넣은 원소를 먼저 꺼내는 더미",
    "deque": "양쪽 끝에서 넣고 뺄 수 있는 줄",
    "dictionary": "키로 값을 바로 찾는 사전",
    "set": "같은 것을 두 번 담지 않고 원소를 모아 둔 묶음",
    "disjoint_set": "서로 겹치지 않는 집합들을 관리하는 자료 구조",
    "ordered_set": "크기 순서를 유지하면서 원소를 모아 둔 묶음",
    "sorted_sequence": "크기 순서로 늘어 놓은 열",
}

OP_GLOSS = {
    "insert": "새 원소를 넣는 일",
    "extract_min": "가장 작은 키의 원소를 꺼내면서 지우는 일",
    "extract_max": "가장 큰 키의 원소를 꺼내면서 지우는 일",
    "find_min": "가장 작은 키의 원소를 지우지 않고 확인하는 일",
    "find_max": "가장 큰 키의 원소를 지우지 않고 확인하는 일",
    "decrease_key": "이미 들어 있는 원소의 키를 더 작은 값으로 바꾸는 일",
    "increase_key": "이미 들어 있는 원소의 키를 더 큰 값으로 바꾸는 일",
    "delete": "원소를 지우는 일",
    "search": "원하는 원소가 있는지 찾는 일",
    "update": "이미 있는 원소의 값을 바꾸는 일",
    "push": "맨 위에 원소를 쌓는 일",
    "pop": "맨 위 원소를 꺼내는 일",
    "peek": "맨 위 원소를 지우지 않고 보는 일",
    "enqueue": "줄의 맨 뒤에 원소를 세우는 일",
    "dequeue": "줄의 맨 앞 원소를 꺼내는 일",
    "union": "두 집합을 하나로 합치는 일",
    "find": "원소가 어느 집합에 속하는지 대표를 찾는 일",
}


def _i(x: str) -> str:
    """변수 이름 뒤 '이/가': 읽는 소리에 받침이 있는 영문자(엘·엠·엔·알)면 '이', 아니면 '가'."""
    return x + ("이" if x[-1].lower() in "lmnr" else "가")


def cost_gloss(expr: str) -> str | None:
    """O(...) 안의 식을 쉬운 말로 ('~라는 뜻이에요' 앞까지). 변수가 하나인 흔한 모양과 상수만 풀고, 나머지는 None (억지로 풀지 않는다)."""
    try:
        p = bigo.parse(expr)
    except bigo.ParseError:
        return None
    names = sorted(bigo.variables(p))
    if not names:
        return "입력이 커져도 걸리는 시간이 거의 그대로라는 뜻이에요"
    if len(names) != 1:
        return None
    x = names[0]
    for shape, text in (
            (x, f"{x}에 비례해서 늘어난다는 뜻이에요"),
            (f"log {x}", f"{_i(x)} 아주 커져도 아주 천천히만 늘어난다는 뜻이에요"),
            (f"{x} log {x}", f"{x}에 비례하는 것보다 조금 더 가파르게 늘어난다는 뜻이에요"),
            (f"{x}^2", f"{_i(x)} 두 배가 되면 대략 네 배가 된다는 뜻이에요")):
        if bigo.equivalent(p, bigo.parse(shape)):
            return text
    return None
