"""생성된 새 지식을 보여 주기 전에 한 번 더 확인한다 (웹 검색 없음, 모델 한 번).

생성 단계의 코드 검사(URL 이 검색 결과에 있는가, 인용이 원문 그대로인가, 노트와 같은 말이 아닌가)가 못 막는 실패가 있다. 실제 노트 6건을 읽어 보니:
  - 결론이 A·B 와 다른 주제로 넘어갔다 (TLS Master Secret × 개인키 → SSH 세션 식별자). 출처도 전부 그 다른 주제였다.
  - 출처가 검색 결과에는 있지만 결론과 무관했다 (HTTP 무상태성 결론에 모바일 기기 콘텐츠 가져오기 특허, DBSCAN 결론에 양자 알고리즘 논문).
  - 노트 문장이 틀렸는데 모델이 조용히 고쳐서 결론을 썼다.
그래서 노트 두 문장과 결론, 출처 목록(제목·URL·모델이 옮긴 내용)을 주고 다음을 묻는다.
  onTopic   결론이 A 와 B 가 다루는 주제 안에 있다 (다른 주제로 갈아타지 않았다)
  bothUsed  결론이 A 와 B 를 둘 다 쓴다 (한쪽만으로도 말할 수 있는 내용이 아니다)
  newInfo   A·B 에 없는 내용이 있다
  reach     결론이 노트 문장에서 얼마나 멀리 갔는가 — derived(문장에서 따라 나온다) · extended(문장이 다루는 개념에 바깥 사실을 더했다) · hop(중심 주제가 문장이 다루는 개념과 다른 개념·기법·정리다).
            막지 않고 보여 준다: 멀리 간 카드도 쓸 만할 수 있다 (실제로 읽어 보니 hop 인 카드 가운데 쓸 만한 것이 많았다)
  selfContained  결론이 입력 문장을 'A'·'B' 같은 라벨로 가리키지 않아 혼자 읽힌다 (실제 결과에서 "B에 나온 세 가지 환경은" 이 나왔다)
  correct   결론에 틀렸다고 확신하는 주장이 없다 (단순화·예외 생략은 감점하지 않는다. 처음에는 "확실하지 않으면 false" 로 했더니 SSH 포트 포워딩처럼 쓸 만한 카드를 막았다)
  noteError A 나 B 에 틀린 내용이 있고 결론이 그것을 조용히 바로잡았다
  sources   출처마다 결론을 뒷받침하는가 (yes · partly · no) — 제목과 URL 만 보고 판단한다
코드가 정한다: onTopic·bothUsed·newInfo·selfContained 중 하나라도 false 이면 보류(held) — 이 셋은 글만 보고 확인되는 구조 검사다.
correct 와 noteError 는 막지 않고 경고로 보인다. 사실 여부는 이 확인이 새로 검증하지 못한다 (모델 판단이라 틀린다: Sonnet 이 "NP-중간 문제가 없다 ⇔ P=NP" 를 틀렸다고 해서
옳은 카드를 세 번 모두 반려했다). 그래서 사실 의심은 이유와 함께 보여 주고 사람이 판단하게 한다. noteError 도 결론은 쓸모 있고 노트가 부정확하다는 것 자체가 학생에게 지식이다.
`no` 인 출처는 버리고 남은 출처가 없으면 보류.
출처의 "모델이 옮긴 내용(quote)" 은 검증기에 주지 않는다: 생성 모델이 직접 쓴 글이라 무관한 페이지(특허)도 그럴듯하게 요약해 두면 검증기가 속았다 (제목·URL 만 주면 걸러진다).
이 확인은 같은 모델 계열의 판단이라 틀릴 수 있다 (사실 확인을 새로 하지 않는다). 걸러낸 이유는 화면에 그대로 보인다.
"""

import hashlib
import json
from pathlib import Path

from app.services.llm import LLMError

PROMPT_VERSION = "knowledgecheck-llm-3"      # 2: selfContained (결론이 입력 문장을 'A'·'B' 라벨로 가리키지 않는가) 추가. 3: reach (결론이 노트 문장에서 얼마나 멀리 갔는가) 추가 · 받아들인 지식이 낀 쪽에는 원래 노트 문장(noteSentences)을 준다
SUPPORT = ("yes", "partly", "no")
REACH = ("derived", "extended", "hop")

SYSTEM = """너는 학생의 공부 노트에서 뽑은 두 문장(A, B)과 웹 자료로 만든 "새 지식 결론"을 보여 주기 전에 확인하는 검수자다. 결론을 새로 쓰지 않는다. 입력 JSON 을 보고 아래를 판정한다.
- onTopic: 결론이 A 와 B 가 다루는 주제 안에 있다. 다른 기술·다른 분야로 넘어가 비유하거나 갈아탔으면 false.
- bothUsed: 결론이 A 와 B 의 내용을 둘 다 실제로 쓴다. 한쪽 문장만으로 말할 수 있으면 false.
- newInfo: 결론에 A·B 에 쓰여 있지 않은 구체적인 내용이 있다. 두 문장을 바꿔 쓴 것뿐이면 false.
- selfContained: 결론이 입력 문장을 "A", "B", "A 문장", "위 문장" 같은 라벨로 가리키지 않고 혼자 읽힌다. 내용 안의 이름(서버 A, B-트리, A* 알고리즘, 집합 A 처럼 결론이 다루는 대상의 이름)이면 라벨이 아니므로 true 다. "노트의 그림", "노트에서 말한 …" 처럼 학생이 아는 자기 노트를 가리키는 말도 라벨이 아니라 true 다. 학생이 무엇인지 알 수 없는 입력의 라벨(A, B, 위 문장)일 때만 false.
- reach: 결론이 학생의 노트 문장에서 얼마나 멀리 갔는가. derived = 결론의 내용이 노트 문장들에서 따라 나온다 (일반 상식을 조금 보태도). extended = 노트 문장이 다루는 바로 그 개념들에 대해 바깥 사실을 더했다. hop = 결론의 중심 주제가 노트 문장이 다루는 개념과 다른 개념·기법·시스템·정리다 (관련은 있어도 다른 주제). 입력의 "noteSentences" 가 있으면 그것이 학생이 쓴 원래 노트 문장이다: text 는 앞서 바깥 자료로 넓힌 글이므로, 이 경우에는 noteSentences 를 기준으로 본다.
- correct: 결론에 틀린 주장이 없다. 틀렸다고 확신하는 주장이 있을 때만 false 다. 쉬운 말로 풀어 쓴 것, 예외를 다 적지 않은 단순화, 학부 수업 수준에서 통용되는 설명은 감점하지 않는다. 모르겠으면 true 다 (노트 문장이 아니라 결론에 대한 판정이다).
- noteError: A 나 B 자체에 틀리거나 부정확한 내용이 있는데 결론이 그것을 말없이 바로잡아 썼다 (예: 노트는 "클라이언트가 개인키로 복호화" 라고 했는데 결론은 "서명" 이라고 고쳐 썼다). 그런 일이 없으면 false.
- sources: 출처마다 id(S1..) 로 결론을 뒷받침하는지 yes(페이지의 주제가 결론이 다루는 개념이다) · partly(주제는 같은 분야이고 일부만 다룬다) · no(결론의 개념과 무관하거나, 낱말만 겹치거나, 다른 분야다). 제목과 URL 만 보고 판단한다. 낱말 하나가 겹쳐도 주제가 다르면 no.
- reason: 하나라도 문제가 있으면 무엇이 문제인지 한국어 한 문장. 문제가 없으면 빈 문자열.
엄격하게 본다. 입력은 {"A":{keyword,text}, "B":{keyword,text}, "newKeyword", "conclusion", "sources":[{id,title,url}]} 이다."""

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["onTopic", "bothUsed", "newInfo", "selfContained", "reach", "correct", "noteError", "sources", "reason"],
    "properties": {
        "onTopic": {"type": "boolean"}, "bothUsed": {"type": "boolean"}, "newInfo": {"type": "boolean"}, "selfContained": {"type": "boolean"}, "reach": {"type": "string", "enum": list(REACH)},
        "correct": {"type": "boolean"}, "noteError": {"type": "boolean"},
        "sources": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["id", "support"],
            "properties": {"id": {"type": "string"}, "support": {"type": "string", "enum": list(SUPPORT)}}}},
        "reason": {"type": "string"},
    },
}

PROBLEMS = {
    "onTopic": "결론이 두 노트 문장의 주제를 벗어났다",
    "bothUsed": "결론이 두 문장 중 한쪽만으로도 말할 수 있는 내용이다",
    "newInfo": "결론에 새로운 내용이 없다",
    "selfContained": "결론이 입력 문장을 'A'·'B' 같은 라벨로 가리켜 혼자 읽히지 않는다",
}


class KnowledgeChecker:
    def __init__(self, llm, cache_dir: Path | None = None):
        self.llm, self.cache_dir = llm, cache_dir
        self.prompt_version = PROMPT_VERSION

    @property
    def model(self) -> str:
        return self.llm.model

    def check(self, a: dict, b: dict, new_keyword: str, conclusion: str, external: list[dict]) -> dict:
        """external: [{url, title, quote}] (코드 검사를 통과한 것). 반환: {ok, problems[str], warnings[str], external(뒷받침하는 것만), dropped, reason, flags, reach}."""
        sources = [{"id": f"S{i}", "title": e.get("title", ""), "url": e["url"]} for i, e in enumerate(external, 1)]          # quote 는 주지 않는다 (위 설명)
        user = json.dumps({"A": self._side(a), "B": self._side(b), "newKeyword": new_keyword, "conclusion": conclusion, "sources": sources}, ensure_ascii=False)
        key = hashlib.sha256(json.dumps([self.llm.provider, self.llm.model, PROMPT_VERSION, user], ensure_ascii=False).encode()).hexdigest()
        hit = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if hit is not None and hit.exists():
            data = json.loads(hit.read_text(encoding="utf-8"))
        else:
            data = self.llm.complete_json(SYSTEM, user, SCHEMA, name="knowledge_check").data
            self._decide(data, external)                                                  # 형식이 틀린 답은 저장하지 않는다 (LLMError)
            if hit is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                hit.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return self._decide(data, external)

    @staticmethod
    def _side(x: dict) -> dict:
        """검수자에게 줄 요소. 받아들인 새 지식(generated)은 결론이 바깥 자료로 넓힌 글이라, 그것이 기댄 원래 노트 문장도 준다 (노트에서 멀어진 정도는 이것을 기준으로 본다)."""
        out = {"keyword": x["keyword"], "text": x["text"]}
        if x.get("origin") == "generated" and x.get("sources"):
            out["noteSentences"] = [r["quote"][:300] for r in x["sources"][:6]]
        return out

    @staticmethod
    def _decide(r: dict, external: list[dict]) -> dict:
        try:
            flags = {k: bool(r[k]) for k in ("onTopic", "bothUsed", "newInfo", "selfContained", "correct", "noteError")}
            support = {s["id"]: s["support"] for s in r["sources"] if s.get("support") in SUPPORT}
            reason = str(r["reason"]).strip()
            reach = r["reach"]
            if reach not in REACH:
                raise KeyError(f"reach={reach!r}")
        except (KeyError, TypeError) as e:
            raise LLMError("invalid_output", str(e))
        problems = [msg for k, msg in PROBLEMS.items() if not flags[k]]
        warnings = (["노트 문장이 부정확할 수 있는데 결론이 말없이 바로잡아 썼다"] if flags["noteError"] else []) + \
                   (["결론에 틀린 주장이 있을 수 있다고 봤다 (사실 확인은 하지 않았다)"] if not flags["correct"] else [])
        # 출처: 모델이 id 를 빠뜨리면 뒷받침한다고 보지 않는다 (확인하지 못한 출처를 남기지 않는다)
        kept = [e for i, e in enumerate(external, 1) if support.get(f"S{i}") in ("yes", "partly")]
        dropped = [e for i, e in enumerate(external, 1) if support.get(f"S{i}") not in ("yes", "partly")]
        if not kept and external:
            problems.append("결론을 뒷받침하는 외부 출처가 없다")
        return {"ok": not problems, "problems": problems, "warnings": warnings, "external": kept, "dropped": dropped, "reason": reason, "flags": flags, "reach": reach}
