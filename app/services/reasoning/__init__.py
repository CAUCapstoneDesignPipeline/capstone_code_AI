"""노트 사이 추론 단계 (병합 판단·매개 검증·간선 검증·설명 생성·설명 검증). 설명 생성만 LLM, 나머지는 결정적이다.

  identity     두 occurrence 가 같은 개념인가          → /merge-decisions, /verifications/bridges
  entailment   간선 근거가 그 관계를 말하는가          → /verifications/edges
  explain_llm  A-B-C 경로 설명 = 새 지식 제안 (LLM)   → /explanations
  explain      설명 검증 (결정적)                       → /verifications/explanations
  explanation_llm 설명 검증의 LLM 판정 (근거 뒷받침·구체성·새로움) → /verifications/explanations (hybrid)
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

_SERVICES = Path(__file__).parent.parent


def code_revision(*files: str) -> str:
    """방법 코드의 hash. 코드를 바꾸면 modelRevision 이 바뀌어 BE 가 다른 결과임을 안다."""
    h = hashlib.sha256()
    for f in files:
        h.update((_SERVICES / f).read_bytes())
    return h.hexdigest()[:12]


@dataclass(frozen=True)
class ReasoningProfile:
    """불변 추론 profile. 검증 profile 은 생성 profile 과 model 또는 prompt 가 달라야 한다 (계약 §5)."""
    id: str
    operation: str              # merge | bridges | edges | explain | explanation | links
    model: str
    prompt: str
    files: tuple[str, ...]
    requires: tuple[str, ...] = ()   # 이 profile 이 쓰는 서버 구성 요소 (app.state 속성): extractor, typer, explainer, edge_judge, explanation_judge

    @property
    def revision(self) -> str:
        return code_revision(*self.files)

    def ready(self, state) -> bool:
        return all(getattr(state, r, None) is not None for r in self.requires)


_ID = ("reasoning/identity.py", "reasoning/text.py",       # 문맥은 추출기의 명사구 후보라 그 코드도 넣는다
       "concepts/extract.py", "concepts/markdown.py", "concepts/config.py")
_EX = ("reasoning/explain.py", "reasoning/text.py")
_LINKS = ("links/corpus.py", "links/linker.py", "links/context.py")

REASONING_PROFILES = {p.id: p for p in (
    ReasoningProfile("merge-det-1", "merge", "rules-identity", "merge-rules-1", _ID, ("extractor",)),
    ReasoningProfile("bridges-det-1", "bridges", "rules-identity", "bridges-rules-1", _ID, ("extractor",)),
    ReasoningProfile("edges-det-1", "edges", "rules-entailment-cues", "edges-cues-1",
                     ("reasoning/entailment.py", "reasoning/text.py")),
    # LLM 이 근거를 읽고 간선을 판정한다. 관계 판정(relation-typing)과 prompt 가 달라 계약 §5 독립성을 지킨다
    ReasoningProfile("edges-llm-1", "edges", "llm", "edges-llm-1", ("reasoning/entailment_llm.py",), ("edge_judge",)),
    # 새 지식 제안은 항상 LLM 이 쓴다 (해요체, 모든 사실은 근거 인용). 경로·근거·합성 규칙은 입력으로 고정, 응답은 §3.9 로 코드가 다시 검사한다
    ReasoningProfile("explain-llm-8", "explain", "llm", "explain-llm-8",
                     ("reasoning/explain_llm.py", "reasoning/explain.py", "reasoning/text.py"), ("explainer",)),
    ReasoningProfile("explanation-det-2", "explanation", "rules-explanation-check", "explanation-rules-1", _EX),
    # det-2 + 문맥 근거만 인용하는 보조 전제를 받는다
    ReasoningProfile("explanation-det-3", "explanation", "rules-explanation-check", "explanation-rules-2", _EX),
    # det-3 + LLM 이 주장의 근거 뒷받침과 결론의 구체성·새로움을 본다. 구체적이지 않거나 새롭지 않은 결론은 fail
    ReasoningProfile("explanation-hybrid-1", "explanation", "llm", "explanation-llm-1",
                     (*_EX, "reasoning/explanation_llm.py"), ("explanation_judge",)),
    # 노트 사이 연결·경로 후보. 계약 v0.1.0 에는 없는 AI 쪽 제안 operation (경로·순위는 계약상 BE 책임)
    ReasoningProfile("links-det-1", "links", "kiwi-tfidf-adamic-adar", "links-det-1", _LINKS, ("extractor",)),
    # 같은 방법에 관계를 LLM 으로 보탠다 (extract-hybrid-2 의 관계와 같다). 규칙만으로는 관계가 적어 경로가 거의 없다
    ReasoningProfile("links-hybrid-1", "links", "kiwi-tfidf-adamic-adar+llm-typing", "links-hybrid-1",
                     (*_LINKS, "relations/typer.py"), ("extractor", "typer")),
)}
