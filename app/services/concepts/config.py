"""개념 추출 설정. 값이 바뀌면 version() 이 바뀌어 캐시와 결과가 새로 계산된다."""

import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# 여러 노트에 흔히 나오지만 개념이 아닌 말. 1주차 노트 141개에서 노트 70개 이상에 나온 명사와,
# 마크다운 제목에 틀처럼 쓰이는 말을 모았다. 바꾸면 version 이 바뀐다.
STOPWORDS = frozenset("""
사용 다음 문제 필요 가능 전체 방법 핵심 결과 경우 방식 설명 이유 특징 장점 단점 차이 비교 개념 종류 내용 부분 정도
예 예시 예제 정리 요약 참고 출처 정의 질문 답 답변 풀이 해설 목차 목표 개요 소개 배경 결론 메모 링크 주의 팁
것 수 등 때 중 후 전 위 아래 이상 이하 각 모든 여러 다른 같은 자체 기본 일반 실제 관련 대한 통해 위해
상태 과정 단계 구조 기능 역할 처리 작업 정보 데이터 값 시간 경우 의미 형태 대상 기준 범위 요소 단위
아이디어 관찰 계산 해결 해결책 문제점 직관 목적 구분 유형 조건 한계 증명 요구 정의 개요 복습 퀴즈 연습 실습 과제
구성 일종 일부 사례 하위 상위
이후 이전 가능성 안녕 동기
issue issues problem problems case cases question tip
step part chapter ch lecture section level page note notes summary example examples overview intro introduction
the a an of to for is are was were be in on at by as and or not with from this that these those it its into
""".split())
# 노트 여러 개에서 제목 틀로 반복된 말 (1주차 노트에서 노트 5개 이상에 개념으로 뽑힌 것 중 개념이 아닌 것).
# 구(phrase)의 모든 낱말이 불용어여도 버린다 ("핵심 아이디어", "문제 정의", "step 1").
# 관계를 나타내는 말(구성 요소·일종·하위 …)도 그것만으로는 개념이 아니다. "X는 Y의 구성 요소다" 에서 '구성 요소' 를
# 개념으로 뽑으면 'X is_a 구성 요소' 같은 틀린 관계가 생긴다.


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


@dataclass(frozen=True)
class ExtractConfig:
    # 구조 가중치 (design §4①): 제목 > 굵은 글씨 > 표 첫 열
    heading_weight: float = 1.0
    bold_weight: float = 0.5
    table_weight: float = 0.3
    # 선택 기준: tfidf_norm(노트 안 최댓값으로 나눈 TF-IDF, 0~1) + 구조 가중치 >= threshold
    threshold: float = 0.5
    max_phrase_tokens: int = 4      # 복합 명사구 최대 명사 수
    min_surface_chars: int = 2
    max_surface_chars: int = 40
    stopwords: frozenset[str] = field(default=STOPWORDS)

    def version(self) -> str:
        """설정·불용어·형태소 분석기 버전·추출 코드의 해시. 결과마다 기록하고 캐시 키에 넣는다.
        추출 코드가 바뀌면 설정이 같아도 버전이 바뀌어, 예전 캐시를 잘못 재사용하지 않는다."""
        payload = asdict(self)
        payload["stopwords"] = sorted(self.stopwords)
        payload["kiwipiepy"] = _pkg_version("kiwipiepy")
        payload["kiwipiepy_model"] = _pkg_version("kiwipiepy_model")
        here = Path(__file__).parent
        payload["code"] = hashlib.sha256(b"".join(
            (here / f).read_bytes() for f in ("extract.py", "markdown.py"))).hexdigest()
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
