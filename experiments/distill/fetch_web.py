"""여러 분야의 공개 글을 모은다 (위키백과, CC BY-SA). 학생 모델의 분야 일반화를 재기 위한 학습·평가 자료다.

  uv run python experiments/distill/fetch_web.py [--per-lang 4] [--max-chars 9000]

분야마다 영어·한국어 글 `--per-lang` 개 (없는 제목은 건너뛴다). 위키 제목 표기(== 절 ==)를 마크다운 제목으로 바꾸고, 참고 문헌 같은 끝 절은 뗀다.
data/private/notes-web/<분야>__<언어>__<제목>.md 에 쓴다 (저장소에 넣지 않는다).
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/private/notes-web"
UA = "concept-extraction-research/1.0"
DOMAINS = {
    "physics": (["Quantum mechanics", "Special relativity", "Thermodynamics", "Electromagnetism", "Fluid dynamics", "Nuclear fission"],
                ["양자역학", "특수 상대성이론", "열역학", "전자기학", "유체역학", "핵분열"]),
    "biology": (["Photosynthesis", "DNA replication", "Natural selection", "Mitosis", "CRISPR", "Immune system"],
                ["광합성", "DNA 복제", "자연선택", "체세포 분열", "면역계", "크리스퍼"]),
    "chemistry": (["Chemical bond", "Organic chemistry", "Redox", "Acid–base reaction", "Catalysis", "Periodic table"],
                  ["화학 결합", "유기화학", "산화·환원 반응", "산-염기 반응", "촉매", "주기율표"]),
    "math_stats": (["Linear algebra", "Eigenvalues and eigenvectors", "Bayesian inference", "Statistical hypothesis test", "Fourier transform", "Markov chain"],
                   ["선형대수학", "고유값과 고유벡터", "베이즈 추론", "가설 검정", "푸리에 변환", "마르코프 연쇄"]),
    "economics": (["Supply and demand", "Inflation", "Game theory", "Comparative advantage", "Monetary policy", "Gross domestic product"],
                  ["수요와 공급", "인플레이션", "게임 이론", "비교우위", "통화 정책", "국내총생산"]),
    "medicine": (["Hypertension", "Diabetes", "Antibiotic", "Vaccine", "Magnetic resonance imaging", "Cardiac cycle"],
                 ["고혈압", "당뇨병", "항생제", "백신", "자기 공명 영상", "심장 주기"]),
    "engineering": (["Control theory", "PID controller", "Transistor", "Digital signal processing", "Finite element method", "Operational amplifier"],
                    ["제어 이론", "PID 제어기", "트랜지스터", "디지털 신호 처리", "유한요소법", "연산 증폭기"]),
    "psychology": (["Cognitive dissonance", "Classical conditioning", "Working memory", "Operant conditioning", "Cognitive bias", "Attachment theory"],
                   ["인지 부조화", "고전적 조건 형성", "작업 기억", "조작적 조건화", "인지 편향", "애착 이론"]),
    "history_social": (["Industrial Revolution", "French Revolution", "Cold War", "Separation of powers", "Social contract", "Feudalism"],
                       ["산업 혁명", "프랑스 혁명", "냉전", "권력 분립", "사회 계약", "봉건제"]),
    "cs_other": (["Public-key cryptography", "Distributed computing", "Computer graphics", "Reinforcement learning", "Information retrieval", "Garbage collection (computer science)"],
                 ["공개 키 암호 방식", "분산 컴퓨팅", "컴퓨터 그래픽스", "강화 학습", "정보 검색", "가비지 컬렉션"]),
    "earth_env": (["Plate tectonics", "Greenhouse effect", "Water cycle", "Carbon cycle", "Climate change", "Ecosystem"],
                  ["판 구조론", "온실 효과", "물의 순환", "탄소 순환", "기후 변화", "생태계"]),
    "neuro_cogsci": (["Neuron", "Synapse", "Action potential", "Hippocampus", "Neuroplasticity", "Cerebral cortex"],
                     ["뉴런", "시냅스", "활동 전위", "해마", "신경 가소성", "대뇌 피질"]),
}
END_SECTIONS = re.compile(r"^==+\s*(See also|References|External links|Further reading|Notes|Citations|Bibliography|Sources|참고 문헌|각주|같이 보기|외부 링크|바깥 고리|참고 자료|출처|주석)\s*==+\s*$", re.I)


def fetch(lang: str, title: str) -> str | None:
    q = urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "redirects": 1,
                                "titles": title, "format": "json", "formatversion": 2})
    req = urllib.request.Request(f"https://{lang}.wikipedia.org/w/api.php?{q}", headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                page = json.load(r)["query"]["pages"][0]
            return None if page.get("missing") else page.get("extract")
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 10 * 2 ** attempt)        # 제한에 걸리면 알려 준 만큼(없으면 늘려 가며) 쉰다


def to_markdown(title: str, text: str, max_chars: int) -> str:
    out, size = [f"# {title}\n"], 0
    for block in re.split(r"\n{2,}", text.replace("\r", "")):
        block = block.strip()
        if not block:
            continue
        if END_SECTIONS.match(block):
            break
        m = re.match(r"^(=+)\s*(.+?)\s*=+$", block)
        block = f"{'#' * (len(m.group(1)) if m else 0)} {m.group(2)}" if m and len(m.group(1)) >= 2 else block
        if size + len(block) > max_chars and size > 2000:
            break
        out.append(block)
        size += len(block) + 2
    return "\n\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=9000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for domain, (en, ko) in DOMAINS.items():
        for lang, titles in (("en", en), ("ko", ko)):
            got = 0
            for title in titles:
                if got >= args.per_lang:
                    break
                slug = re.sub(r"[^\w가-힣]+", "-", title).strip("-")[:50]
                if (OUT / f"{domain}__{lang}__{slug}.md").exists():
                    got += 1
                    total += 1
                    continue
                try:
                    text = fetch(lang, title)
                except Exception as e:                          # 일시 오류는 건너뛴다
                    print(f"  {lang}:{title}: {e}")
                    continue
                time.sleep(2.0)
                if not text or len(text) < 1500:
                    print(f"  {lang}:{title}: 없거나 너무 짧다")
                    continue
                slug = re.sub(r"[^\w가-힣]+", "-", title).strip("-")[:50]
                (OUT / f"{domain}__{lang}__{slug}.md").write_text(to_markdown(title, text, args.max_chars), encoding="utf-8")
                got += 1
                total += 1
            print(f"{domain}/{lang}: {got}")
    print(f"saved {total} articles → {OUT}")


if __name__ == "__main__":
    main()
