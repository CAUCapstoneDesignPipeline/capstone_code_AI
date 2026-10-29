"""평가 묶음 (`uv run ai eval`): 추출·노트 쌍·숨겼다 다시 찾기·설명 사람 판정을 한 번에 재고, 지난 보고서와 비교한다.

정답 자료가 작아서(노트 8개, 노트 쌍 9개) 숫자 하나로는 개선인지 우연인지 모른다. 그래서
  - 신뢰 구간: 단위(노트·질의·쌍)를 다시 뽑는 bootstrap 95% 구간. seed 를 고정해 같은 입력이면 같은 구간이다
  - dev / test 나누기: 규칙을 고칠 때는 dev 노트만 보고, test 노트는 보고서에서만 본다 (같은 노트로 고치고 재면 숫자가 부풀려진다).
    지금 정답 노트 8개는 모두 규칙을 고치며 본 적이 있어 test 도 깨끗하지 않다. 새로 표시하는 노트는 split=test 로 둔다
  - 비교: 새 값이 지난 보고서의 95% 구간 아래로 떨어지면 '나빠짐' 으로 표시한다
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Callable, Sequence

TEST_SHARE = 0.35          # test 로 떼어 두는 노트 비율 (노트 id hash 로 고정)


def assign_splits(note_ids, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """노트 → dev / test. 정답 CSV 의 split 칸이 있으면 그것을 따르고, 나머지는 노트 id hash 순으로 줄 세워 앞의
    TEST_SHARE 만큼(적어도 1개)을 test 로 둔다. 노트가 적어도 비율이 지켜지고, 같은 노트 묶음이면 늘 같은 결과다."""
    overrides = {n: v for n, v in (overrides or {}).items() if v in ("dev", "test")}
    rest = sorted((n for n in set(note_ids) if n not in overrides),
                  key=lambda n: hashlib.sha256(n.encode("utf-8")).hexdigest())
    n_test = max(1, round(TEST_SHARE * (len(rest) + len(overrides)))) - sum(v == "test" for v in overrides.values())
    out = dict(overrides)
    for i, n in enumerate(rest):
        out[n] = "test" if i < n_test else "dev"
    return out


@dataclass(frozen=True)
class Estimate:
    value: float
    lo: float
    hi: float
    n: int                 # 단위 수

    def to_dict(self) -> dict:
        return {"value": round(self.value, 4), "lo": round(self.lo, 4), "hi": round(self.hi, 4), "n": self.n}

    def __str__(self) -> str:
        return f"{self.value:4.0%} [{self.lo:.0%}–{self.hi:.0%}] n={self.n}"


def bootstrap(units: Sequence, stat: Callable[[Sequence], float], rounds: int = 1000, seed: int = 0) -> Estimate:
    """units 를 복원 추출로 다시 뽑아 stat 의 2.5%·97.5% 분위. 단위가 없으면 0."""
    if not units:
        return Estimate(0.0, 0.0, 0.0, 0)
    rng = random.Random(seed)
    vals = sorted(stat([units[rng.randrange(len(units))] for _ in units]) for _ in range(rounds))
    return Estimate(stat(units), vals[int(0.025 * rounds)], vals[min(rounds - 1, int(0.975 * rounds))], len(units))


def ratio(num: Callable, den: Callable) -> Callable[[Sequence], float]:
    """단위마다 (분자, 분모) 를 더한 비율 (예: 노트마다 찾은 core 수 / core 수)."""
    def f(units):
        d = sum(den(u) for u in units)
        return sum(num(u) for u in units) / d if d else 0.0
    return f


def compare(new: dict, old: dict) -> list[tuple[str, float, float, str]]:
    """같은 이름의 지표마다 (이름, 새 값, 지난 값, 판정). 판정: 나빠짐 / 좋아짐 / 비슷함 (지난 값의 95% 구간 기준)."""
    out = []
    for key in sorted(set(new) & set(old)):
        a, b = new[key], old[key]
        if a["value"] < b["lo"]:
            verdict = "나빠짐"
        elif a["value"] > b["hi"]:
            verdict = "좋아짐"
        else:
            verdict = "비슷함"
        out.append((key, a["value"], b["value"], verdict))
    return out


# ---- ⑤ 설명 사람 판정: 표(사람이 채움)와 시스템 판정 열쇠를 따로 둔다 (판정을 보고 표시하지 않게) ----

AS_HUMAN = {"candidate": "valid", "hypothesis": "unsure", "rejected": "invalid"}
LABELS = ("valid", "invalid", "unsure")


def labelled_explanations(sheet_rows: list[dict], key_rows: list[dict]) -> list[tuple[str, str]]:
    """(시스템 판정을 사람 값으로 바꾼 것, 사람 표시) 목록. 사람이 표시한 줄만. 열쇠에 없는 id 는 오류."""
    key = {r["id"]: r["system"] for r in key_rows}
    out = []
    for r in sheet_rows:
        h = (r.get("human") or "").strip()
        if h not in LABELS:
            continue
        if r["id"] not in key:
            raise ValueError(f"열쇠에 없는 id: {r['id']}")
        out.append((AS_HUMAN[key[r["id"]]], h))
    return out


# ---- ④ 정답 노트 쌍 ----

def gold_pairs(rows: list[dict], include_draft: bool) -> list[tuple[str, str, str]]:
    """pairs.csv 줄 → (note_a, note_b, 매개 개념). status 가 없으면 reviewed (1주차에 사람이 만든 쌍)."""
    from app.services.concepts.markdown import nfc
    out = []
    for r in rows:
        status = (r.get("status") or "reviewed").strip()
        if status == "reviewed" or (include_draft and status == "draft"):
            out.append((nfc(r["note_a"]).strip(), nfc(r["note_b"]).strip(), (r.get("bridge_concept") or "").strip()))
    return out


def judged_precision(top: dict[str, list[str]], judgments: list[dict], k: int = 3) -> list[float]:
    """질의마다 시스템 상위 k 개 중 '이어진다(yes)' 로 판정된 비율. 상위 k 개가 모두 판정된 질의만 센다
    (판정 풀에 든 질의는 S 상위 3 개를 모두 판정했다). 판정은 (query, candidate) 순서 그대로만 본다."""
    from app.services.concepts.markdown import nfc
    label = {(nfc(j["query"]).strip(), nfc(j["candidate"]).strip()): j["label"].strip() for j in judgments}
    out = []
    for q, cands in sorted(top.items()):
        got = [label.get((q, c)) for c in cands[:k]]
        if len(got) == k and None not in got:
            out.append(sum(g == "yes" for g in got) / k)
    return out
