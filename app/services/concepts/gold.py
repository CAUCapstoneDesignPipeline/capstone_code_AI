"""정답 개념(사람 표시)과 추출 결과 비교 (재현율·정밀도). 기준은 docs/eval/gold-concepts.md.

정답 한 줄은 개념 하나이고 같은 개념의 표기를 `|` 로 잇는다. 표기는 대소문자·공백·하이픈 등을 무시하고 정확히
같아야 맞다. 추출 개념은 대표 표기와 언급 표기 중 하나라도 맞으면 그 정답 개념을 찾은 것이다.
  재현율 = 찾은 core 개념 ÷ core 개념 (all 은 mentioned 포함)
  정밀도 = 정답의 어느 개념과 맞는 추출 개념 ÷ 추출 개념
"""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from .extract import concept_key
from .markdown import nfc

LEVELS = ("core", "mentioned")


def loose(s: str) -> str:
    return re.sub(r"[\s\-_·./()\[\]'\"]+", "", concept_key(s))


@dataclass(frozen=True)
class GoldConcept:
    aliases: frozenset[str]          # loose 표기
    label: str                       # 첫 표기 (보고용)
    level: str


@dataclass(frozen=True)
class Extracted:
    label: str
    surfaces: frozenset[str]         # loose 표기 (대표·언급)
    confidence: float


@dataclass
class Score:
    core: int = 0
    core_hit: int = 0
    all: int = 0
    all_hit: int = 0
    extracted: int = 0
    extracted_hit: int = 0
    missed: list[str] = field(default_factory=list)          # 놓친 core 개념
    wrong: list[str] = field(default_factory=list)           # 정답에 없는 추출 개념

    def add(self, o: "Score"):
        for k in ("core", "core_hit", "all", "all_hit", "extracted", "extracted_hit"):
            setattr(self, k, getattr(self, k) + getattr(o, k))

    @property
    def recall_core(self) -> float:
        return self.core_hit / self.core if self.core else 0.0

    @property
    def recall_all(self) -> float:
        return self.all_hit / self.all if self.all else 0.0

    @property
    def precision(self) -> float:
        return self.extracted_hit / self.extracted if self.extracted else 0.0


def load_gold(path: Path, include_draft: bool = False) -> dict[str, list[GoldConcept]]:
    """note_id → 정답 개념. status 가 reviewed 인 줄만 (include_draft 면 draft 도)."""
    out: dict[str, list[GoldConcept]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            status = (row.get("status") or "reviewed").strip()
            if status != "reviewed" and not (include_draft and status == "draft"):
                continue
            level = (row.get("level") or "core").strip()
            if level not in LEVELS:
                raise ValueError(f"level 은 {LEVELS} 중 하나: {row}")
            names = [n.strip() for n in nfc(row["concept"]).split("|") if loose(n)]
            if names:
                out.setdefault(nfc(row["note_id"]).strip(), []).append(
                    GoldConcept(frozenset(loose(n) for n in names), names[0], level))
    return out


def score(gold: list[GoldConcept], extracted: list[Extracted]) -> Score:
    s = Score(core=sum(g.level == "core" for g in gold), all=len(gold), extracted=len(extracted))
    found = set().union(*(e.surfaces for e in extracted)) if extracted else set()
    for g in gold:
        hit = bool(g.aliases & found)
        s.all_hit += hit
        if g.level == "core":
            s.core_hit += hit
            if not hit:
                s.missed.append(g.label)
    known = set().union(*(g.aliases for g in gold)) if gold else set()
    for e in extracted:
        if e.surfaces & known:
            s.extracted_hit += 1
        else:
            s.wrong.append(e.label)
    return s
