"""노트마다 출고한 경로(학생 개념 + pairs-1 관계)가 낸 개념·관계를 한 파일씩 캐시한다. 후보 순위를 관계로 매기는 재료 (relation_rank.py).

노트 글이 같으면 다시 부르지 않는다 (키 = profile 이름 + 노트 id + 글의 해시). 읽기만 하는 `read` 는 모델을 부르지 않고,
부르는 쪽은 `compute` 와 `Warmup` 뿐이다 — 노트 원문이 OpenAI 로 나가므로 (AI_DECISIONS_JUDGE=1) 누른 뒤에만 돈다.
"""

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from app.services.links.relation_rank import Graph

VERSION = "extract-student-pairs-1"


class NoteRelations:
    def __init__(self, cache_dir: Path, version: str = VERSION):
        self.dir, self.version = Path(cache_dir), version

    def path(self, note_id: str, text: str) -> Path:
        return self.dir / (hashlib.sha256(f"{self.version}\x1f{note_id}\x1f{text}".encode()).hexdigest()[:24] + ".json")

    def read(self, note_id: str, text: str) -> dict | None:
        p = self.path(note_id, text)
        try:
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        except (OSError, ValueError):
            return None

    def compute(self, student, extractor, pair_relations, note_id: str, text: str, policy_types, limits) -> dict:
        """한 노트: 개념(키 → 이름·위치)과 관계(원천 키, 대상 키, 유형, 확신도, 근거 위치). OpenAI 를 부른다."""
        cached = self.read(note_id, text)
        if cached is not None:
            return cached
        selected = student.select(extractor, note_id, 1, text)
        evidence, relations = pair_relations.extract(text, selected, policy_types, limits)
        out = {"note": note_id,
               "concepts": {k: {"surface": ms[0].surface, "mentions": [[m.span.start, m.span.end] for m in ms]} for k, ms in selected.items()},
               "relations": [{"source": r.source, "target": r.target, "code": r.code, "confidence": r.confidence,
                              "evidence": [[evidence[i].start16, evidence[i].end16] for i in r.evidence]} for r in relations]}
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path(note_id, text).with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path(note_id, text))
        return out

    def graph(self, texts: Mapping[str, str]) -> tuple[Graph, int]:
        """캐시에 있는 노트만으로 만든 그래프와 그 노트 수. 모델을 부르지 않는다."""
        found = {n: d for n, t in texts.items() if (d := self.read(n, t)) is not None}
        return Graph(found), len(found)

    def missing(self, texts: Mapping[str, str]) -> list[str]:
        return sorted(n for n, t in texts.items() if not self.path(n, t).exists())


class Warmup:
    """빠진 노트의 관계를 배경 스레드 하나로 채운다. 요청 하나가 20만 토큰이 되는 긴 노트가 있어 분당 토큰 한도(429)에 걸리면 기다렸다 다시 한다."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.done = self.total = 0
        self.error: str | None = None

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, todo: list[str], work: Callable[[str], None], attempts: int = 6, wait: float = 15.0) -> bool:
        """이미 돌고 있으면 False."""
        with self.lock:
            if self.running:
                return False
            self.done, self.total, self.error = 0, len(todo), None

            def run():
                for n in todo:
                    for attempt in range(attempts):
                        try:
                            work(n)
                            break
                        except Exception as e:
                            if "rate_limit" in str(e) and attempt < attempts - 1:
                                time.sleep(wait * (attempt + 1))
                                continue
                            self.error = f"{n}: {e}"[:300]       # 이 노트만 건너뛰고 계속한다
                            break
                    self.done += 1

            self.thread = threading.Thread(target=run, daemon=True, name="note-relations-warmup")
            self.thread.start()
            return True

    def status(self) -> dict:
        return {"running": self.running, "done": self.done, "todo": self.total, "error": self.error}
