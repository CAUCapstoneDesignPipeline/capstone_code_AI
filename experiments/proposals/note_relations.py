"""라벨 쌍이 들어 있는 노트마다 출고한 경로(학생 개념 + pairs-1 관계)를 돌려 개념과 관계를 모아 둔다.

  uv run --env-file .env python experiments/proposals/note_relations.py [--all]

AI_DECISIONS_JUDGE=1 이 필요하다 (노트 원문이 OpenAI 로 간다 — 서비스의 pairs profile 과 같다). 결과는 노트 글 해시마다 한 파일로
data/private/cache/experiments/proposals/relations/ 에 남아 다시 돌려도 부르지 않는다.
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data/private/cache/experiments/proposals/relations"
LABELED = ROOT / "data/private/cache/experiments/proposals/labeled.pkl"
from app.services.links.note_relations import VERSION, NoteRelations  # noqa: E402


def key(note_id: str, text: str) -> Path:
    return NoteRelations(CACHE, VERSION).path(note_id, text)


def run_note(state, note_id: str, text: str, policy_types, limits) -> dict:
    """한 노트: 개념(키 → 이름·위치)과 관계. 운영 코드(app/services/links/note_relations.py)와 같은 것이다."""
    return NoteRelations(CACHE, VERSION).compute(state.student, state.extractor, state.pair_relations, note_id, text, policy_types, limits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="라벨 쌍이 없는 노트도 (141개 전부)")
    args = ap.parse_args()
    os.environ.update({"AI_SERVICE_TOKEN": os.environ.get("AI_SERVICE_TOKEN", "local-dev-token"), "AI_DECISIONS_JUDGE": "1", "AI_DEV_UI": "1"})
    os.environ.setdefault("AI_EMBEDDING_PROFILES", "embeddings-hash-dev-1")
    notes_dir = Path(os.environ.get("AI_NOTES_DIR", "data/private/notes")).expanduser()
    os.environ["AI_NOTES_DIR"] = str(notes_dir)
    import run_playground
    os.environ["AI_CONCEPT_IDF"] = str(run_playground.ensure_idf(notes_dir))
    from concurrent.futures import ThreadPoolExecutor

    from fastapi.testclient import TestClient

    from app.core.config import EXTRACT_PROFILES
    from app.dev import router as R
    from app.main import app
    with TestClient(app) as client:
        state = client.app.state
        assert state.pair_relations is not None and state.student is not None, "AI_DECISIONS_JUDGE=1 과 OPENAI_API_KEY 가 필요하다"
        corpus, _ = R._links_corpus(state)
        index, _, _ = R._pairing_index(state, corpus, 0.0)
        if args.all:
            names = sorted(corpus.notes)
        else:
            pairs = pickle.load(open(LABELED, "rb"))["pairs"]
            names = sorted({index.el[p[s]].note for p in pairs for s in ("a", "b")})
        profile = EXTRACT_PROFILES[VERSION]
        policy = R._policy_types()
        t0 = time.monotonic()
        todo = [n for n in names if not key(n, corpus.notes[n].text).exists()]
        print(f"노트 {len(names)}개 중 새로 부를 것 {len(todo)}개")

        def go(n):
            t = time.monotonic()
            for attempt in range(6):                    # 분당 토큰 한도(429)에 걸리면 기다렸다 다시 (긴 노트는 요청 하나가 20만 토큰이다)
                try:
                    r = run_note(state, n, corpus.notes[n].text, policy, profile.limits)
                    break
                except Exception as e:
                    if "rate_limit" not in str(e) or attempt == 5:
                        raise
                    time.sleep(15 * (attempt + 1))
            return n, len(r["concepts"]), len(r["relations"]), time.monotonic() - t

        with ThreadPoolExecutor(1) as pool:
            for n, c, r, s in pool.map(go, todo):
                print(f"  {n[:40]:40s} 개념 {c:3d} 관계 {r:3d} {s:5.1f}s", flush=True)
        print(f"끝 {time.monotonic() - t0:.0f}s")


if __name__ == "__main__":
    main()
