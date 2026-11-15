"""실험 스크립트의 캐시·중간 산출물 자리: data/private/cache/experiments/ (git 에 올라가지 않는다). 폴더는 처음 쓸 때 만든다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scratch(name: str) -> Path:
    path = ROOT / "data/private/cache/experiments" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
