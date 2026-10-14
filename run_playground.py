"""개발용 playground 를 한 번에 띄운다.

    uv run python run_playground.py              # http://127.0.0.1:8000/dev/ 를 열어 준다
    uv run python run_playground.py --port 8010 --no-browser

하는 일
  1. .env 가 있으면 읽는다 (이미 설정된 환경 변수는 덮어쓰지 않는다)
  2. AI_SERVICE_TOKEN 이 없으면 로컬용 값을 쓴다 (BE 와 연결하지 않는 로컬 테스트 전용)
  3. 개념 추출용 IDF 가 없거나 지금 코드와 맞지 않으면
       - data/private/notes 에 노트가 있으면 새로 만든다
       - 노트가 없으면 빈 IDF 로 띄운다 (개념 품질은 조금 떨어지지만 "직접 입력" 은 동작한다)
  4. 개발용 화면(/dev/)을 켜고 서버를 띄운 뒤 브라우저를 연다

배포용이 아니다. 로컬 노트를 읽어 보여 주므로 127.0.0.1 에서만 띄운다.
"""

import argparse
import json
import os
import threading
import unicodedata
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NOTES_DIR = Path("data/private/notes")
DEFAULT_IDF = Path("data/private/concepts/idf.json")
EMPTY_IDF = Path("data/private/concepts/idf-empty.json")


def load_dotenv(path: Path):
    """KEY=VALUE 줄만 읽는다. 따옴표는 벗기고, 줄 끝 # 주석은 무시한다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.split(" #", 1)[0].strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def ensure_idf() -> Path:
    """쓸 수 있는 IDF 경로를 돌려준다. 필요하면 만든다."""
    from app.services.concepts import ConceptExtractor, ExtractConfig, Idf, StaleIdfError, build_idf

    path = Path(os.environ.get("AI_CONCEPT_IDF", DEFAULT_IDF))
    config = ExtractConfig()
    if path.exists():
        try:
            ConceptExtractor(config, Idf.load(path))
            return path                                  # 지금 코드와 맞는다
        except StaleIdfError:
            print("🔁 IDF 가 지금 추출 설정과 맞지 않는다. 다시 만든다.")

    notes = sorted(p for p in NOTES_DIR.glob("*") if p.suffix in (".md", ".txt")) if NOTES_DIR.exists() else []
    if notes:
        print(f"📚 노트 {len(notes)}개로 IDF 를 만든다…")
        texts = {unicodedata.normalize("NFC", p.stem): p.read_text(encoding="utf-8") for p in notes}
        extractor = ConceptExtractor(config)
        idf = build_idf(extractor, texts)
        path.parent.mkdir(parents=True, exist_ok=True)
        idf.save(path, extractor.config_version)
        print(f"   → {path} ({idf.version})")
        return path

    # 노트가 없으면 빈 IDF: 모든 낱말의 IDF 가 같아 빈도·마크다운 구조만으로 고른다
    print("⚠️  data/private/notes 에 노트가 없어 빈 IDF 로 띄운다. '직접 입력' 으로 시험할 수 있다.")
    EMPTY_IDF.parent.mkdir(parents=True, exist_ok=True)
    EMPTY_IDF.write_text(json.dumps({"version": "empty", "config_version": config.version(), "n_docs": 0, "df": {}}),
                         encoding="utf-8")
    return EMPTY_IDF


def main():
    ap = argparse.ArgumentParser(description="개발용 playground 를 띄운다")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true", help="브라우저를 자동으로 열지 않는다")
    args = ap.parse_args()

    os.chdir(ROOT)                                       # data/… 같은 상대 경로가 어디서 실행해도 맞게
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("AI_SERVICE_TOKEN", "local-dev-token")
    os.environ.setdefault("AI_EMBEDDING_PROFILES", "embeddings-hash-dev-1")
    os.environ["AI_DEV_UI"] = "1"
    os.environ["AI_CONCEPT_IDF"] = str(ensure_idf())

    url = f"http://127.0.0.1:{args.port}/dev/"
    print(f"🧪 playground: {url}   (끄려면 Ctrl+C)")
    if not args.no_browser:
        threading.Timer(2.0, webbrowser.open, args=(url,)).start()

    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
