"""공개 어휘집(lexicon)용 글 모으기: 위키백과 검색으로 주제별 글을 찾고(영어), 같은 글의 한국어판을 언어 링크로 찾아 함께 받는다 (CC BY-SA).

  uv run python experiments/distill/fetch_lexicon_corpus.py [--per-query 8] [--max-chars 9000]

과목 노트와 같은 컴퓨터 분야를 많이 (30개 질의) 두고 다른 분야도 섞었다 (16개). data/private/notes-lex/<언어>__<제목>.md 에 쓰고 저장소에 넣지 않는다.
기본은 글의 첫 부분(요약 절)만 20개씩 한 번에 받는다 (위키백과가 요청을 제한해 전체 글을 하나씩 받으면 몇 시간이 걸린다). `--full` 은 전체 글을 하나씩 받는다.
이미 받은 글(notes-web, notes-lex)은 건너뛴다. 위키백과의 제한(429)에는 알려 준 시간만큼 쉬고 다시 한다.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from experiments.distill.fetch_web import to_markdown

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/private/notes-lex"
UA = "concept-extraction-research/1.0"
QUERIES = [
    "operating system process scheduling", "virtual memory paging", "file system journaling", "relational database transactions", "database indexing B-tree",
    "SQL query optimization", "TCP congestion control", "internet routing protocol", "HTTP web protocol", "DNS domain name system",
    "public key cryptography TLS", "compiler parsing", "compiler code optimization", "type system programming language", "garbage collection memory management",
    "concurrency synchronization deadlock", "distributed consensus algorithm", "cache memory hierarchy", "machine learning classification",
    "neural network training backpropagation", "reinforcement learning", "natural language processing", "convolutional neural network computer vision",
    "dynamic programming algorithm", "graph algorithm shortest path", "sorting algorithm complexity", "software testing methodology",
    "software requirements engineering", "agile software development", "object-oriented design patterns",
    "quantum mechanics", "thermodynamics", "genetics", "ecology", "organic chemistry", "calculus", "probability theory", "microeconomics", "macroeconomics",
    "cognitive psychology", "immune system", "cardiovascular disease", "signal processing", "control systems", "robotics", "climate science", "linear algebra", "statistics",
]


def api(lang: str, params: dict):
    url = f"https://{lang}.wikipedia.org/w/api.php?{urllib.parse.urlencode({**params, 'format': 'json', 'formatversion': 2})}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == 5:
                raise
            time.sleep(float(e.headers.get("Retry-After") or 0) or 10 * 2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=8)
    ap.add_argument("--max-chars", type=int, default=9000)
    ap.add_argument("--full", action="store_true", help="요약이 아니라 전체 글을 하나씩 받는다 (느리다)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    have = {p.stem.split("__", 2)[-1].lower() for p in (ROOT / "data/private/notes-web").glob("*.md")} | {p.stem.split("__", 1)[-1].lower() for p in OUT.glob("*.md")}
    cache = OUT.parent / "notes-lex-titles.json"
    titles: list[str] = []
    ko = {}
    if cache.exists():
        c = json.loads(cache.read_text(encoding="utf-8"))
        titles, ko = c["titles"], c["ko"]
    for q in ([] if titles else QUERIES):
        res = api("en", {"action": "query", "list": "search", "srsearch": q, "srnamespace": 0, "srlimit": args.per_query})
        titles += [h["title"] for h in res["query"]["search"] if not h["title"].lower().startswith("list of")]
        time.sleep(0.5)
    titles = list(dict.fromkeys(titles))
    print(f"{len(titles)} english titles from {len(QUERIES)} queries", flush=True)
    for i in range(0, len(titles) if not ko else 0, 40):
        res = api("en", {"action": "query", "prop": "langlinks", "lllang": "ko", "lllimit": 500, "titles": "|".join(titles[i:i + 40]), "redirects": 1})
        for p in res["query"]["pages"]:
            for ll in p.get("langlinks", []):
                ko[p["title"]] = ll["title"]
        time.sleep(0.5)
    cache.write_text(json.dumps({"titles": titles, "ko": ko}, ensure_ascii=False), encoding="utf-8")
    jobs = [("en", t) for t in titles] + [("ko", ko[t]) for t in titles if t in ko]
    print(f"{len(ko)} have a korean version; {len(jobs)} articles to fetch", flush=True)
    saved = skipped = short = 0
    if not args.full:
        for lang in ("en", "ko"):
            todo = [t for l, t in jobs if l == lang]
            for i in range(0, len(todo), 20):
                batch = todo[i:i + 20]
                try:
                    res = api(lang, {"action": "query", "prop": "extracts", "explaintext": 1, "exintro": 1, "exlimit": 20, "redirects": 1, "titles": "|".join(batch)})
                except Exception as e:
                    print(f"  {lang} batch {i}: {e}", flush=True)
                    continue
                for page in res["query"]["pages"]:
                    text, title = page.get("extract") or "", page.get("title", "")
                    slug = re.sub(r"[^\w가-힣]+", "-", title).strip("-")[:60]
                    if page.get("missing") or len(text) < 500 or f"{lang}__intro__{slug}".lower() in have:
                        short += 1
                        continue
                    (OUT / f"{lang}__intro__{slug}.md").write_text(f"# {title}\n\n{text.strip()}\n", encoding="utf-8")
                    saved += 1
                print(f"  {lang}: {min(i + 20, len(todo))}/{len(todo)} requested, {saved} saved", flush=True)
                time.sleep(1.5)
        print(f"done: saved {saved}, too short/missing {short}", flush=True)
        return
    for lang, title in jobs:
        slug = re.sub(r"[^\w가-힣]+", "-", title).strip("-")[:60]
        if f"{lang}__{slug}".lower() in have or slug.lower() in have:
            skipped += 1
            continue
        try:
            res = api(lang, {"action": "query", "prop": "extracts", "explaintext": 1, "exsectionformat": "wiki", "redirects": 1, "titles": title})
        except Exception as e:
            print(f"  {lang}:{title}: {e}", flush=True)
            continue
        time.sleep(2.0)
        page = res["query"]["pages"][0]
        text = page.get("extract") or ""
        if page.get("missing") or len(text) < 3000:
            short += 1
            continue
        (OUT / f"{lang}__{slug}.md").write_text(to_markdown(title, text, args.max_chars), encoding="utf-8")
        saved += 1
        if saved % 25 == 0:
            print(f"  saved {saved} ...", flush=True)
    print(f"done: saved {saved}, skipped {skipped} (already have), too short/missing {short}", flush=True)


if __name__ == "__main__":
    main()
