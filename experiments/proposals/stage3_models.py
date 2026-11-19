"""3단계(지식 제안)를 OpenAI · Gemini 모델로 돌려 선별·카드 만들기의 성능과 지연을 잰다 (프로토타입, 운영 코드는 바꾸지 않는다).

  uv run --env-file .env python experiments/proposals/stage3_models.py screen     # 선별: 평가 쌍 184개를 60쌍씩 (운영과 같은 묶음 10 · 동시 8)
  uv run --env-file .env python experiments/proposals/stage3_models.py generate   # 카드: Haiku 가 고른 25쌍을 모델마다 만들고 확인한다
  uv run python experiments/proposals/stage3_models.py report                     # 표 (모델 호출 없음)

운영의 선별(`pair_screen.py`)·생성(`generate.py`)·확인(`check.py`) 코드를 그대로 쓰고 모델 어댑터만 바꾼다. 운영 어댑터는 웹 검색을 anthropic 에만 두었으므로
여기서 OpenAI(Responses API `web_search`)와 Gemini(`google_search` 그라운딩)의 complete_web 을 만든다. 출처 규칙은 같다: 모델이 쓴 URL 이 이 호출의 검색 결과에 실제로 있어야 남는다.
  - Gemini 의 검색 결과 URL 은 리다이렉트 주소라 따라가서 실제 주소로 바꾼 뒤 비교한다 (그 시간도 지연에 든다). 검색 횟수를 한 번으로 묶을 수 없다.
  - 생성 prompt 의 "submit_knowledge 도구로 제출" 한 줄은 구조화 출력(JSON) 으로 답하라로 바꾼다 (운영의 Claude 는 도구로 받는다).
effort 는 운영과 맞춘다: 선별·확인 low, 생성 medium (OpenAI reasoning.effort, Gemini thinking_level). 확인은 같은 회사의 큰 모델이 한다 (운영: Opus 생성 · Sonnet 확인).
Claude 는 Anthropic 크레딧이 없어 새로 돌리지 못했다. 선별은 저장된 Haiku · Sonnet 결과와 비교하고, 카드 생성의 Claude 기준은 앞선 측정(다른 쌍)뿐이다.
쌍의 두 문장이 OpenAI · Google 로 간다. 결과는 data/private/cache/experiments/proposals/stage3/ 에 남는다.
"""

import argparse
import csv
import json
import pickle
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.proposals.decisions_screen import load_labels  # noqa: E402
from experiments.proposals.relation_features import CACHE  # noqa: E402

from app.services.links.check import KnowledgeChecker  # noqa: E402
from app.services.links.generate import SYSTEM as GEN_SYSTEM, KnowledgeGenerator  # noqa: E402
from app.services.links.pair_screen import PairScreen  # noqa: E402
from app.services.llm import LLMError, LLMResult, parse_object  # noqa: E402
from app.services.llm.anthropic_llm import WebResult  # noqa: E402

GOLD = ROOT / "data/private/gold"
OUT = CACHE / "stage3"
MODELS = {                                   # 이름 → (제공자, 모델 id, 확인 모델)
    "gpt-5.4-mini": ("openai", "gpt-5.4-mini", "gpt-5.5"),
    "gpt-5.5": ("openai", "gpt-5.5", "gpt-5.5"),
    "gemini-3.5-flash": ("google", "gemini-3.5-flash", "gemini-pro-latest"),
    "gemini-pro-latest": ("google", "gemini-pro-latest", "gemini-pro-latest"),
}
SUBMIT_LINE = "(3) 아래 규칙대로 결과를 만들어 submit_knowledge 도구를 한 번 호출해 제출한다 (일반 글로 답하지 않는다)."
JSON_LINE = "(3) 아래 규칙대로 결과를 만들어 JSON 하나로 답한다."
assert SUBMIT_LINE in GEN_SYSTEM
USAGE: dict = {}                             # (모델, user) → 토큰·검색 (생성 결과에 붙인다)
LOCK = threading.Lock()


def _openai_error(e):
    import openai
    if isinstance(e, openai.APITimeoutError):
        return LLMError("timeout", str(e))
    if isinstance(e, openai.RateLimitError):
        return LLMError("rate_limit", str(e))
    if isinstance(e, openai.APIStatusError):
        return LLMError("provider", f"{e.status_code} {str(e)[:200]}")
    return LLMError("provider", str(e)[:200])


class OpenAIModel:
    provider = "openai"

    def __init__(self, model: str, effort: str, timeout: float = 240):
        import openai
        self.client, self.model, self.effort = openai.OpenAI(max_retries=2, timeout=timeout), model, effort

    def complete_json(self, system, user, schema, name):
        try:
            r = self.client.responses.create(model=self.model, instructions=system, input=user, reasoning={"effort": self.effort}, store=False,
                                             text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}})
        except Exception as e:
            raise _openai_error(e)
        if r.status != "completed" or not r.output_text:
            raise LLMError("invalid_output", f"status={r.status}")
        return LLMResult(parse_object(r.output_text), self.provider, r.model or self.model)

    def complete_web(self, system, user, schema, name, max_searches=1, tool=None, on_event=None, submit=None):
        t0 = time.perf_counter()
        try:
            r = self.client.responses.create(
                model=self.model, instructions=system.replace(SUBMIT_LINE, JSON_LINE), input=user, reasoning={"effort": self.effort}, store=False,
                tools=[{"type": "web_search"}], max_tool_calls=max_searches, include=["web_search_call.action.sources"],
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}})
        except Exception as e:
            raise _openai_error(e)
        if r.status != "completed" or not r.output_text:
            raise LLMError("invalid_output", f"status={r.status}")
        sources, searches = {}, 0
        for item in r.output:
            if item.type == "web_search_call":
                searches += 1
                for s in getattr(getattr(item, "action", None), "sources", None) or []:
                    url = getattr(s, "url", None)
                    if url:
                        sources.setdefault(url, getattr(s, "title", "") or "")
            elif item.type == "message":
                for c in item.content:
                    for a in getattr(c, "annotations", None) or []:
                        if getattr(a, "type", "") == "url_citation":
                            sources[a.url] = getattr(a, "title", "") or sources.get(a.url, "")
        usage = {"input": r.usage.input_tokens, "output": r.usage.output_tokens}
        with LOCK:
            USAGE[(self.model, user)] = {**usage, "searches": searches, "seconds": time.perf_counter() - t0, "nSources": len(sources)}
        return WebResult(parse_object(r.output_text), self.provider, r.model or self.model, [{"url": u, "title": t} for u, t in sources.items()], searches, usage)


class GeminiModel:
    provider = "google"

    def __init__(self, model: str, effort: str, timeout: float = 240):
        import os

        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(timeout=int(timeout * 1000)))
        self.model, self.effort, self.resolved = model, effort, None

    def _config(self, system, schema, search: bool):
        from google.genai import types
        kw = dict(system_instruction=system, response_mime_type="application/json", response_json_schema=schema, max_output_tokens=16000,
                  automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if self.effort == "low":
            kw["thinking_config"] = types.ThinkingConfig(thinking_level="low")
        if search:
            kw["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        return types.GenerateContentConfig(**kw)

    def _call(self, system, user, schema, search):
        from google.genai import errors
        for attempt in range(4):
            try:
                return self.client.models.generate_content(model=self.model, contents=user, config=self._config(system, schema, search))
            except errors.APIError as e:
                if e.code == 429 and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise LLMError("rate_limit" if e.code == 429 else "provider", f"{e.code} {(e.message or '')[:200]}")
            except Exception as e:
                raise LLMError("provider", str(e)[:200])

    def _text(self, r):
        if not r.candidates:
            raise LLMError("invalid_output", "no candidates")
        finish = getattr(r.candidates[0].finish_reason, "value", str(r.candidates[0].finish_reason))
        if finish != "STOP":
            raise LLMError("invalid_output", f"finish_reason={finish}")
        self.resolved = r.model_version or self.resolved
        return r.text

    def complete_json(self, system, user, schema, name):
        r = self._call(system, user, schema, False)
        return LLMResult(parse_object(self._text(r)), self.provider, r.model_version or self.model)

    def complete_web(self, system, user, schema, name, max_searches=1, tool=None, on_event=None, submit=None):
        import httpx
        t0 = time.perf_counter()
        r = self._call(system.replace(SUBMIT_LINE, JSON_LINE), user, schema, True)
        data = parse_object(self._text(r))
        gm = r.candidates[0].grounding_metadata
        chunks = [(c.web.uri, c.web.title or "") for c in (getattr(gm, "grounding_chunks", None) or []) if getattr(c, "web", None)]
        t1 = time.perf_counter()

        def real(uri):                                  # 그라운딩 URL 은 리다이렉트 주소다: 따라가서 실제 주소를 얻는다
            try:
                loc = httpx.get(uri, follow_redirects=False, timeout=10).headers.get("location")
                return loc or uri
            except httpx.HTTPError:
                return uri
        with ThreadPoolExecutor(8) as pool:
            urls = list(pool.map(real, [u for u, _ in chunks]))
        sources = [{"url": u, "title": t} for u, (_, t) in zip(urls, chunks)]
        searches = len(getattr(gm, "web_search_queries", None) or [])
        # 모델은 검색 결과의 실제 URL 을 보지 못해서 (그라운딩 메타데이터에만 있다) 모델이 쓴 URL 은 거의 맞지 않는다. 그래서 출처는 그라운딩이 돌려준 페이지로 바꾸고
        # (많이 뒷받침한 순, 최대 3개, quote 는 그 페이지가 뒷받침한 결론 구간), 무관한 출처는 확인 단계가 거른다. 모델이 쓴 URL 수는 따로 남긴다.
        model_urls = len(data.get("external") or [])
        if sources:
            count, quote = {}, {}
            for sup in getattr(gm, "grounding_supports", None) or []:
                for j in sup.grounding_chunk_indices or []:
                    count[j] = count.get(j, 0) + 1
                    quote.setdefault(j, (getattr(sup.segment, "text", "") or "")[:300])
            order = sorted(range(len(sources)), key=lambda j: (-count.get(j, 0), j))
            seen, ext = set(), []
            for j in order:
                if sources[j]["url"] not in seen:
                    seen.add(sources[j]["url"])
                    ext.append({"url": sources[j]["url"], "quote": quote.get(j, "")})
            data["external"] = ext[:3]
        um = r.usage_metadata
        usage = {"input": um.prompt_token_count or 0, "output": (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)}
        with LOCK:
            USAGE[(self.model, user)] = {**usage, "searches": searches, "seconds": time.perf_counter() - t0, "resolveSeconds": time.perf_counter() - t1,
                                         "nSources": len(sources), "modelUrls": model_urls}
        return WebResult(data, self.provider, r.model_version or self.model, sources, searches, usage)


def make(provider, model, effort):
    return (OpenAIModel if provider == "openai" else GeminiModel)(model, effort)


def eval_pairs():
    pairs = pickle.load(open(CACHE / "labeled.pkl", "rb"))["pairs"]
    haiku = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-haiku-5-5.csv", encoding="utf-8"))}
    return [p for p in pairs if p["arm"] != "random" and float(p["cos"] or 0) >= 0.5 and p["id"] in haiku], haiku


def screen(names):
    pairs, _ = eval_pairs()
    items = [{"id": p["id"], "a": {"keyword": p["A_keyword"][:100], "text": p["A_text"][:600]}, "b": {"keyword": p["B_keyword"][:100], "text": p["B_text"][:600]}}
             for p in pairs]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = OUT / f"screen-{name}.json"
        if path.exists():
            print(f"{name}: 있음 ({path.name})")
            continue
        provider, model, _ = MODELS[name]
        sc = PairScreen(make(provider, model, "low"), None)
        labels, chunks = {}, []
        for i in range(0, len(items), 60):                       # 운영 후보 60쌍 = 노트 하나의 선별
            part = items[i:i + 60]
            t0 = time.perf_counter()
            try:
                res = sc.screen(part)
            except LLMError as e:
                print(f"  {name} 묶음 {i // 60}: 오류 {e.kind} {e}")
                res = {}
            chunks.append({"n": len(part), "seconds": time.perf_counter() - t0})
            labels.update({k: v["label"] for k, v in res.items()})
            print(f"  {name} {len(part)}쌍 {chunks[-1]['seconds']:.1f}s  useful {sum(v['label'] == 'useful' for v in res.values())}", flush=True)
        json.dump({"model": model, "resolved": getattr(sc.llm, "resolved", None), "labels": labels, "chunks": chunks}, open(path, "w"), ensure_ascii=False)


def side(index, el_id: str) -> dict:
    e = index.el[int(el_id)] if not isinstance(el_id, int) else index.el[el_id]
    return {"keyword": e.keyword, "note": e.note, "noteId": e.note_id, "text": e.text, "start": e.start, "end": e.end, "quote": e.quote, "origin": e.origin}


def generate(names, workers):
    from experiments.proposals.labeled_pairs import build_index
    pairs, haiku = eval_pairs()
    chosen = [p for p in pairs if haiku[p["id"]] == "useful"]
    _, index = build_index()
    for name in names:
        path = OUT / f"cards-{name}.json"
        if path.exists():
            print(f"{name}: 있음 ({path.name})")
            continue
        provider, model, check_model = MODELS[name]
        checker = KnowledgeChecker(make(provider, check_model, "low"), None)
        timings: dict = {}

        class Timed:                                              # 확인 단계 시간을 따로 잰다
            model, prompt_version = checker.model, checker.prompt_version

            def check(self, a, b, *args):
                t = time.perf_counter()
                try:
                    return checker.check(a, b, *args)
                finally:
                    timings[(a["text"], b["text"])] = time.perf_counter() - t
        gen = KnowledgeGenerator(make(provider, model, "medium"), None, None, checker=Timed())

        def one(p):
            a, b = side(index, p["a"]), side(index, p["b"])
            t0 = time.perf_counter()
            try:
                r = gen.generate(a, b, "00000000-0000-4000-8000-000000000000")
                err = None
            except LLMError as e:
                r, err = {"status": "error", "reasons": [f"{e.kind}: {str(e)[:160]}"]}, e.kind
            total = time.perf_counter() - t0
            user = json.dumps({"A": {k: a[k] for k in ("keyword", "note", "text")}, "B": {k: b[k] for k in ("keyword", "note", "text")}}, ensure_ascii=False)
            u = USAGE.get((model, user), {})
            return {"id": p["id"], "A": a["text"], "B": b["text"], "Akw": a["keyword"], "Bkw": b["keyword"], "status": r["status"], "reasons": r.get("reasons", []),
                    "conclusion": r.get("conclusion", ""), "newKeyword": r.get("newKeyword", ""), "external": r.get("external", []), "check": r.get("check"),
                    "model": r.get("model"), "error": err, "seconds": total, "checkSeconds": timings.get((a["text"], b["text"])), **{f"u_{k}": v for k, v in u.items()}}
        t0 = time.perf_counter()
        with ThreadPoolExecutor(workers) as pool:
            rows = list(pool.map(one, chosen))
        print(f"{name}: {len(rows)}장 {time.perf_counter() - t0:.0f}s · " + " ".join(f"{s}={sum(r['status'] == s for r in rows)}" for s in ("candidate", "held", "known", "error")), flush=True)
        json.dump(rows, open(path, "w"), ensure_ascii=False, indent=1)


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def report():
    pairs, haiku = eval_pairs()
    ids = [p["id"] for p in pairs]
    labels = load_labels(pairs)
    sonnet = {r["id"]: r["label"] for r in csv.DictReader(open(GOLD / "pairing-screen-claude-sonnet-5-5.csv", encoding="utf-8"))}
    runs = {"Haiku 5.5 (저장됨)": ({i: haiku[i] for i in ids}, None), "Sonnet 5.5 (저장됨)": ({i: sonnet[i] for i in ids}, None)}
    for name in MODELS:
        f = OUT / f"screen-{name}.json"
        if f.exists():
            d = json.load(open(f))
            runs[f"{name}" + (f" ({d['resolved']})" if d.get("resolved") and d["resolved"] != d["model"] else "")] = (d["labels"], d["chunks"])
    print("## 선별 — 평가 쌍 184개 (정밀도 / 재현율), 지연은 60쌍 한 번 (운영: 묶음 10쌍 · 동시 8)\n")
    labs = ("AI 라벨러", "Sonnet", "Gemini", "셋 중 둘")
    print("| 모델 | useful 로 고른 쌍 | 빠진 쌍 | " + " | ".join(labs) + " | 60쌍 지연 (중앙값 · 최대) |")
    print("|---|---|---|" + "---|" * len(labs) + "---|")
    for name, (lab, chunks) in runs.items():
        sel = np.array([lab.get(i) == "useful" for i in ids])
        miss = sum(i not in lab or lab[i] == "unsure" for i in ids)
        cells = []
        for L in labs:
            has = np.array([L in labels[i] for i in ids])
            y = np.array([labels[i].get(L, False) for i in ids])
            s, yy = sel[has], y[has]
            cells.append(f"{(s & yy).sum() / max(1, s.sum()):.0%} / {(s & yy).sum() / max(1, yy.sum()):.0%}")
        full = [c["seconds"] for c in chunks if c["n"] == 60] if chunks else []
        lat = f"{statistics.median(full):.1f} s · {max(full):.1f} s" if full else "4.6 s · 5.6 s (앞선 측정)" if "Haiku" in name else "3.5~6 s (앞선 측정, 36쌍)"
        print(f"| {name} | {int(sel.sum())} | {miss} | " + " | ".join(cells) + f" | {lat} |")

    print("\n## 카드 만들기 + 확인 — Haiku 가 고른 25쌍\n")
    print("| 생성 모델 (확인 모델) | 후보 | 보류 | 기존 내용 | 오류 | 남은 출처 (카드당) | 지어낸 URL 로 보류 | 확인 단계가 보류 | 카드 하나 (중앙값 · p90) | 그중 확인 | 검색 횟수 | 입력 / 출력 토큰 (중앙값) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, (_, _, check_model) in MODELS.items():
        f = OUT / f"cards-{name}.json"
        if not f.exists():
            continue
        rows = json.load(open(f))
        st = lambda s: sum(r["status"] == s for r in rows)
        cand = [r for r in rows if r["status"] == "candidate"]
        fake = sum(any("외부 출처가 없다" in x for x in r["reasons"]) for r in rows)
        chk = sum(r["status"] == "held" and bool(r.get("check")) and not r["check"].get("error") for r in rows)
        secs = [r["seconds"] for r in rows if r["status"] != "error"]
        cs = [r["checkSeconds"] for r in rows if r.get("checkSeconds")]
        srch = [r.get("u_searches", 0) for r in rows if "u_searches" in r]
        ins, outs = [r["u_input"] for r in rows if "u_input" in r], [r["u_output"] for r in rows if "u_output" in r]
        print(f"| {name} ({check_model}) | {len(cand)} | {st('held')} | {st('known')} | {st('error')} | {np.mean([len(r['external']) for r in cand]) if cand else 0:.1f} | {fake} | {chk} | "
              f"{statistics.median(secs):.1f} s · {pct(secs, 90):.1f} s | {statistics.median(cs) if cs else float('nan'):.1f} s | "
              f"{statistics.mean(srch) if srch else 0:.1f} | {int(statistics.median(ins)) if ins else 0:,} / {int(statistics.median(outs)) if outs else 0:,} |")
    print("\n보류 이유 (모델마다 많은 순)")
    for name in MODELS:
        f = OUT / f"cards-{name}.json"
        if f.exists():
            from collections import Counter
            c = Counter(x[:40] for r in json.load(open(f)) if r["status"] != "candidate" for x in r["reasons"])
            print(f"  {name}: " + " · ".join(f"{k} ×{v}" for k, v in c.most_common(5)))


def dump():
    """후보 카드를 읽기 좋게 (모델 이름 없이 섞어서, 열쇠는 따로)."""
    import random
    rows = []
    for name in MODELS:
        f = OUT / f"cards-{name}.json"
        if f.exists():
            rows += [{**r, "gen": name} for r in json.load(open(f)) if r["status"] == "candidate"]
    random.Random(0).shuffle(rows)
    with open(OUT / "cards-blind.md", "w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            fh.write(f"### {i}\nA [{r['Akw']}]: {r['A']}\nB [{r['Bkw']}]: {r['B']}\n→ [{r['newKeyword']}] {r['conclusion']}\n출처: " +
                     " | ".join(e["title"][:60] or e["url"][:60] for e in r["external"]) + "\n\n")
    json.dump([{"n": i, "gen": r["gen"], "id": r["id"]} for i, r in enumerate(rows, 1)], open(OUT / "cards-blind-key.json", "w"))
    print(f"{len(rows)}장 → {OUT / 'cards-blind.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "generate", "report", "dump"])
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    {"screen": lambda: screen(args.models), "generate": lambda: generate(args.models, args.workers), "report": report, "dump": dump}[args.cmd]()
