"""추출 시험 v0 실행 스크립트 (instructions.md 의 노트 묶음 방식).

  python run_extract.py groups    # 준비: 정답 쌍을 담은 노트 묶음 초안 groups.csv 작성
  python run_extract.py extract   # 절차 1~4: 묶음마다 관계 추출·개념 병합을 N번 실행하고 기록
  python run_extract.py sheets    # 결과표 results.csv, merges.csv 작성 (코드로 근거 일치 판정)
  python run_extract.py judge     # 평가: 채점 LLM 3개의 판정과 다수결
  python run_extract.py sample    # 평가: 사람이 따로 채점할 관계 50개 human_sample.csv
  python run_extract.py metrics   # 지표 표와 판단 기준 출력

모델 호출 (extract --provider, judge --judges):
  claude   `claude -p` 를 호출마다 새 세션으로 실행. Claude 구독 사용.
           `claude setup-token` 으로 만든 토큰을 .env 의 CLAUDE_CODE_OAUTH_TOKEN 에 둔다.
  gemini   Gemini API. .env 의 GEMINI_API_KEY 사용 (무료 등급은 모델당 하루 20회).

입력 (커밋 금지):
  ai/data/private/notes/*.md|*.txt          노트. 파일 이름(확장자 제외)이 note id
  ai/data/private/extract-v0/selected.txt   시험에 쓸 note id 한 줄에 하나. groups 가 빈자리를 채울 때 쓴다
  ai/data/private/extract-v0/gold_pairs.csv 추출 전에 적은 정답 쌍. 열: note_a,note_b,bridge_concept
                                            bridge_concept 는 표기 변형을 | 로 잇는다 (Bellman 방정식|Bellman equation)
결과 (이 폴더): groups.csv, results.csv, merges.csv, human_sample.csv
원문 응답·실행 기록·채점 원문은 노트 문장을 담으므로 ai/data/private/extract-v0/runs/ 에 둔다.

놓친 관계는 채점자가 results.csv 끝에 행을 추가해 적는다: rel_label=놓친 관계, decided_by=채점자 이름.
"""

import argparse
import csv
import difflib
import itertools
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("extract")

HERE = Path(__file__).resolve().parent
AI_ROOT = HERE.parents[1]
PRIVATE = AI_ROOT / "data" / "private" / "extract-v0"
NOTES_DIR = AI_ROOT / "data" / "private" / "notes"
SELECTED = PRIVATE / "selected.txt"
GOLD_PAIRS = PRIVATE / "gold_pairs.csv"
RUNS_DIR = PRIVATE / "runs"
RELATION_TYPES = HERE / "relation_types.json"
GROUPS = HERE / "groups.csv"
RESULTS = HERE / "results.csv"
MERGES = HERE / "merges.csv"
HUMAN_SAMPLE = HERE / "human_sample.csv"
SHEETS_SOURCE = HERE / "sheets_source.json"  # results/merges 를 어느 실행에서 만들었는지

PROMPT_VERSION = "group-v1"

# 평가 라벨 (instructions.md "평가 지표")
REL_LABELS = ["맞음", "유형 틀림", "방향 틀림", "근거 불일치", "없는 관계"]
MISSED = "놓친 관계"
EVI_LABELS = ["일치", "변형", "없음"]
MERGE_LABELS = ["맞음", "잘못 합침", "못 합침"]
BRIDGE_LABELS = ["찾아서 병합함", "양쪽에 있지만 못 찾음", "한쪽에만 있음", "둘 다 없음"]

# 판단 기준 (instructions.md "판단 기준", 첫 회의에서 확정)
THRESHOLDS = {
    "precision": (0.70, 0.50),
    "recall": (0.50, 0.20),
    "false_link": 0.10,
}

RESULT_FIELDS = ["group_id", "run", "rel_id", "subject_doc", "subject", "relation_type", "object_doc", "object",
                 "subject_evidence", "object_evidence", "subject_evidence_match", "object_evidence_match",
                 "certainty", "judge_1", "judge_2", "judge_3", "rel_label", "decided_by"]
MERGE_FIELDS = ["group_id", "run", "concept_id", "concept", "members", "n_docs",
                "judge_1", "judge_2", "judge_3", "merge_label", "decided_by"]
GROUP_FIELDS = ["group_id", "docs", "gold_pairs"]
DEFAULT_JUDGES = ["claude:claude-sonnet-5-5", "claude:claude-haiku-4-5", "gemini:gemini-3.5-flash"]


# ---------- 공통 ----------

class TqdmHandler(logging.Handler):
    """진행 막대를 깨지 않도록 tqdm.write 로 로그를 찍는다."""

    def emit(self, record):
        try:
            from tqdm import tqdm
            tqdm.write(self.format(record), file=sys.stderr)
        except ImportError:
            print(self.format(record), file=sys.stderr)


def setup_logging(verbose):
    handler = TqdmHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    log.propagate = False
    for noisy in ("httpx", "httpcore", "google_genai"):  # 요청마다 찍히는 INFO 줄 끄기
        logging.getLogger(noisy).setLevel(logging.WARNING)


def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def squash(s):
    return re.sub(r"\s+", " ", nfc(s)).strip()


def loose(s):
    """표기 비교용: 공백·기호를 지우고 소문자로."""
    return re.sub(r"[\s\-_·.,/()\[\]]+", "", nfc(s).lower())


def pair_key(a, b):
    return tuple(sorted((a, b)))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{k: nfc(v or "") for k, v in row.items()} for row in csv.DictReader(f)]


def write_csv(path, fields, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)  # 쓰는 도중 끊겨도 원래 파일이 깨지지 않게


def guard_overwrite(path, force):
    if path.exists() and not force:
        sys.exit(f"⏭️ 이미 있다 (덮어쓰려면 --force): {path}")


def load_relation_types():
    return json.loads(RELATION_TYPES.read_text(encoding="utf-8"))["types"]


def note_paths():
    return {nfc(p.stem): p for p in NOTES_DIR.glob("*") if p.suffix in (".md", ".txt")}


def load_note_texts(ids):
    paths = note_paths()
    missing = [i for i in ids if i not in paths]
    if missing:
        sys.exit("노트 파일을 찾지 못했다:\n  " + "\n  ".join(missing))
    return {i: nfc(paths[i].read_text(encoding="utf-8")) for i in ids}


def load_selected():
    if not SELECTED.exists():
        sys.exit(f"노트 선택 목록이 없다: {SELECTED}")
    return [nfc(l).strip() for l in SELECTED.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.lstrip().startswith("#")]


def load_gold_pairs():
    if not GOLD_PAIRS.exists():
        sys.exit(f"정답 쌍 목록이 없다: {GOLD_PAIRS}\n추출을 실행하기 전에 먼저 적어 둔다.")
    pairs = []
    for row in read_csv(GOLD_PAIRS):
        aliases = [a.strip() for a in row["bridge_concept"].split("|") if a.strip()]
        pairs.append({"note_a": row["note_a"].strip(), "note_b": row["note_b"].strip(), "aliases": aliases})
    return pairs


def load_groups():
    if not GROUPS.exists():
        sys.exit(f"묶음 목록이 없다: {GROUPS}\n먼저 `groups` 로 초안을 만들고 검토한다.")
    groups = []
    for row in read_csv(GROUPS):
        docs = [d.strip() for d in row["docs"].split(";") if d.strip()]
        if len(docs) < 2:
            sys.exit(f"묶음 {row['group_id']} 의 노트가 2개 미만이다.")
        groups.append({"group_id": row["group_id"].strip(), "docs": docs})
    return groups


def group_gold_pairs(docs, gold):
    inside = set(docs)
    return [(p["note_a"], p["note_b"]) for p in gold if p["note_a"] in inside and p["note_b"] in inside]


def parse_json_body(text):
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fence:
        body = fence.group(1).strip()
    return json.loads(body)


def notes_block(docs, texts):
    """노트를 N1, N2 ... 로 바꿔 넣는다. 긴 노트 id 를 모델이 그대로 따라 쓰다 틀리는 일을 막는다."""
    alias = {f"N{i}": d for i, d in enumerate(docs, 1)}
    body = "\n\n".join(f'<note id="{a}">\n{texts[d]}\n</note>' for a, d in alias.items())
    return alias, body


# ---------- 준비: 노트 묶음 ----------

def cmd_groups(args):
    guard_overwrite(GROUPS, args.force)
    gold = load_gold_pairs()
    selected = load_selected()
    gold_docs = {d for p in gold for d in (p["note_a"], p["note_b"])}
    load_note_texts(sorted(gold_docs | set(selected)))  # 파일이 다 있는지 먼저 확인
    rng = random.Random(args.seed)

    # 정답 쌍을 섞어 앞에서부터 묶음에 담는다. 쌍이 들어갈 자리가 없으면 새 묶음을 연다.
    pairs = [(p["note_a"], p["note_b"]) for p in gold]
    rng.shuffle(pairs)
    groups, cur = [], []
    for a, b in pairs:
        new = [d for d in (a, b) if d not in cur]
        if cur and len(cur) + len(new) > args.size:
            groups.append(cur)
            cur = []
            new = [a, b]
        cur += new
    if cur:
        groups.append(cur)

    # 빈자리는 정답 쌍과 관계없는 노트로 채운다. 없으면 다른 정답 쌍의 노트를 빌린다.
    for g in groups:
        while len(g) < args.size:
            outside = [d for d in selected if d not in g and d not in gold_docs]
            borrow = [d for d in selected + sorted(gold_docs) if d not in g]
            pool = outside or borrow
            if not pool:
                break
            g.append(rng.choice(pool))
        rng.shuffle(g)  # 정답 쌍이 늘 앞에 오지 않게

    rows = []
    for i, g in enumerate(groups, 1):
        inside = group_gold_pairs(g, gold)
        rows.append({"group_id": f"g{i:02d}", "docs": ";".join(g),
                     "gold_pairs": ";".join(f"{a}+{b}" for a, b in inside)})
        log.info(f"🧺 g{i:02d} · 노트 {len(g)}개 · 정답 쌍 {len(inside)}개: " + " | ".join(d[:28] for d in g))
    write_csv(GROUPS, GROUP_FIELDS, rows)
    log.info(f"📝 작성: {GROUPS.name} (묶음 {len(rows)}개). 검토하고 고친 뒤 extract 를 실행한다.")


# ---------- 모델 호출 (claude -p / Gemini) ----------

class RetryableError(Exception):
    """잠시 뒤 다시 시도하면 될 수 있는 오류 (과부하, 분당 한도, 타임아웃)."""


class QuotaExhausted(Exception):
    """오늘 한도를 다 쓴 경우. 다시 시도해도 소용없으므로 그 모델 호출을 멈춘다."""


class FatalError(Exception):
    """로그인 안 됨처럼 모든 호출이 똑같이 실패할 오류. 그 모델 호출을 멈춘다."""


@dataclass
class CallResult:
    raw_text: str
    finish: str          # 끝난 이유 (모델 쪽 표기 그대로)
    finished_ok: bool    # 답이 잘리거나 거절되지 않고 끝났는가
    model_served: str
    n_in: int
    n_out: int
    n_think: int
    cost_usd: float | None  # 제공자가 알려 준 비용. 없으면 None
    raw_json: str


def find_claude_bin():
    """CLAUDE_BIN → PATH 의 claude → 데스크톱 앱에 들어 있는 최신 claude 순으로 찾는다."""
    if os.environ.get("CLAUDE_BIN"):
        return os.environ["CLAUDE_BIN"]
    if shutil.which("claude"):
        return shutil.which("claude")
    base = Path.home() / "Library" / "Application Support" / "Claude" / "claude-code"
    found = sorted(base.glob("*/claude.app/Contents/MacOS/claude"),
                   key=lambda p: [int(x) if x.isdigit() else 0 for x in p.parts[-5].split(".")])
    if not found:
        raise FatalError("claude CLI 를 찾지 못했다. CLAUDE_BIN 환경 변수로 경로를 알려 줘라.")
    return str(found[-1])


class ClaudeCLI:
    """`claude -p` 를 호출마다 새 세션으로 실행한다. Claude 구독(CLAUDE_CODE_OAUTH_TOKEN)으로 동작한다.
    도구·MCP·설정 파일·세션 저장을 모두 끄고, 빈 폴더에서 실행해 입력 외의 맥락이 들어가지 않게 한다."""

    name = "claude"

    def __init__(self, model, timeout, effort=None, **_):
        self.bin = find_claude_bin()
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.workdir = Path(tempfile.mkdtemp(prefix="extract-v0-"))

    def call(self, system, user_text):
        cmd = [self.bin, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--strict-mcp-config", "--setting-sources", "", "--system-prompt", system, "--model", self.model]
        if self.effort:
            cmd += ["--effort", self.effort]
        try:
            proc = subprocess.run(cmd, input=user_text, capture_output=True, text=True,
                                  timeout=self.timeout, cwd=self.workdir)
        except subprocess.TimeoutExpired:
            raise RetryableError(f"timeout {self.timeout:g}s")
        try:
            out = json.loads(proc.stdout)
        except json.JSONDecodeError:
            raise FatalError(f"claude 출력이 JSON 이 아니다 (exit {proc.returncode}): "
                             f"{(proc.stderr or proc.stdout).strip()[:300]}")
        if out.get("is_error"):
            msg = str(out.get("result", ""))
            status = out.get("api_error_status")
            low = msg.lower()
            if "not logged in" in low or "/login" in low or "invalid api key" in low or status in (401, 403):
                raise FatalError(f"claude 로그인 필요: {msg}  →  `claude setup-token` 으로 만든 토큰을 "
                                 f".env 의 CLAUDE_CODE_OAUTH_TOKEN 에 넣어라")
            if "usage limit" in low or "limit reached" in low or "limit will reset" in low:
                raise QuotaExhausted(msg)
            if status in (429, 500, 502, 503, 504, 529) or "overloaded" in low:
                raise RetryableError(f"{status} {msg[:120]}")
            raise FatalError(f"claude 오류 ({status}): {msg[:300]}")
        u = out.get("usage") or {}
        stop = out.get("stop_reason") or ""
        return CallResult(
            raw_text=str(out.get("result", "")),
            finish=stop,
            finished_ok=stop not in ("max_tokens", "refusal"),
            model_served=",".join((out.get("modelUsage") or {}).keys()) or self.model,
            n_in=(u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
                 + (u.get("cache_read_input_tokens") or 0),
            n_out=u.get("output_tokens") or 0,
            n_think=(u.get("output_tokens_details") or {}).get("thinking_tokens") or 0,
            cost_usd=out.get("total_cost_usd"),
            raw_json=json.dumps(out, ensure_ascii=False, indent=2),
        )


class GeminiAPI:
    """Gemini API. GEMINI_API_KEY 로 인증. SDK 자체 재시도는 끄고 (조용히 멈춘 것처럼 보이므로) 여기서 재시도한다."""

    name = "gemini"

    def __init__(self, model, timeout, price_in=0.0, price_out=0.0, **_):
        from google import genai
        from google.genai import types
        self.model = model
        self.price_in, self.price_out = price_in, price_out
        self.client = genai.Client(http_options=types.HttpOptions(
            timeout=int(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1)))

    def call(self, system, user_text):
        import httpx
        from google.genai import errors, types
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    # 도구를 쓰지 않으므로 자동 함수 호출을 끈다 (SDK 기본값이 켜짐이라 경고가 뜬다)
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                ),
            )
        except errors.APIError as e:
            msg = e.message or ""
            if e.code == 429 and ("free_tier" in msg or "PerDay" in msg or "per day" in msg.lower()):
                raise QuotaExhausted(msg.split("\n")[0] + " " + next(
                    (l for l in msg.splitlines() if "Quota exceeded for metric" in l), ""))
            if e.code in (429, 500, 502, 503, 504):
                raise RetryableError(f"{e.code} {e.status}")
            raise FatalError(f"{e.code} {e.status}: {msg[:300]}")
        except (httpx.TimeoutException, httpx.TransportError) as e:
            raise RetryableError(type(e).__name__)

        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish = getattr(fr, "value", str(fr))
        else:
            fb = response.prompt_feedback
            finish = f"BLOCKED:{fb.block_reason}" if fb and fb.block_reason else "NO_CANDIDATES"
        u = response.usage_metadata
        n_in = (u.prompt_token_count or 0) if u else 0
        n_out = (u.candidates_token_count or 0) if u else 0
        n_think = (u.thoughts_token_count or 0) if u else 0
        return CallResult(
            raw_text=response.text or "",
            finish=finish,
            finished_ok=finish == "STOP",
            model_served=response.model_version or self.model,
            n_in=n_in, n_out=n_out, n_think=n_think,
            cost_usd=(n_in * self.price_in + (n_out + n_think) * self.price_out) / 1e6,
            raw_json=response.model_dump_json(indent=2, exclude={"sdk_http_response"}),
        )


PROVIDERS = {"claude": ClaudeCLI, "gemini": GeminiAPI}
DEFAULT_MODELS = {"claude": "claude-opus-5-5", "gemini": "gemini-3.5-flash"}
DEFAULT_SLEEP = {"claude": 0.0, "gemini": 7.0}  # gemini 무료 등급 분당 한도


def wait_with_heartbeat(pool, fn, label, bar, every=15):
    """fn 을 다른 스레드에서 돌리며, 끝날 때까지 every 초마다 대기 로그를 남긴다."""
    future = pool.submit(fn)
    t0 = time.monotonic()
    while True:
        try:
            return future.result(timeout=every)
        except FutureTimeout:
            waited = time.monotonic() - t0
            bar.set_postfix_str(f"waiting {waited:.0f}s")
            log.info(f"⌛ {label} 응답 기다리는 중... {waited:.0f}s")


def call_with_retry(pool, provider, retries, system, text, label, bar):
    """RetryableError 는 물러났다가 다시 시도한다. 마지막 오류는 그대로 올린다."""
    for attempt in range(1, retries + 1):
        try:
            return wait_with_heartbeat(pool, lambda: provider.call(system, text), label, bar)
        except RetryableError as e:
            if attempt == retries:
                raise
            delay = min(5 * 2 ** (attempt - 1), 90)
            bar.set_postfix_str(f"retry {attempt}/{retries - 1}")
            log.warning(f"🔁 {label} {e} → {delay}s 뒤 다시 시도 ({attempt}/{retries - 1})")
            time.sleep(delay)


def tokens_str(res):
    return f"토큰 in {res.n_in:,} · out {res.n_out:,} · 사고 {res.n_think:,}"


def cmd_models(args):
    if args.provider != "gemini":
        sys.exit("models 명령은 --provider gemini 에서만 쓴다. claude 는 --model 에 claude-opus-5-5 같은 이름을 준다.")
    from google import genai
    client = genai.Client()  # 목록을 넘기는 동안 클라이언트가 닫히지 않게 붙잡아 둔다
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print(m.name.removeprefix("models/"))


# ---------- 절차 1~4: 관계 추출과 개념 병합 ----------

def build_extract_prompt(types):
    type_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in types)
    return f"""너는 여러 공부 노트를 함께 읽고, 서로 다른 노트에 나온 개념 사이의 관계를 찾는다.
입력은 <note id="N1"> 처럼 id 가 붙은 노트 여러 개다.

관계 (relations):
- 관계는 서로 다른 두 노트를 잇는다. subject 는 subject_doc 노트에, object 는 object_doc 노트에 나온 개념이다.
  같은 노트 안의 관계는 뽑지 않는다.
- 두 노트의 원문을 함께 읽으면 드러나는 관계만 뽑는다. 노트에 없는 지식으로 잇지 않는다.
  이을 근거가 없는 노트끼리는 관계를 만들지 않는다. 관계가 하나도 없으면 빈 목록을 낸다.
- 관계 유형은 아래 목록 중 하나만 쓴다.
{type_lines}
- subject, object 는 각 노트에 나온 표기 그대로의 짧은 개념 이름으로 쓴다.
- subject_evidence 는 subject_doc 원문에서, object_evidence 는 object_doc 원문에서 한 글자도 바꾸지 않고 옮긴 문장이다.
- certainty 는 두 노트가 그 관계를 얼마나 분명히 뒷받침하는지 0.0~1.0 으로 적는다.

개념 병합 (merges):
- 서로 다른 노트에서 같은 개념을 가리키는 표기를 하나로 묶는다. 표기가 달라도 같은 개념이면 묶는다.
- members 의 term 은 그 노트에 쓰인 표기 그대로다. 개념 하나는 노트 두 개 이상에 걸쳐야 하고, 세 개 이상이어도 된다.
- concept 는 묶은 개념의 대표 이름이다.

노트는 입력의 id (N1, N2 ...) 로만 가리킨다.
출력은 JSON 하나만, 다른 글 없이:
{{"relations": [{{"subject_doc": "N1", "subject": "...", "relation_type": "...", "object_doc": "N2", "object": "...",
  "subject_evidence": "...", "object_evidence": "...", "certainty": 0.0}}],
 "merges": [{{"concept": "...", "members": [{{"doc": "N1", "term": "..."}}, {{"doc": "N2", "term": "..."}}]}}]}}"""


def parse_extract(text, type_names, alias):
    """원문 응답 → (관계, 병합, 오류). 노트 id 는 실제 note id 로 되돌린다. 오류가 있으면 둘 다 비운다."""
    try:
        data = parse_json_body(text)
    except json.JSONDecodeError as e:
        return [], [], f"json: {e}"
    if not isinstance(data, dict) or not isinstance(data.get("relations"), list) \
            or not isinstance(data.get("merges"), list):
        return [], [], "missing 'relations' or 'merges' list"
    rels, merges = [], []
    keys = ("subject_doc", "subject", "relation_type", "object_doc", "object",
            "subject_evidence", "object_evidence", "certainty")
    for i, r in enumerate(data["relations"]):
        if not isinstance(r, dict) or any(k not in r for k in keys):
            return [], [], f"relation {i}: missing keys"
        if r["relation_type"] not in type_names:
            return [], [], f"relation {i}: unknown type {r['relation_type']!r}"
        if r["subject_doc"] not in alias or r["object_doc"] not in alias:
            return [], [], f"relation {i}: unknown doc {r['subject_doc']!r}/{r['object_doc']!r}"
        if r["subject_doc"] == r["object_doc"]:
            return [], [], f"relation {i}: same doc on both sides"
        rels.append({**r, "subject_doc": alias[r["subject_doc"]], "object_doc": alias[r["object_doc"]]})
    for i, m in enumerate(data["merges"]):
        members = m.get("members") if isinstance(m, dict) else None
        if not isinstance(members, list) or "concept" not in m:
            return [], [], f"merge {i}: missing keys"
        if any(not isinstance(x, dict) or x.get("doc") not in alias or "term" not in x for x in members):
            return [], [], f"merge {i}: bad member"
        if len({x["doc"] for x in members}) < 2:
            return [], [], f"merge {i}: fewer than 2 docs"
        merges.append({"concept": m["concept"],
                       "members": [{"doc": alias[x["doc"]], "term": x["term"]} for x in members]})
    return rels, merges, ""


def run_jobs(jobs, provider, args, system, desc, on_result, on_error):
    """jobs: (label, user_text, key) 목록. 호출마다 재시도·대기 로그·진행 막대를 처리한다.
    on_result(key, res, elapsed) 가 성공 여부를 bool 로 돌려준다. 한도·치명 오류면 남은 일을 건너뛴다."""
    from tqdm import tqdm
    n_ok = n_fail = n_err = 0
    done = 0
    with tqdm(total=len(jobs), desc=desc, unit="call", dynamic_ncols=True) as bar:
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            for i, (label, text, key) in enumerate(jobs):
                if i and args.sleep:
                    bar.set_postfix_str(f"sleep {args.sleep:g}s")
                    time.sleep(args.sleep)
                log.info(f"🚀 {label} 보냄 ({len(text):,}자)")
                bar.set_postfix_str("calling")
                t0 = time.monotonic()
                try:
                    res = call_with_retry(pool, provider, args.retries, system, text, label, bar)
                except (RetryableError, QuotaExhausted, FatalError) as e:
                    n_err += 1
                    done += 1
                    bar.update(1)
                    on_error(key, e, time.monotonic() - t0)
                    if isinstance(e, RetryableError):
                        log.error(f"⚠️ {label} 재시도 다 씀: {e}")
                        continue
                    if isinstance(e, QuotaExhausted):
                        log.error(f"🪫 {label} 오늘 한도를 다 썼다. 남은 호출은 건너뛴다: {e}")
                    else:
                        log.error(f"🛑 {label} 멈춤: {e}")
                    break
                ok = on_result(key, res, time.monotonic() - t0, label)
                n_ok += ok
                n_fail += not ok
                done += 1
                bar.update(1)
                bar.set_postfix(ok=n_ok, fail=n_fail, err=n_err)
        except KeyboardInterrupt:
            log.warning("🛑 중단됨. 지금까지 결과는 남아 있다.")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)  # 진행 중인 요청을 기다리지 않고 바로 끝낸다
    return n_ok, n_fail, n_err, done


def make_provider(provider, model, args):
    try:
        p = PROVIDERS[provider](model=model, timeout=args.timeout, effort=getattr(args, "effort", None),
                                price_in=getattr(args, "price_in", 0.0), price_out=getattr(args, "price_out", 0.0))
    except FatalError as e:
        sys.exit(f"🛑 {e}")
    if provider == "claude":
        log.info(f"🔧 claude CLI: {p.bin}")
    return p


def cmd_extract(args):
    gold = load_gold_pairs()
    groups = load_groups()
    if args.groups:
        groups = [g for g in groups if g["group_id"] in set(args.groups)]
    texts = load_note_texts(sorted({d for g in groups for d in g["docs"]}))
    uncovered = [(p["note_a"], p["note_b"]) for p in gold
                 if not any({p["note_a"], p["note_b"]} <= set(g["docs"]) for g in load_groups())]
    if uncovered:
        log.warning(f"⚠️ 어느 묶음에도 함께 들어가지 않은 정답 쌍 {len(uncovered)}개: {uncovered}")

    types = load_relation_types()
    type_names = {t["name"] for t in types}
    system = build_extract_prompt(types)
    provider = make_provider(args.provider, args.model, args)
    total_calls = len(groups) * args.runs
    log.info(f"📂 묶음 {len(groups)}개 · 노트 {len(texts)}개 ({sum(map(len, texts.values())):,}자) · "
             f"정답 쌍 {len(gold)}개 · 관계 유형 {len(types)}개")
    log.info(f"🤖 {args.provider} · {args.model} · 묶음마다 {args.runs}번 · 호출 {total_calls}번 · "
             f"호출 간격 {args.sleep:g}s · 타임아웃 {args.timeout:g}s · 시도 {args.retries}번")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    (run_dir / "raw").mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({
        "run_id": run_id, "kind": "group", "provider": args.provider, "model": args.model, "effort": args.effort,
        "prompt_version": PROMPT_VERSION, "runs": args.runs, "groups": groups,
        "cost_note": "claude: total_cost_usd 는 API 요금 환산값 (구독 사용 시 실제 청구 아님)" if args.provider == "claude"
                     else f"gemini: price_per_mtok={[args.price_in, args.price_out]}",
        "relation_types": types, "system_prompt": system,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"📁 기록 위치: {run_dir}")

    log_fields = ["run_id", "group_id", "run", "n_docs", "provider", "model_requested", "model_served",
                  "prompt_version", "elapsed_s", "input_tokens", "output_tokens", "thinking_tokens", "cost_usd",
                  "finish_reason", "parse_ok", "parse_error", "n_relations", "n_merges", "api_error"]
    lf = (run_dir / "log.csv").open("w", encoding="utf-8", newline="")
    pf = (run_dir / "parsed.jsonl").open("w", encoding="utf-8")
    csv_log = csv.DictWriter(lf, fieldnames=log_fields)
    csv_log.writeheader()
    total_cost = [0.0]

    jobs = []
    for g in groups:
        alias, body = notes_block(g["docs"], texts)
        for r in range(1, args.runs + 1):
            label = f"[{len(jobs) + 1}/{total_calls}] {g['group_id']} #{r} (노트 {len(g['docs'])}개)"
            jobs.append((label, body, (g, r, alias)))

    def base_row(g, r):
        return {"run_id": run_id, "group_id": g["group_id"], "run": r, "n_docs": len(g["docs"]),
                "provider": args.provider, "model_requested": args.model, "prompt_version": PROMPT_VERSION}

    def on_result(key, res, elapsed, label):
        g, r, alias = key
        (run_dir / "raw" / f"{g['group_id']}__r{r}.json").write_text(res.raw_json, encoding="utf-8")
        if not res.finished_ok:
            rels, merges, err = [], [], f"finish_reason={res.finish}"
        else:
            rels, merges, err = parse_extract(res.raw_text, type_names, alias)
        total_cost[0] += res.cost_usd or 0.0
        csv_log.writerow({**base_row(g, r), "model_served": res.model_served, "elapsed_s": round(elapsed, 2),
                          "input_tokens": res.n_in, "output_tokens": res.n_out, "thinking_tokens": res.n_think,
                          "cost_usd": "" if res.cost_usd is None else round(res.cost_usd, 5),
                          "finish_reason": res.finish, "parse_ok": not err, "parse_error": err,
                          "n_relations": len(rels), "n_merges": len(merges), "api_error": ""})
        lf.flush()
        pf.write(json.dumps({"group_id": g["group_id"], "run": r, "docs": g["docs"], "raw_text": res.raw_text,
                             "parse_ok": not err, "relations": rels, "merges": merges},
                            ensure_ascii=False) + "\n")
        pf.flush()
        if err:
            log.warning(f"❌ {label} 파싱 실패 {elapsed:.1f}s · {tokens_str(res)} · {err}")
        else:
            log.info(f"✅ {label} {elapsed:.1f}s · 관계 {len(rels)}개 · 병합 {len(merges)}개 · {tokens_str(res)}")
        return not err

    def on_error(key, e, elapsed):
        g, r, _ = key
        csv_log.writerow({**base_row(g, r), "elapsed_s": round(elapsed, 2), "parse_ok": False,
                          "api_error": f"{type(e).__name__}: {e}"})
        lf.flush()

    t0 = time.monotonic()
    n_ok, n_fail, n_err, done = run_jobs(jobs, provider, args, system, "🧠 추출", on_result, on_error)
    lf.close()
    pf.close()

    write_stability(run_dir)
    cost_label = "API 환산 비용" if args.provider == "claude" else "비용"
    log.info(f"🏁 실행 {run_id} 끝 ({(time.monotonic() - t0) / 60:.1f}분): 성공 {n_ok} · 파싱 실패 {n_fail} · "
             f"오류 {n_err} · 미실행 {total_calls - done} / {total_calls} · {cost_label} ${total_cost[0]:.4f}")
    rows = list(csv.DictReader((run_dir / "log.csv").open(encoding="utf-8")))
    redo = sorted({r["group_id"] for r in rows if r["api_error"]} | {k[0]["group_id"] for _, _, k in jobs[done:]})
    if redo:
        log.info("🔂 오류·미실행 묶음만 다시 돌리려면: --groups " + " ".join(redo))


def read_parsed(run_dir):
    with (run_dir / "parsed.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rel_key(r):
    return (r["subject_doc"], loose(r["subject"]), r["relation_type"], r["object_doc"], loose(r["object"]))


def merge_key(m):
    return frozenset((x["doc"], loose(x["term"])) for x in m["members"])


def jaccard(a, b):
    return round(len(a & b) / len(a | b), 3) if a | b else 1.0


def write_stability(run_dir):
    """같은 묶음을 두 번 실행한 결과의 겹침 (Jaccard). 관계는 (노트, 주어, 유형, 노트, 목적어), 병합은 (노트, 표기) 집합."""
    by_group = defaultdict(dict)
    for rec in read_parsed(run_dir):
        if rec["parse_ok"]:
            by_group[rec["group_id"]][rec["run"]] = rec
    rows = []
    for gid, recs in sorted(by_group.items()):
        if 1 in recs and 2 in recs:
            ra, rb = ({rel_key(x) for x in recs[k]["relations"]} for k in (1, 2))
            ma, mb = ({merge_key(x) for x in recs[k]["merges"]} for k in (1, 2))
            rows.append({"group_id": gid, "rel_r1": len(ra), "rel_r2": len(rb), "rel_common": len(ra & rb),
                         "rel_jaccard": jaccard(ra, rb), "merge_r1": len(ma), "merge_r2": len(mb),
                         "merge_common": len(ma & mb), "merge_jaccard": jaccard(ma, mb)})
    fields = ["group_id", "rel_r1", "rel_r2", "rel_common", "rel_jaccard",
              "merge_r1", "merge_r2", "merge_common", "merge_jaccard"]
    with (run_dir / "stability.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if rows:
        rj = sum(r["rel_jaccard"] for r in rows) / len(rows)
        mj = sum(r["merge_jaccard"] for r in rows) / len(rows)
        log.info(f"📊 반복 안정성 (묶음 {len(rows)}개 평균 Jaccard): 관계 {rj:.2f} · 병합 {mj:.2f}")


# ---------- 결과표 ----------

def latest_run_dir(run_id=None):
    if run_id:
        return RUNS_DIR / run_id
    runs = sorted(p for p in RUNS_DIR.glob("*")
                  if (p / "meta.json").exists()
                  and json.loads((p / "meta.json").read_text(encoding="utf-8")).get("kind") == "group")
    if not runs:
        sys.exit(f"묶음 방식 실행 기록이 없다: {RUNS_DIR}")
    return runs[-1]


def sheets_run_dir():
    if not SHEETS_SOURCE.exists():
        sys.exit(f"결과표가 어느 실행에서 나왔는지 모른다: {SHEETS_SOURCE} 가 없다. 먼저 sheets 를 실행한다.")
    return RUNS_DIR / json.loads(SHEETS_SOURCE.read_text(encoding="utf-8"))["run_id"]


def auto_evidence(evidence, note_text):
    """근거 문장 자동 판정: 일치 / 변형 / 없음."""
    ev = nfc(evidence)
    if ev and ev in note_text:
        return "일치"
    ev_s = squash(evidence)
    if not ev_s:
        return "없음"
    # 노트를 문장·줄 단위로 나눠 가장 비슷한 조각과 비교
    pieces = [squash(p) for p in re.split(r"(?<=[.!?。])\s+|\n+", note_text) if p.strip()]
    best = max((difflib.SequenceMatcher(None, ev_s, p).ratio() for p in pieces), default=0.0)
    return "변형" if best >= 0.6 else "없음"


def format_members(members):
    return ";".join(f"{x['doc']}: {x['term']}" for x in members)


def parse_members(cell):
    out = []
    for part in cell.split(";"):
        if ": " in part:
            doc, term = part.split(": ", 1)
            out.append({"doc": doc.strip(), "term": term.strip()})
    return out


def cmd_sheets(args):
    guard_overwrite(RESULTS, args.force)
    guard_overwrite(MERGES, args.force)
    run_dir = latest_run_dir(args.run)
    records = [r for r in read_parsed(run_dir) if r["parse_ok"]]
    texts = load_note_texts(sorted({d for r in records for d in r["docs"]}))

    rel_rows, merge_rows = [], []
    for rec in sorted(records, key=lambda r: (r["group_id"], r["run"])):
        gid, run = rec["group_id"], rec["run"]
        for i, r in enumerate(rec["relations"], 1):
            rel_rows.append({
                "group_id": gid, "run": run, "rel_id": f"{gid}-{run}-{i:02d}",
                "subject_doc": r["subject_doc"], "subject": r["subject"], "relation_type": r["relation_type"],
                "object_doc": r["object_doc"], "object": r["object"],
                "subject_evidence": r["subject_evidence"], "object_evidence": r["object_evidence"],
                "subject_evidence_match": auto_evidence(r["subject_evidence"], texts[r["subject_doc"]]),
                "object_evidence_match": auto_evidence(r["object_evidence"], texts[r["object_doc"]]),
                "certainty": r["certainty"],
            })
        for j, m in enumerate(rec["merges"], 1):
            merge_rows.append({
                "group_id": gid, "run": run, "concept_id": f"{gid}-{run}-c{j:02d}", "concept": m["concept"],
                "members": format_members(m["members"]), "n_docs": len({x["doc"] for x in m["members"]}),
            })
    write_csv(RESULTS, RESULT_FIELDS, rel_rows)
    write_csv(MERGES, MERGE_FIELDS, merge_rows)
    SHEETS_SOURCE.write_text(json.dumps({"run_id": run_dir.name}, ensure_ascii=False), encoding="utf-8")
    ev = Counter(r[k] for r in rel_rows for k in ("subject_evidence_match", "object_evidence_match"))
    log.info(f"📝 작성: {RESULTS.name} (관계 {len(rel_rows)}개), {MERGES.name} (병합 {len(merge_rows)}개) · 실행 {run_dir.name}")
    log.info("🔎 근거 문장 자동 판정 (양쪽 합계): " + ", ".join(f"{k} {ev[k]}" for k in EVI_LABELS))


# ---------- 평가: 채점 LLM 3개 ----------

JUDGE_PROMPT = """너는 공부 노트에서 뽑은 관계와 개념 병합을 채점한다.
판정 기준은 "노트 원문에 그렇게 쓰여 있는가"다. 내용의 사실 여부는 보지 않는다.
입력은 <note id="N1"> 처럼 id 가 붙은 노트들과, 그 노트들에서 뽑은 관계·병합 목록이다.

관계 라벨 (하나만 고른다):
- 맞음: 두 근거 문장이 각 노트 원문에 있고, 함께 읽으면 그 유형과 방향의 관계가 드러난다.
- 유형 틀림: 두 개념은 노트 원문대로 이어지지만 관계 유형이 틀렸다.
- 방향 틀림: 유형은 맞지만 주어와 목적어가 뒤바뀌었다.
- 근거 불일치: 근거 문장이 원문에 없거나, 그 관계를 뒷받침하지 않는다.
- 없는 관계: 노트 원문만으로는 두 개념 사이에 그런 관계가 있다고 말할 수 없다.

병합 라벨 (하나만 고른다):
- 맞음: members 의 표기가 모두 같은 개념을 가리키고, 빠진 표기가 없다.
- 잘못 합침: members 중 하나 이상이 다른 개념을 가리킨다.
- 못 합침: members 는 맞지만, 다른 노트에 같은 개념을 가리키는 표기가 분명히 있는데 빠졌다.

목록의 모든 rel_id 와 concept_id 에 라벨을 하나씩 단다.
출력은 JSON 하나만, 다른 글 없이:
{"relations": [{"rel_id": "...", "label": "..."}], "merges": [{"concept_id": "...", "label": "..."}]}"""


def judge_input(docs, texts, rels, merges):
    alias, body = notes_block(docs, texts)
    back = {d: a for a, d in alias.items()}
    items = {
        "relations": [{"rel_id": r["rel_id"], "subject_doc": back[r["subject_doc"]], "subject": r["subject"],
                       "relation_type": r["relation_type"], "object_doc": back[r["object_doc"]],
                       "object": r["object"], "subject_evidence": r["subject_evidence"],
                       "object_evidence": r["object_evidence"]} for r in rels],
        "merges": [{"concept_id": m["concept_id"], "concept": m["concept"],
                    "members": [{"doc": back.get(x["doc"], x["doc"]), "term": x["term"]}
                                for x in parse_members(m["members"])]} for m in merges],
    }
    return f"{body}\n\n<items>\n{json.dumps(items, ensure_ascii=False, indent=1)}\n</items>"


def parse_judge(text):
    try:
        data = parse_json_body(text)
    except json.JSONDecodeError as e:
        return {}, {}, f"json: {e}"
    if not isinstance(data, dict):
        return {}, {}, "not an object"
    rel = {x.get("rel_id"): x.get("label") for x in data.get("relations") or [] if isinstance(x, dict)}
    mer = {x.get("concept_id"): x.get("label") for x in data.get("merges") or [] if isinstance(x, dict)}
    rel = {k: v for k, v in rel.items() if v in REL_LABELS}
    mer = {k: v for k, v in mer.items() if v in MERGE_LABELS}
    return rel, mer, ""


def majority(votes):
    counts = Counter(v for v in votes if v)
    if counts:
        label, n = counts.most_common(1)[0]
        if n >= 2:
            return label
    return ""


def apply_votes(rows, label_col):
    """다수결을 최종 판정에 적는다. 채점자가 정한 행(decided_by 가 vote 가 아닌 값)은 건드리지 않는다."""
    need_human = 0
    for row in rows:
        if row.get("decided_by") not in ("", "vote"):
            continue
        label = majority([row.get(f"judge_{i}", "") for i in (1, 2, 3)])
        row[label_col] = label
        row["decided_by"] = "vote" if label else ""
        need_human += not label and all(row.get(f"judge_{i}") for i in (1, 2, 3))
    return need_human


def cmd_judge(args):
    if len(args.judges) != 3:
        sys.exit("채점 LLM 은 3개여야 한다 (--judges provider:model 세 개).")
    specs = []
    for s in args.judges:
        provider, _, model = s.partition(":")
        if provider not in PROVIDERS or not model:
            sys.exit(f"--judges 형식은 provider:model 이다: {s!r}")
        specs.append((provider, model))
    run_dir = sheets_run_dir()
    extract_model = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["model"]
    same = [m for _, m in specs if m == extract_model]
    if same:
        sys.exit(f"채점 LLM 은 추출 모델({extract_model})과 달라야 한다.")

    rel_rows = read_csv(RESULTS)
    merge_rows = read_csv(MERGES)
    groups = {g["group_id"]: g["docs"] for g in load_groups()}
    texts = load_note_texts(sorted({d for docs in groups.values() for d in docs}))
    rels_by = defaultdict(list)
    for r in rel_rows:
        if r["rel_label"] != MISSED and r["subject_doc"]:
            rels_by[(r["group_id"], r["run"])].append(r)
    merges_by = defaultdict(list)
    for m in merge_rows:
        merges_by[(m["group_id"], m["run"])].append(m)
    units = sorted(set(rels_by) | set(merges_by))
    log.info(f"⚖️ 채점 대상: 관계 {sum(map(len, rels_by.values()))}개 · 병합 {len(merge_rows)}개 · "
             f"묶음×실행 {len(units)}개 · 채점 LLM {', '.join(args.judges)}")
    (run_dir / "judges").mkdir(exist_ok=True)
    (run_dir / "judges" / "judges.json").write_text(json.dumps(
        {f"judge_{i}": s for i, s in enumerate(args.judges, 1)}, ensure_ascii=False, indent=2), encoding="utf-8")

    for j, (provider_name, model) in enumerate(specs, 1):
        col = f"judge_{j}"
        todo = [u for u in units if args.rejudge
                or any(not r[col] for r in rels_by.get(u, [])) or any(not m[col] for m in merges_by.get(u, []))]
        if not todo:
            log.info(f"⏭️ {col} ({provider_name}:{model}) 이미 다 채점했다")
            continue
        args.sleep = DEFAULT_SLEEP[provider_name] if args.sleep_override is None else args.sleep_override
        provider = make_provider(provider_name, model, args)
        out_dir = run_dir / "judges" / f"{col}_{provider_name}_{model}"
        out_dir.mkdir(exist_ok=True)
        jobs = []
        for gid, run in todo:
            text = judge_input(groups[gid], texts, rels_by.get((gid, run), []), merges_by.get((gid, run), []))
            jobs.append((f"[{col} {len(jobs) + 1}/{len(todo)}] {gid} #{run}", text, (gid, run)))

        def on_result(key, res, elapsed, label, col=col, out_dir=out_dir):
            (out_dir / f"{key[0]}__r{key[1]}.json").write_text(res.raw_json, encoding="utf-8")
            rel, mer, err = parse_judge(res.raw_text) if res.finished_ok else ({}, {}, f"finish={res.finish}")
            if err:
                log.warning(f"❌ {label} 파싱 실패 {elapsed:.1f}s · {err}")
                return False
            for r in rels_by.get(key, []):
                r[col] = rel.get(r["rel_id"], r[col])
            for m in merges_by.get(key, []):
                m[col] = mer.get(m["concept_id"], m[col])
            write_csv(RESULTS, RESULT_FIELDS, rel_rows)  # 호출마다 저장해 중간에 끊겨도 남게
            write_csv(MERGES, MERGE_FIELDS, merge_rows)
            n_items = len(rels_by.get(key, [])) + len(merges_by.get(key, []))
            log.info(f"✅ {label} {elapsed:.1f}s · 라벨 {len(rel) + len(mer)}/{n_items}개 · {tokens_str(res)}")
            return True

        def on_error(key, e, elapsed):
            pass

        run_jobs(jobs, provider, args, JUDGE_PROMPT, f"⚖️ {col}", on_result, on_error)

    need_rel = apply_votes(rel_rows, "rel_label")
    need_mer = apply_votes(merge_rows, "merge_label")
    write_csv(RESULTS, RESULT_FIELDS, rel_rows)
    write_csv(MERGES, MERGE_FIELDS, merge_rows)
    voted_rel = sum(r["decided_by"] == "vote" for r in rel_rows)
    voted_mer = sum(m["decided_by"] == "vote" for m in merge_rows)
    log.info(f"🗳️ 다수결: 관계 {voted_rel}/{len(rel_rows)} · 병합 {voted_mer}/{len(merge_rows)}")
    if need_rel or need_mer:
        log.info(f"🙋 세 판정이 모두 달라 채점자가 정할 행: 관계 {need_rel}개 · 병합 {need_mer}개 "
                 f"(rel_label/merge_label 과 decided_by 에 이름을 적는다)")


# ---------- 평가: 사람 채점 표본 ----------

def cmd_sample(args):
    guard_overwrite(HUMAN_SAMPLE, args.force)
    rows = [r for r in read_csv(RESULTS) if r["rel_label"] != MISSED and r["subject_doc"]]
    picked = random.Random(args.seed).sample(rows, min(args.n, len(rows)))
    fields = ["rel_id", "group_id", "run", "subject_doc", "subject", "relation_type", "object_doc", "object",
              "subject_evidence", "object_evidence", "human_label", "grader", "memo"]
    write_csv(HUMAN_SAMPLE, fields, sorted(picked, key=lambda r: r["rel_id"]))
    log.info(f"📝 작성: {HUMAN_SAMPLE.name} (관계 {len(picked)}개). LLM 판정은 넣지 않았다. "
             f"human_label 에 {' / '.join(REL_LABELS)} 중 하나를 적는다.")


# ---------- 지표 ----------

def ratio(n, d):
    return n / d if d else float("nan")


def pct(x):
    return "  n/a" if x != x else f"{x * 100:5.1f}%"


def cmd_metrics(args):
    rel_rows = read_csv(RESULTS)
    merge_rows = read_csv(MERGES)
    groups = load_groups()
    gold = load_gold_pairs()
    gold_keys = {pair_key(p["note_a"], p["note_b"]) for p in gold}
    runs = sorted({r["run"] for r in rel_rows} | {m["run"] for m in merge_rows})

    extracted = [r for r in rel_rows if r["rel_label"] != MISSED and r["subject_doc"]]
    missed = [r for r in rel_rows if r["rel_label"] == MISSED]
    labeled = [r for r in extracted if r["rel_label"] in REL_LABELS]
    rel_dist = Counter(r["rel_label"] for r in labeled)
    precision = ratio(rel_dist["맞음"], len(labeled))

    sides = [r[k] for r in extracted for k in ("subject_evidence_match", "object_evidence_match")]
    evi_both = ratio(sum(r["subject_evidence_match"] == "일치" and r["object_evidence_match"] == "일치"
                         for r in extracted), len(extracted))

    m_labeled = [m for m in merge_rows if m["merge_label"] in MERGE_LABELS]
    m_dist = Counter(m["merge_label"] for m in m_labeled)

    # 노트 연결: 실행마다, 같은 묶음 안의 노트 쌍이 관계로 이어졌는가
    linked = defaultdict(set)    # (run, group) → 관계가 있는 노트 쌍
    linked_ok = defaultdict(set)  # 그중 최종 판정이 맞음인 관계가 있는 쌍
    for r in extracted:
        k = pair_key(r["subject_doc"], r["object_doc"])
        linked[(r["run"], r["group_id"])].add(k)
        if r["rel_label"] == "맞음":
            linked_ok[(r["run"], r["group_id"])].add(k)
    recall_hits = recall_ok = recall_total = 0
    false_hits = false_ok = false_total = 0
    for run in runs:
        for p in gold:
            k = pair_key(p["note_a"], p["note_b"])
            gs = [g["group_id"] for g in groups if set(k) <= set(g["docs"])]
            if not gs:
                continue
            recall_total += 1
            recall_hits += any(k in linked[(run, gid)] for gid in gs)
            recall_ok += any(k in linked_ok[(run, gid)] for gid in gs)
        for g in groups:
            for k in itertools.combinations(sorted(g["docs"]), 2):
                if k in gold_keys:
                    continue
                false_total += 1
                false_hits += k in linked[(run, g["group_id"])]
                false_ok += k in linked_ok[(run, g["group_id"])]
    recall = ratio(recall_hits, recall_total)
    false_rate = ratio(false_hits, false_total)

    # 매개 개념: 병합 결과에 정답 쌍의 매개 개념이 양쪽 노트로 묶였는가 (코드 판정)
    texts = load_note_texts(sorted({d for p in gold for d in (p["note_a"], p["note_b"])}))
    loose_text = {d: loose(t) for d, t in texts.items()}
    bridge_rows = []
    for run in runs:
        for p in gold:
            a, b = p["note_a"], p["note_b"]
            keys = [loose(x) for x in p["aliases"] if loose(x)]
            hit = lambda s: any(k in loose(s) or loose(s) in k for k in keys if loose(s))
            merged = False
            for m in merge_rows:
                if m["run"] != run:
                    continue
                mem = parse_members(m["members"])
                docs = {x["doc"] for x in mem}
                if {a, b} <= docs and (hit(m["concept"]) or any(hit(x["term"]) for x in mem)):
                    merged = True
                    break
            in_a = any(k in loose_text[a] for k in keys)
            in_b = any(k in loose_text[b] for k in keys)
            label = ("찾아서 병합함" if merged else "양쪽에 있지만 못 찾음" if in_a and in_b
                     else "한쪽에만 있음" if in_a or in_b else "둘 다 없음")
            bridge_rows.append((run, p, label))
    b_dist = Counter(l for _, _, l in bridge_rows)

    stab_path = sheets_run_dir() / "stability.csv"
    stab = list(csv.DictReader(stab_path.open(encoding="utf-8"))) if stab_path.exists() else []
    rel_j = ratio(sum(float(s["rel_jaccard"]) for s in stab), len(stab))
    mer_j = ratio(sum(float(s["merge_jaccard"]) for s in stab), len(stab))

    hi, lo = THRESHOLDS["precision"]
    rhi, rlo = THRESHOLDS["recall"]
    print("| 지표 | 값 | 표본 | 판단 |")
    print("| --- | --- | --- | --- |")
    print(f"| 관계 정밀도 | {pct(precision)} | {len(labeled)} | "
          + ("2주차 파이프라인 연결" if precision >= hi else "추출 방식 재검토" if precision < lo else "보류 (50~70%)") + " |")
    print(f"| **정답 쌍 재현율** | {pct(recall)} | {recall_total} | "
          + ("지금 설계 유지" if recall >= rhi else "설계 재검토 회의" if recall < rlo else "보류 (20~50%)") + " |")
    print(f"| 오연결률 | {pct(false_rate)} | {false_total} | "
          + ("연결 판정 기준 강화" if false_rate > THRESHOLDS["false_link"] else "유지") + " |")
    print(f"| 정답 쌍 재현율 (맞음 판정만) | {pct(ratio(recall_ok, recall_total))} | {recall_total} | |")
    print(f"| 오연결 중 맞음 판정이 있는 쌍 | {pct(ratio(false_ok, false_hits))} | {false_hits} | |")
    print(f"| 근거 문장 양쪽 일치 | {pct(evi_both)} | {len(extracted)} | |")
    print(f"| 잘못 합침 비율 | {pct(ratio(m_dist['잘못 합침'], len(m_labeled)))} | {len(m_labeled)} | |")
    print(f"| 못 합침 비율 | {pct(ratio(m_dist['못 합침'], len(m_labeled)))} | {len(m_labeled)} | |")
    print(f"| 매개 개념 찾아서 병합함 | {pct(ratio(b_dist['찾아서 병합함'], len(bridge_rows)))} | {len(bridge_rows)} | |")
    print(f"| 반복 안정성 (관계 / 병합) | {pct(rel_j)} / {pct(mer_j)} | {len(stab)} | |")

    print("\n관계 판정 분포:", {l: rel_dist[l] for l in REL_LABELS})
    print("근거 문장 (한쪽씩):", {l: sides.count(l) for l in EVI_LABELS})
    print("병합 판정 분포:", {l: m_dist[l] for l in MERGE_LABELS})
    print(f"놓친 관계: {len(missed)}개")
    pending_r = len(extracted) - len(labeled)
    pending_m = len(merge_rows) - len(m_labeled)
    if pending_r or pending_m:
        print(f"최종 판정이 없는 행 (지표에서 빠짐): 관계 {pending_r}개 · 병합 {pending_m}개")

    print("\n매개 개념 (정답 쌍마다, 코드 판정):")
    for run, p, label in bridge_rows:
        print(f"  #{run} [{label}] {p['aliases'][0]}: {p['note_a'][:24]} ↔ {p['note_b'][:24]}")

    if HUMAN_SAMPLE.exists():
        human = {h["rel_id"]: h["human_label"] for h in read_csv(HUMAN_SAMPLE) if h["human_label"] in REL_LABELS}
        final = {r["rel_id"]: r["rel_label"] for r in labeled}
        both = [k for k in human if k in final]
        agree = sum(human[k] == final[k] for k in both)
        print(f"\n사람 채점과 LLM 최종 판정 일치율: {pct(ratio(agree, len(both)))} ({agree}/{len(both)})")
        for k in both:
            if human[k] != final[k]:
                print(f"  - {k}: 사람 {human[k]} / LLM {final[k]}")

    wrong = [r for r in labeled if r["rel_label"] != "맞음"]
    if wrong:
        print("\n틀린 사례 후보:")
        for r in wrong[: args.examples]:
            print(f"  - [{r['rel_label']}] {r['rel_id']}: {r['subject']} ({r['subject_doc'][:20]}) "
                  f"-{r['relation_type']}-> {r['object']} ({r['object_doc'][:20]})")


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true", help="디버그 로그까지 출력")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("groups", help="정답 쌍을 담은 노트 묶음 초안 groups.csv 를 만든다")
    g.add_argument("--size", type=int, default=4, help="묶음 하나의 노트 수 (instructions: 3~5)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--force", action="store_true", help="이미 있는 groups.csv 를 덮어쓴다")
    g.set_defaults(func=cmd_groups)

    e = sub.add_parser("extract", help="묶음마다 관계 추출·개념 병합을 N번 실행하고 기록한다")
    e.add_argument("--provider", choices=list(PROVIDERS), default="claude",
                   help="claude: `claude -p` (Claude 구독), gemini: Gemini API")
    e.add_argument("--model", help=f"기본: {DEFAULT_MODELS}")
    e.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                   help="claude 전용. 생략하면 모델 기본값")
    e.add_argument("--sleep", type=float, help="호출 사이 대기 초 (기본: claude 0, gemini 7)")
    e.add_argument("--timeout", type=float, default=600, help="호출 하나의 최대 대기 초")
    e.add_argument("--retries", type=int, default=6, help="과부하·타임아웃 때 총 시도 횟수")
    e.add_argument("--price-in", type=float, default=0.0, help="gemini: USD / 1M 입력 토큰 (무료 등급은 0)")
    e.add_argument("--price-out", type=float, default=0.0, help="gemini: USD / 1M 출력·사고 토큰 (무료 등급은 0)")
    e.add_argument("--runs", type=int, default=2, help="같은 묶음 반복 횟수 (흔들림 확인)")
    e.add_argument("--groups", nargs="*", help="일부 group_id 만 실행")
    e.set_defaults(func=cmd_extract)

    s = sub.add_parser("sheets", help="results.csv, merges.csv 를 만든다")
    s.add_argument("--run", help="run_id (기본: 가장 최근 묶음 실행)")
    s.add_argument("--force", action="store_true", help="이미 있는 결과표를 덮어쓴다 (채점 내용도 지워진다)")
    s.set_defaults(func=cmd_sheets)

    j = sub.add_parser("judge", help="채점 LLM 3개로 판정하고 다수결을 적는다")
    j.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES, help="provider:model 세 개")
    j.add_argument("--sleep", dest="sleep_override", type=float, help="호출 사이 대기 초 (기본: claude 0, gemini 7)")
    j.add_argument("--timeout", type=float, default=600)
    j.add_argument("--retries", type=int, default=6)
    j.add_argument("--rejudge", action="store_true", help="이미 판정한 칸도 다시 채점한다")
    j.set_defaults(func=cmd_judge)

    sp = sub.add_parser("sample", help="사람이 따로 채점할 관계 표본 human_sample.csv 를 만든다")
    sp.add_argument("--n", type=int, default=50)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_sample)

    m = sub.add_parser("metrics", help="지표 표와 판단 기준을 출력한다")
    m.add_argument("--examples", type=int, default=10, help="틀린 사례 출력 개수")
    m.set_defaults(func=cmd_metrics)

    ml = sub.add_parser("models", help="이 API 키로 쓸 수 있는 Gemini 모델 이름을 출력한다")
    ml.add_argument("--provider", choices=list(PROVIDERS), default="gemini")
    ml.set_defaults(func=cmd_models)

    args = ap.parse_args()
    setup_logging(args.verbose)
    if args.cmd == "extract":
        args.model = args.model or DEFAULT_MODELS[args.provider]
        if args.sleep is None:
            args.sleep = DEFAULT_SLEEP[args.provider]
    args.func(args)


if __name__ == "__main__":
    main()
