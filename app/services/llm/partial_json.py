"""모델이 쓰는 JSON 을 글자가 오는 대로 읽어 필드마다 흘려 보낸다 (구조화 출력도 토큰마다 화면에 보이게).

구조화 출력은 JSON 한 덩어리라서 끝나야 읽을 수 있지만, 문자열 값은 쓰이는 대로 이어 붙일 수 있다. JsonStream.feed(조각) 은 조각을 먹고 이벤트를 돌려준다:
  {"path": "conclusion", "delta": "DB 도 "}              문자열 값이 이만큼 더 쓰였다 (이스케이프·유니코드·서로게이트 쌍은 풀어서)
  {"path": "results.0.label", "value": "useful"}          값 하나가 끝났다 (문자열 전체, 숫자, true/false/null)
path 는 키와 배열 번호를 점으로 이은 것이다 ("results.2.conclusion"). 조각을 어디서 자르든 같은 이벤트를 낸다.
모델 출력이라 틀린 JSON 이 올 수 있다: 파서는 예외를 던지지 않고 읽을 수 있는 데까지만 읽는다 (최종 값의 검증은 호출한 쪽이 완성된 JSON 으로 한다).
"""

import json

_ESC = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "/": "/", "\\": "\\", '"': '"'}


class JsonStream:
    def __init__(self):
        self.stack: list[list] = []         # ["obj", 지금 키 또는 None] · ["arr", 번호]
        self.in_str = False
        self.is_key = False
        self.esc = False
        self.uni: str | None = None         # \uXXXX 의 16진수를 모으는 중
        self.high: str | None = None        # 앞 반쪽 서로게이트
        self.text: list[str] = []           # 지금 문자열의 전체
        self.scalar: list[str] = []         # 숫자·true·false·null 을 모으는 중
        self._out: list[dict] = []
        self._pending: list[str] = []       # 이번 조각에서 지금 path 로 쓰인 글
        self._cur_path = ""

    def path(self) -> str:
        return ".".join(str(v) for kind, v in self.stack if v is not None) if self.stack else ""

    def feed(self, chunk: str) -> list[dict]:
        self._out = []
        for ch in chunk:
            self._char(ch)
        self._flush_delta()
        out, self._out = self._out, []
        return out

    # --- 안쪽 ---
    def _flush_delta(self):
        if self._pending and not self.is_key:
            self._out.append({"path": self._cur_path, "delta": "".join(self._pending)})
        self._pending = []

    def _emit_char(self, c: str):
        self.text.append(c)
        if not self.is_key:
            self._pending.append(c)

    def _char(self, ch: str):
        if self.in_str:
            if self.uni is not None:
                self.uni += ch
                if len(self.uni) == 4:
                    try:
                        code = int(self.uni, 16)
                    except ValueError:
                        code = 0xFFFD
                    self.uni = None
                    self._unicode(code)
                return
            if self.esc:
                self.esc = False
                if ch == "u":
                    self.uni = ""
                else:
                    self._emit_char(_ESC.get(ch, ch))
                return
            if ch == "\\":
                self.esc = True
            elif ch == '"':
                self._end_string()
            else:
                self._emit_char(ch)
            return
        if ch == '"':
            self._flush_scalar()
            self.in_str, self.text = True, []
            top = self.stack[-1] if self.stack else None
            self.is_key = bool(top and top[0] == "obj" and top[1] is None)
            if not self.is_key:
                self._flush_delta()
                self._cur_path = self.path()
        elif ch in "{[":
            self._flush_scalar()
            self.stack.append(["obj", None] if ch == "{" else ["arr", 0])
        elif ch in "}]":
            self._flush_scalar()
            if self.stack:
                self.stack.pop()
        elif ch == ",":
            self._flush_scalar()
            if self.stack:
                top = self.stack[-1]
                if top[0] == "obj":
                    top[1] = None
                else:
                    top[1] += 1
        elif ch == ":" or ch.isspace():
            self._flush_scalar()
        else:
            self.scalar.append(ch)
            self._cur_path = self.path()

    def _unicode(self, code: int):
        if 0xD800 <= code <= 0xDBFF:
            self.high = chr(code)
        elif 0xDC00 <= code <= 0xDFFF and self.high:
            self._emit_char(chr(0x10000 + ((ord(self.high) - 0xD800) << 10) + (code - 0xDC00)))
            self.high = None
        else:
            self.high = None
            self._emit_char(chr(code))

    def _end_string(self):
        self.in_str = False
        text = "".join(self.text)
        if self.is_key:
            self.stack[-1][1] = text
        else:
            self._flush_delta()
            self._out.append({"path": self._cur_path, "value": text})
        self.is_key, self.text = False, []

    def _flush_scalar(self):
        if not self.scalar:
            return
        token = "".join(self.scalar)
        self.scalar = []
        try:
            value = json.loads(token)
        except ValueError:
            return
        self._out.append({"path": self._cur_path, "value": value})
