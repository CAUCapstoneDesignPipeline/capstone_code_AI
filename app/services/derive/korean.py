"""받침에 맞는 조사 (설명 문장을 이름 뒤에 붙여 쓸 때)."""

_PAIRS = {"은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"), "가": ("이", "가"), "을": ("을", "를"),
          "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와"), "으로": ("으로", "로"), "로": ("으로", "로"),
          "이에요": ("이에요", "예요"), "예요": ("이에요", "예요")}
# 영문·숫자로 끝나는 이름: 읽을 때 받침이 있는지 (L·M·N·R → 엘·엠·엔·알, 0·1·3·6·7·8 → 영·일·삼·육·칠·팔)
_LATIN_FINAL = set("lmnrLMNR0136785")
_LATIN_RIEUL = set("lLrR18")          # ㄹ 받침 (엘·알·일·팔): '으로' 가 아니라 '로'


def _final(word: str) -> int | None:
    """마지막 글자의 받침: None = 받침 없음, 8 = ㄹ, 그 밖의 수 = 다른 받침."""
    w = word.rstrip(" )]}'\"")
    if not w:
        return None
    ch = w[-1]
    if "가" <= ch <= "힣":
        jong = (ord(ch) - 0xAC00) % 28
        return jong or None
    if ch in _LATIN_FINAL:
        return 8 if ch in _LATIN_RIEUL else 1
    return None


def josa(word: str, particle: str) -> str:
    """word 에 맞는 조사를 붙인다. josa('힙', '을') → '힙을', josa('큐', '을') → '큐를'."""
    with_final, without = _PAIRS[particle]
    f = _final(word)
    if particle in ("으로", "로"):
        return word + (without if f in (None, 8) else with_final)
    return word + (without if f is None else with_final)
