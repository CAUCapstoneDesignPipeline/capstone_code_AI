"""노트 한 편의 줄·문단·절 위치 (결정적). 키워드 사이 연결의 증거 수준(같은 문장 / 제목과 본문 / 같은 문단)을 가르는 데 쓴다.

문단 = 빈 줄이나 제목으로 끊기는 연속된 줄. 절 = 제목 줄부터 다음 제목 줄 직전까지 (첫 제목 앞은 제목 없는 절 0).
코드 블록은 mask 로 가려서 그 안의 줄을 제목·문단으로 보지 않는다. 위치는 모두 원문(Python 문자) 기준이다.
"""

from bisect import bisect_right

from app.services.concepts.markdown import mask


class Layout:
    def __init__(self, text: str):
        self.text = text
        self._starts: list[int] = []
        self._ends: list[int] = []
        self._para: list[int] = []        # 줄 → 문단 번호 (빈 줄·제목 줄은 -1)
        self._sec: list[int] = []         # 줄 → 절 번호
        self._head: list[bool] = []
        self.heading: dict[int, tuple[int, int]] = {}      # 절 번호 → 제목 줄 위치
        pos, para, sec, prev_blank = 0, -1, 0, True
        for line in mask(text).split("\n"):
            start, end = pos, pos + len(line)
            pos = end + 1
            s = line.strip()
            is_head = s.startswith("#")
            if not s:
                prev_blank, p = True, -1
            elif is_head:
                sec += 1
                self.heading[sec] = (start, end)
                prev_blank, p = True, -1
            else:
                if prev_blank:
                    para += 1
                prev_blank, p = False, para
            self._starts.append(start)
            self._ends.append(end)
            self._para.append(p)
            self._sec.append(sec)
            self._head.append(is_head)

    def _line(self, pos: int) -> int:
        return max(0, bisect_right(self._starts, pos) - 1)

    def paragraph(self, pos: int) -> int:
        """pos 가 든 문단 번호. 빈 줄·제목 줄이면 -1."""
        return self._para[self._line(pos)]

    def section(self, pos: int) -> int:
        return self._sec[self._line(pos)]

    def in_heading(self, pos: int) -> bool:
        return self._head[self._line(pos)]

    def heading_text(self, sec: int) -> str:
        a, b = self.heading.get(sec, (0, 0))
        return self.text[a:b].strip().lstrip("#").strip()

    def line_text(self, pos: int) -> str:
        i = self._line(pos)
        return self.text[self._starts[i]:self._ends[i]].strip()
