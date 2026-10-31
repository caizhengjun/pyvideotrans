# -*- coding: utf-8 -*-
"""SRT parsing / writing helpers (standalone, no pyvideotrans imports)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

TS_RE = re.compile(
    r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass
class Cue:
    start: float  # seconds
    end: float
    text: str
    index: int = 0
    flags: list[str] = field(default_factory=list)
    orig_index: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def copy(self) -> "Cue":
        return Cue(self.start, self.end, self.text, self.index, list(self.flags), self.orig_index)


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def parse_srt(path: str | Path) -> list[Cue]:
    text = _decode(Path(path).read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip())
    cues: list[Cue] = []
    for block in blocks:
        lines = [x.strip() for x in block.split("\n")]
        if not lines:
            continue
        # tolerate missing numeric index line
        time_pos = None
        for i, line in enumerate(lines[:3]):
            if "-->" in line:
                time_pos = i
                break
        if time_pos is None:
            continue
        m = TS_RE.search(lines[time_pos])
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        body = lines[time_pos + 1 :]
        body = [x for x in body if x and not re.fullmatch(r"\d+", x)]
        if not body:
            continue
        cues.append(Cue(start=start, end=end, text="\n".join(body), index=len(cues) + 1))
    for i, c in enumerate(cues, 1):
        c.index = i
    return cues


def ms_to_ts(ms: int) -> str:
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms2 = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def write_srt(cues: Iterable[Cue], path: str | Path, crlf: bool = True, bom: bool = False) -> None:
    out: list[str] = []
    for i, c in enumerate(cues, 1):
        text = c.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        out.append(str(i))
        out.append(f"{ms_to_ts(round(c.start * 1000))} --> {ms_to_ts(round(c.end * 1000))}")
        out.append(text)
        out.append("")
    content = "\n".join(out).rstrip() + "\n"
    if crlf:
        content = content.replace("\n", "\r\n")
    data = content.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(data)
