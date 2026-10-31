# -*- coding: utf-8 -*-
"""Align word-level ASR output to the original SRT timeline, then lightly
re-split / re-merge cues for readable subtitles."""
from __future__ import annotations

import logging
import re
from bisect import bisect_right

from srt_utils import Cue

log = logging.getLogger("align")

_PUNCT_END = re.compile(r"[.!?;:,]$")

# Part-number prefixes that are routinely split by Whisper ("LM 2596").
_PART_PREFIX = re.compile(
    r"^(?i)(LM|NE|TL|LT|MAX|TPS|IRF|IRL|BAT|AD|UC|SG|MC|CS|NCP|LMC|OPA|INA|"
    r"LTC|1N|SS|MBR|SR|PAM|MP|SG|KA|LM)$"
)

_UNIT_WORD = re.compile(
    r"^(amps?|amperes?|volts?|watts?|ohms?|hertz|farads?|henr(?:y|ies)|"
    r"milliamps?|millivolts?|milliwatts?|milliohms?|kilovolts?|kilowatts?|"
    r"microfarads?|nanofarads?|picofarads?|microhenr(?:y|ies)|millihenr(?:y|ies)|"
    r"microseconds?|nanoseconds?|milliseconds?|seconds?|percent|"
    r"v|a|w|hz|khz|mhz|kohm|mohm|ohm|uf|nf|pf|mh|uh|ms|us|ns)$",
    re.I,
)

_VALUE_WITH_UNIT = re.compile(
    r"\d(?:\.\d+)?\s*(?:mV|kV|V|mA|A|mW|kW|W|kHz|MHz|Hz|kΩ|mΩ|Ω|"
    r"µF|uF|nF|pF|F|mH|uH|µH|H|ms|us|µs|ns|%)$"
)


def _merge_dc_phrases(words: list[dict]) -> list[dict]:
    out: list[dict] = []
    i = 0
    while i < len(words):
        w = words[i]
        a = (w.get("word") or "").strip().upper()
        if (a in {"DC", "AC"} and i + 2 < len(words)
                and (words[i + 1].get("word") or "").strip().lower() == "to"
                and (words[i + 2].get("word") or "").strip().upper() in {"DC", "AC"}):
            m = dict(w)
            m["word"] = a + "-to-" + (words[i + 2].get("word") or "").strip().upper()
            m["end"] = words[i + 2]["end"]
            out.append(m)
            i += 3
            continue
        out.append(w)
        i += 1
    return out


# Multi-word terms that must never be split across two cues.  They are merged
# into a single token (the source text is unchanged once detokenised).
_PROTECTED_PHRASES = [
    ("bipolar", "junction", "transistor"),
    ("bipolar", "transistor"),
    ("operational", "amplifier"),
    ("switching", "regulator"),
    ("linear", "regulator"),
    ("switching", "converter"),
    ("linear", "converter"),
    ("buck", "boost", "converter"),
    ("buck", "converter"),
    ("boost", "converter"),
    ("voltage", "divider"),
    ("voltage", "drop"),
    ("voltage", "level"),
    ("voltage", "regulator"),
    ("voltage", "converter"),
    ("output", "voltage"),
    ("input", "voltage"),
    ("reference", "voltage"),
    ("duty", "cycle"),
    ("power", "supply"),
    ("power", "source"),
    ("power", "consumption"),
    ("active", "low"),
    ("active", "high"),
    ("feedback", "loop"),
    ("feedback", "pin"),
    ("square", "wave"),
    ("integrated", "circuit"),
    ("frequency", "compensation"),
    ("dead", "time"),
    ("peak", "current"),
    ("output", "current"),
    ("load", "resistor"),
    ("current", "flow"),
    ("saw", "tooth"),
]
_PROTECTED_MAXLEN = max(len(p) for p in _PROTECTED_PHRASES)
_PROTECTED_SET = {tuple(p) for p in _PROTECTED_PHRASES}


def _merge_protected_phrases(words: list[dict]) -> list[dict]:
    out: list[dict] = []
    i = 0
    n = len(words)
    while i < n:
        matched = None
        for size in range(min(_PROTECTED_MAXLEN, n - i), 1, -1):
            seq = []
            ok = True
            for k in range(size):
                raw = (words[i + k].get("word") or "").strip()
                clean = raw.strip(".,!?;:-").lower()
                # never merge past a sentence boundary
                if k < size - 1 and re.search(r"[.!?;:,]$", raw):
                    ok = False
                    break
                seq.append(clean)
            if ok and tuple(seq) in _PROTECTED_SET:
                matched = size
                break
        if not matched:
            out.append(words[i])
            i += 1
            continue
        merged = dict(words[i])
        merged["word"] = " ".join((words[i + k].get("word") or "").strip()
                                  for k in range(matched))
        merged["end"] = words[i + matched - 1]["end"]
        out.append(merged)
        i += matched
    return out


def normalize_words(words: list[dict]) -> list[dict]:
    """Repair token boundaries that Whisper produces around numbers,
    part numbers, hyphenated compounds and duplicate units."""
    words = _merge_dc_phrases(words)
    out: list[dict] = []
    for raw in words:
        w = dict(raw)
        tok = (w.get("word") or "").strip()
        if not tok:
            continue
        if out:
            prev = out[-1]
            ptxt = (prev.get("word") or "").strip()
            ptxt_clean = ptxt.rstrip(".,!?;:")
            low = tok.rstrip(".,!?;:").lower()
            join = False
            # 3 + .3V -> 3.3V ; 0 + .6 -0 + .7V -> 0.6-0.7V
            if re.match(r"^[.\-]\d", tok) and re.search(r"\d$", ptxt_clean):
                join = True
            # DC + -to ; N + -channel ; turn + -on
            elif tok.startswith("-") and len(tok) > 1 and re.search(r"[A-Za-z0-9)]$", ptxt_clean):
                join = True
            elif ptxt.endswith("-") and tok:
                join = True
            # LM + 2596 ; R + 1 ; C + 2
            elif _PART_PREFIX.match(ptxt_clean) and re.fullmatch(r"\d+[A-Za-z]?", tok):
                join = True
            elif re.fullmatch(r"[RCLDQ]", ptxt_clean) and re.fullmatch(r"\d+", tok):
                join = True
            # duplicate unit after a value: "1.5V. volts," / "18W watts"
            elif _UNIT_WORD.match(low) and _VALUE_WITH_UNIT.search(ptxt_clean):
                continue
            # adjacent exact duplicate word: "volts. volts." or "VOUT. out."
            elif (low and len(low) > 2 and ptxt_clean
                  and (low == ptxt_clean.lower()
                       or (low.isalpha() and ptxt_clean.isalpha() and len(ptxt_clean) >= 4
                           and (len(low) >= 4 or low in {"out", "put", "top"})
                           and (ptxt_clean.lower().endswith(low)
                                or low.endswith(ptxt_clean.lower()))))):
                continue
            if join:
                sep = "" if (tok.startswith((".", "-")) or ptxt.endswith("-")) else ""
                prev["word"] = ptxt + sep + tok
                prev["end"] = w["end"]
                continue
        out.append(w)
    return _merge_protected_phrases(out)


def flatten_words(asr: dict) -> list[dict]:
    out: list[dict] = []
    for seg in asr.get("segments", []):
        for w in seg.get("words", []):
            txt = (w.get("word") or "").strip()
            if not txt:
                continue
            s, e = float(w["start"]), float(w["end"])
            if e <= s:
                continue
            out.append({
                "start": s, "end": e, "word": txt,
                "prob": float(w.get("prob", 0.0) or 0.0),
                "seg_logprob": float(seg.get("avg_logprob", 0.0) or 0.0),
            })
    out.sort(key=lambda x: (x["start"], x["end"]))
    # fix tiny overlaps to keep gap search sane
    for i in range(1, len(out)):
        if out[i]["start"] < out[i - 1]["end"]:
            out[i]["start"] = out[i - 1]["end"]
    return out


def _nearest_gap(words: list[dict], t: float, window: float = 0.6):
    """Return (gap_start, gap_end) nearest to time t, or None."""
    best = None
    best_score = 1e9
    for i in range(len(words) - 1):
        a, b = words[i]["end"], words[i + 1]["start"]
        if b - a < 0.03:
            continue
        mid = (a + b) / 2
        dist = abs(mid - t)
        if dist > window:
            continue
        # prefer larger gaps when equally close
        score = dist - min(b - a, 0.5) * 0.15
        if score < best_score:
            best_score = score
            best = (a, b)
    return best


def _snap_boundaries(orig: list[Cue], words: list[dict], window: float = 0.6,
                     min_dur: float = 0.8) -> list[dict]:
    items = [{"start": c.start, "end": c.end, "orig": c, "words": []} for c in orig]
    for i in range(len(items) - 1):
        left, right = items[i], items[i + 1]
        t = left["end"]
        gap = _nearest_gap(words, t, window)
        if not gap:
            continue
        a, b = gap
        cand = (a + b) / 2
        if cand - left["start"] < min_dur or right["end"] - cand < min_dur:
            continue
        left["end"] = cand
        right["start"] = cand
    # enforce monotonic non-overlap
    for i in range(1, len(items)):
        if items[i]["start"] < items[i - 1]["end"]:
            mid = (items[i]["start"] + items[i - 1]["end"]) / 2
            items[i - 1]["end"] = mid
            items[i]["start"] = mid
    return items


def _assign_words(items: list[dict], words: list[dict]) -> None:
    if not items or not words:
        return
    starts = [it["start"] for it in items]
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        idx = bisect_right(starts, mid) - 1
        if idx < 0:
            if items[0]["start"] - mid <= 1.0:
                idx = 0
            else:
                continue
        elif idx >= len(items):
            idx = len(items) - 1
            if mid - items[idx]["end"] > 1.0:
                continue
        elif mid >= items[idx]["end"]:
            # inside a gap: pick nearer cue
            if idx + 1 < len(items) and (items[idx + 1]["start"] - mid) < (mid - items[idx]["end"]):
                idx += 1
            elif mid - items[idx]["end"] > 0.8:
                continue
        items[idx]["words"].append(w)


def _detokenize(tokens: list[str]) -> str:
    s = " ".join(tokens)
    s = re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", s)
    s = re.sub(r"([\(\[\{])\s+", r"\1", s)
    s = re.sub(r"\s+'", "'", s)
    s = re.sub(r"\bn't\b", "n't", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    # domain homophones / ASR shorthand
    s = re.sub(r"(?i)\b(converter|regulator)\s+I\s+see\s+the\s+LM(?=\d|\b)",
               lambda m: m.group(1) + " IC, the LM", s)
    s = re.sub(r"(?i)\b(converter|regulator|into|from|the|this|that|an?)\s+I\s+see\b",
               lambda m: m.group(1) + " IC", s)
    s = re.sub(r"(?i)\bI\s+see\s+the\s+LM(?=\d|\b)", "IC, the LM", s)
    s = re.sub(r"\bILC\b", "IC", s)
    s = re.sub(r"\b1C\b", "IC", s)
    s = re.sub(r"(?i)\bics\b", "ICs", s)
    s = re.sub(r"(?i)\bway-in\b", "VIN", s)
    s = re.sub(r"(?i)\bhigh-fragment\b", "high-frequency", s)
    s = re.sub(r"(?i)\bsoft-tooth\b", "sawtooth", s)
    # technical casing
    reps = [
        (r"\bmosfets\b", "MOSFETs"), (r"\bmosfet\b", "MOSFET"),
        (r"\bbjts\b", "BJTs"), (r"\bbjt\b", "BJT"),
        (r"\bigbts?\b", lambda m: m.group(0).upper()),
        (r"\bpwm\b", "PWM"), (r"\bpvm\b", "PWM"), (r"\besr\b", "ESR"), (r"\bemi\b", "EMI"),
        (r"\bdarlington\b", "Darlington"),
        (r"\bop[\s-]?amps?\b", "op-amp"),
        (r"\bdc[\s-]+to[\s-]+dc\b", "DC-DC"),
        (r"\bac[\s-]+to[\s-]+dc\b", "AC-DC"),
        (r"\bdc[\s-]+dc\b", "DC-DC"),
        (r"\bzener\b", "Zener"),
        (r"\bschottky\b", "Schottky"),
        (r"\bccm\b", "CCM"), (r"\bdcm\b", "DCM"),
        (r"\blm\s*(\d{2,4})\b", lambda m: "LM" + m.group(1)),
        (r"\bvin\b", "VIN"), (r"\bvout\b", "VOUT"),
    ]
    for pat, rep in reps:
        s = re.sub(pat, rep, s, flags=re.I)
    # spoken units -> SI notation (3.3 volts -> 3.3V, 150 kilohertz -> 150kHz)
    units = [
        (r"milliamp(?:ere)?s?", "mA"), (r"amp(?:ere)?s?", "A"),
        (r"millivolts?", "mV"), (r"kilovolts?", "kV"), (r"volts?", "V"),
        (r"milliwatts?", "mW"), (r"kilowatts?", "kW"), (r"watts?", "W"),
        (r"kilohertz", "kHz"), (r"megahertz", "MHz"), (r"hertz", "Hz"),
        (r"milliohms?", "mΩ"), (r"kiloohms?", "kΩ"), (r"ohms?", "Ω"),
        (r"microfarads?", "µF"), (r"nanofarads?", "nF"), (r"picofarads?", "pF"),
        (r"farads?", "F"),
        (r"microhenr(?:y|ies)", "µH"), (r"millihenr(?:y|ies)", "mH"),
        (r"henr(?:y|ies)", "H"),
        (r"microseconds?", "µs"), (r"nanoseconds?", "ns"), (r"milliseconds?", "ms"),
        (r"seconds?", "s"), (r"percent", "%"),
    ]
    for pat, rep in units:
        s = re.sub(rf"\b(\d+(?:\.\d+)?)\s*{pat}\b", rf"\g<1>{rep}", s, flags=re.I)
    s = re.sub(r"(\d)\s+%", r"\1%", s)
    if s:
        s = s[0].upper() + s[1:]
    s = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), s)
    # Whisper capitalises the first token of every segment; undo that inside a sentence
    keep_cap = {"Zener", "Schottky", "English", "Celsius", "Kelvin", "Bode",
                "Thevenin", "Norton", "Faraday", "Germanium", "Silicon"}
    def _lower_mid(m):
        word = m.group(1)
        if word in keep_cap or word.isupper():
            return m.group(0)
        return " " + word.lower()
    s = re.sub(r"(?<=[A-Za-z0-9,;:\)])\s+([A-Z][a-z]+)\b", _lower_mid, s)
    return s


def _text_of(item: dict) -> str:
    return _detokenize([w["word"] for w in item["words"]])


def _fix_split_compounds(items: list[dict]) -> None:
    """Keep hyphenated tokens such as 'DC' + '-DC' in the same cue, and do
    not leave a hyphenated compound stranded at the end of a cue."""
    for i in range(len(items) - 1):
        a, b = items[i], items[i + 1]
        if not a["words"] or not b["words"]:
            continue
        first = b["words"][0]["word"].strip()
        last = a["words"][-1]["word"].strip()
        if (first.startswith(("-", "'")) and len(first) > 1) or (last in ("DC", "AC") and first.startswith(("DC", "AC"))):
            if len(a["words"]) > 1 and a["words"][-1]["start"] - a["start"] > 0.3:
                b["words"].insert(0, a["words"].pop())
                continue
        if (len(a["words"]) > 1
                and re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+[.,!?;:]?", last)
                and not re.match(r"^[.,!?;:]", first)):
            b["words"].insert(0, a["words"].pop())


def _fix_boundary_tokens(items: list[dict]) -> None:
    """Move unit words back across a number/unit cue boundary and drop
    duplicated words that Whisper sometimes repeats at a window edge."""
    out: list[dict] = []
    for it in items:
        if not it["words"]:
            continue
        if out:
            a, b = out[-1], it
            a_text = _text_of(a).strip()
            la = a["words"][-1]["word"].strip().strip(".,!?;:")
            fb = b["words"][0]["word"].strip().strip(".,!?;:")
            # duplicated word across the boundary: "... VOUT." / "Out. The ..."
            if fb and la and len(b["words"]) > 1 and (
                fb.lower() == la.lower() or (len(fb) >= 3 and la.lower().endswith(fb.lower()))
            ):
                b["words"].pop(0)
            # "1.5" | "Amps" -> keep the unit with the number
            elif re.search(r"\d+(?:\.\d+)?[.,!?]?$", a_text) and _UNIT_WORD.match(fb) and len(b["words"]) > 1:
                a["words"].append(b["words"].pop(0))
        if it["words"]:
            out.append(it)
    items[:] = out


def _choose_split(words: list[dict], start: float, end: float):
    n = len(words)
    if n < 4:
        return None
    center = (start + end) / 2
    best_i, best_score = None, -1e9
    for i in range(2, n - 1):
        left_end, right_start = words[i - 1]["end"], words[i]["start"]
        if right_start - left_end > 30:  # nonsense
            continue
        if words[i - 1]["end"] - start < 1.0:
            continue
        if end - words[i]["start"] < 1.0:
            continue
        gap = max(0.0, right_start - left_end)
        punct = 1.0 if _PUNCT_END.search(words[i - 1]["word"]) else 0.0
        boundary = (left_end + right_start) / 2
        last_word = re.sub(r"[^\w']", "", words[i - 1]["word"]).lower()
        next_word = re.sub(r"[^\w']", "", words[i]["word"]).lower()
        dangling = -2.0 if last_word in _CONT_WORDS else 0.0
        starts_weak = -1.5 if next_word in _STRONG_CONT else 0.0
        if words[i]["word"].lstrip()[:1].isdigit():
            starts_weak -= 0.8
        score = (gap * 2.0 + punct * 1.2 + dangling + starts_weak
                 - abs(boundary - center) / max(end - start, 1) * 2.0)
        if score > best_score:
            best_score, best_i = score, i
    return best_i


def _split_long(items: list[dict], max_dur: float = 7.5, max_words: int = 26,
                max_chars: int = 110) -> list[dict]:
    out: list[dict] = []
    queue = list(items)
    while queue:
        it = queue.pop(0)
        text = _text_of(it)
        dur = it["end"] - it["start"]
        if dur <= max_dur and len(it["words"]) <= max_words and len(text) <= max_chars:
            out.append(it)
            continue
        i = _choose_split(it["words"], it["start"], it["end"])
        if i is None:
            out.append(it)
            continue
        left_words, right_words = it["words"][:i], it["words"][i:]
        a, b = left_words[-1]["end"], right_words[0]["start"]
        boundary = (a + b) / 2 if b > a else a
        if boundary - it["start"] < 1.0 or it["end"] - boundary < 1.0:
            out.append(it)
            continue
        left = {"start": it["start"], "end": boundary, "orig": it.get("orig"), "words": left_words,
                "flags": list(it.get("flags", [])) + ["split"]}
        right = {"start": boundary, "end": it["end"], "orig": it.get("orig"), "words": right_words,
                 "flags": list(it.get("flags", [])) + ["split"]}
        queue = [left, right] + queue
    out.sort(key=lambda x: x["start"])
    return out


_CONT_WORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "into", "over", "under", "between", "through", "during", "and", "or", "but",
    "that", "which", "who", "whom", "whose", "where", "when", "while", "if",
    "because", "as", "than", "is", "are", "was", "were", "be", "been", "being",
    "will", "would", "can", "could", "should", "may", "might", "must", "have",
    "has", "had", "do", "does", "did", "it", "its", "this", "these", "those",
    "they", "them", "their", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "not", "so", "then", "thus", "also", "very", "more", "most",
    "such", "only", "just", "even", "still", "well", "called", "known", "like",
    "another", "each", "every", "some", "any", "all", "both", "either",
    "neither", "much", "many", "few", "several", "no", "one", "two",
    "around", "about", "approximately", "roughly", "nearly", "almost",
    "across", "along", "against", "without", "within", "per", "via",
    "usually", "always", "never", "often", "sometimes", "really", "actually",
    "simply", "basically", "quite", "rather", "too", "enough", "first",
    "second", "third", "next", "last", "finally", "however", "therefore",
    "otherwise", "instead", "let's", "lets", "please", "now",
    "active", "takes", "take", "uses", "use", "needs", "need", "requires",
    "require", "includes", "include", "contains", "contain", "offers", "offer",
    "provides", "provide", "gives", "give", "makes", "make", "gets", "get",
    "shows", "show", "means", "mean", "causes", "cause", "allows", "allow",
    "helps", "help", "keeps", "keep", "comes", "come", "goes", "go", "works",
    "work", "changes", "change", "becomes", "become", "remains", "remain",
    "happens", "happen", "occurs", "occur", "applies", "apply", "creates",
    "create", "produces", "produce", "generates", "generate", "converts",
    "convert", "controls", "control", "drives", "drive", "flows", "flow",
    "drops", "drop", "rises", "rise", "falls", "fall",
}

_STRONG_CONT = {
    "to", "of", "and", "or", "but", "that", "which", "who", "whom", "whose",
    "where", "when", "while", "if", "because", "as", "than", "so", "then",
    "thus", "in", "on", "at", "by", "for", "with", "from", "into", "over",
    "under", "between", "through", "during", "across", "along", "against",
    "without", "within", "per", "via", "about", "around", "after", "before",
    "above", "below", "up", "down", "off", "out", "also", "only", "just",
    "even", "still", "very",
}


def _ends_open(item: dict) -> bool:
    if not item.get("words"):
        return False
    last = item["words"][-1]["word"].strip()
    if re.search(r"[.!?]$", last):
        return False
    word = re.sub(r"[^\w']", "", last).lower()
    return word in _CONT_WORDS


def _starts_lower(item: dict) -> bool:
    if not item.get("words"):
        return False
    first = item["words"][0]["word"].lstrip()
    return bool(first) and first[0].islower()


def _starts_strong(item: dict) -> bool:
    if not item.get("words"):
        return False
    first = re.sub(r"[^\w']", "", item["words"][0]["word"]).lower()
    return first in _STRONG_CONT


def _starts_digit(item: dict) -> bool:
    if not item.get("words"):
        return False
    first = item["words"][0]["word"].lstrip()
    return bool(first) and first[0].isdigit()


def _merge_dangling(items: list[dict], max_dur: float = 13.0,
                    max_words: int = 36) -> list[dict]:
    """Merge across cue boundaries that split a phrase after a function word
    ("... regulators is that" | "they will ...") or where Whisper's mid-sentence
    casing shows the next cue is a continuation."""
    changed = True
    while changed:
        changed = False
        for i in range(len(items) - 1):
            a, b = items[i], items[i + 1]
            if not a.get("words") or not b.get("words"):
                continue
            last = a["words"][-1]["word"].strip()
            terminal = bool(re.search(r"[.!?]$", last))
            if terminal:
                continue
            if not (_ends_open(a) or _starts_lower(b) or _starts_strong(b) or _starts_digit(b)):
                continue
            if b["start"] - a["end"] > 1.1:
                continue
            if b["end"] - a["start"] > max_dur:
                continue
            if len(a["words"]) + len(b["words"]) > max_words:
                continue
            a["words"] = a["words"] + b["words"]
            a["end"] = b["end"]
            a["flags"] = list(dict.fromkeys(
                list(a.get("flags", [])) + list(b.get("flags", [])) + ["merged"]))
            items[i:i + 2] = [a]
            changed = True
            break
    return items


def _shift_dangling(items: list[dict]) -> list[dict]:
    """When a cue still ends on a function word (and merging would be too long),
    move that word (and any further function words) to the next cue."""
    for i in range(len(items) - 1):
        a, b = items[i], items[i + 1]
        if not a.get("words") or not b.get("words"):
            continue
        moved = 0
        while (a.get("words") and _ends_open(a) and len(a["words"]) > 3
               and moved < 3):
            w = a["words"][-1]
            if w["start"] - a["start"] < 0.8:
                break
            a["words"].pop()
            b["words"].insert(0, w)
            a["end"] = w["start"]
            b["start"] = min(b["start"], w["start"])
            moved += 1
        if moved:
            a["flags"] = list(dict.fromkeys(list(a.get("flags", [])) + ["shifted"]))
            b["flags"] = list(dict.fromkeys(list(b.get("flags", [])) + ["shifted"]))
    return items


def _merge_short(items: list[dict], min_dur: float = 1.2, max_dur: float = 8.5,
                 max_words: int = 26) -> list[dict]:
    changed = True
    while changed:
        changed = False
        for i in range(len(items) - 1):
            a, b = items[i], items[i + 1]
            gap = b["start"] - a["end"]
            if gap > 1.2:
                continue
            dur = b["end"] - a["start"]
            nwords = len(a["words"]) + len(b["words"])
            if dur > max_dur or nwords > max_words:
                continue
            a_short = a["end"] - a["start"] < min_dur or len(a["words"]) < 3
            b_short = b["end"] - b["start"] < min_dur or len(b["words"]) < 3
            if not (a_short or b_short):
                continue
            if gap > 0.6 and not (len(a["words"]) <= 2 or len(b["words"]) <= 2):
                continue
            merged = {
                "start": a["start"], "end": b["end"], "orig": a.get("orig"),
                "words": a["words"] + b["words"],
                "flags": list(dict.fromkeys(list(a.get("flags", [])) + list(b.get("flags", [])) + ["merged"])),
            }
            items[i:i + 2] = [merged]
            changed = True
            break
    return items


def align(asr: dict, orig: list[Cue], duration: float = 0.0,
          snap_window: float = 0.6) -> tuple[list[Cue], dict]:
    words = normalize_words(flatten_words(asr))
    items = _snap_boundaries(orig, words, window=snap_window)
    _assign_words(items, words)

    # drop pure-silence items before splitting
    cleaned = []
    for it in items:
        if not it["words"] and it["end"] - it["start"] < 2.5:
            continue
        cleaned.append(it)
    items = cleaned

    _fix_boundary_tokens(items)
    items = _split_long(items)
    _fix_boundary_tokens(items)
    items = _merge_dangling(items)
    _shift_dangling(items)
    items = _split_long(items)
    _shift_dangling(items)
    _fix_boundary_tokens(items)
    items = _merge_short(items)
    _fix_split_compounds(items)
    _shift_dangling(items)
    items = _merge_short(items)
    _fix_boundary_tokens(items)

    cues: list[Cue] = []
    for it in items:
        text = _text_of(it)
        if not text:
            continue
        flags = list(it.get("flags", []))
        probs = [w["prob"] for w in it["words"] if w.get("prob")]
        if probs and sum(probs) / len(probs) < 0.35:
            flags.append("low_asr_conf")
        cues.append(Cue(
            start=max(0.0, it["start"]),
            end=min(duration, it["end"]) if duration else it["end"],
            text=text,
            flags=flags,
            orig_index=(it.get("orig").index if it.get("orig") else None),
        ))

    # fallback: if ASR produced nothing for an interval, keep original text
    for c in cues:
        if c.text:
            continue
        overlaps = [o for o in orig if min(c.end, o.end) - max(c.start, o.start) > 0.4 * min(c.duration, o.duration)]
        if len(overlaps) == 1:
            c.text = overlaps[0].text.replace("\n", " ").strip()
            c.flags.append("asr_empty_fallback")
    cues = [c for c in cues if c.text.strip()]
    cues.sort(key=lambda x: x.start)
    # final safety: no overlap
    for i in range(1, len(cues)):
        if cues[i].start < cues[i - 1].end:
            mid = (cues[i].start + cues[i - 1].end) / 2
            cues[i - 1].end = max(cues[i - 1].start + 0.1, mid)
            cues[i].start = min(cues[i].end - 0.1, mid)
    for i, c in enumerate(cues, 1):
        c.index = i

    stats = {
        "orig_cues": len(orig),
        "asr_words": len(words),
        "final_cues": len(cues),
        "split": sum(1 for c in cues if "split" in c.flags),
        "merged": sum(1 for c in cues if "merged" in c.flags),
        "low_conf": sum(1 for c in cues if "low_asr_conf" in c.flags),
        "fallback": sum(1 for c in cues if "asr_empty_fallback" in c.flags),
    }
    log.info("align: %s", stats)
    return cues, stats
