# -*- coding: utf-8 -*-
"""Hy-MT2-7B local server client + subtitle translation prompts/validation."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Iterable

import httpx

warnings.filterwarnings("ignore", category=UserWarning, module="jieba")
try:
    import jieba

    for _term in (
        "开关稳压器", "线性稳压器", "开关电源", "开关元件", "开关器件", "开关频率",
        "降压转换器", "升压转换器", "升降压转换器", "反激式转换器", "正激式转换器",
        "分压电路", "输出电压", "输入电压", "电压等级", "电压电平", "占空比",
        "导通电阻", "开关损耗", "导通损耗", "续流二极管", "稳压二极管",
        "运算放大器", "参考电压", "反馈引脚", "数据手册", "示波器", "纹波",
        "效率", "电感", "电容", "负载", "引脚", "接地", "功耗", "振荡器",
    ):
        jieba.add_word(_term)
except Exception:  # noqa: BLE001
    jieba = None

from config import (
    LATIN_WHITELIST, MAX_CUE_CHARS, MAX_LINE_CHARS, MAX_SOURCE_CHARS_PER_CHUNK,
    MAX_SOURCE_CUES_PER_CHUNK, MT_CACHE, MT_JOBS, MT_MODEL_NAME, MT_MODEL_PATH,
    PROMPT_VERSION, REPEAT_PENALTY, REVIEW_MIN_RATIO, TEMP_RETRY, TEMP_REVIEW,
    TEMP_TRANSLATE, TOP_P, GLOSSARY,
)
from srt_utils import Cue

log = logging.getLogger("mt")

MARK_RE = re.compile(r"@@\s*(\d+)\s*@@")
NUM_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(mV|kV|V|mA|A|mW|kW|W|Hz|kHz|MHz|%|Ω|ohm|F|uF|µF|nF|pF|H|mH|uH|s|ms|us|µs|ns|°C)\b",
    re.I,
)
PARTNO_RE = re.compile(r"\b[A-Z]{2,}\d{2,}[A-Z0-9-]*\b")


# --------------------------------------------------------------------------- prompts
def glossary_text() -> str:
    lines = ["参考下面的翻译："]
    for src, variants in GLOSSARY.items():
        if variants[0].isascii():
            lines.append(f"{src} 保持为 {variants[0]}")
        else:
            lines.append(f"{src} 翻译成 {variants[0]}")
    return "\n".join(lines)


_GLOSSARY_TEXT = glossary_text()


def _marked(lines: Iterable[str]) -> str:
    return "\n".join(f"@@{i}@@ {t}" for i, t in enumerate(lines, 1))


def build_translate_prompt(title: str, context: list[str], sources: list[str],
                           extra_note: str = "") -> str:
    ctx = ""
    if context:
        ctx = "\n【前文参考】（只用于理解上下文，不要翻译，也不要重复）\n" + "\n".join(context)
    note = f"\n【本次特别注意】\n{extra_note}\n" if extra_note else ""
    return f"""你是一名电力电子领域的专业字幕译者。请把下面的英文字幕逐条翻译成简体中文。

【视频主题】{title}
【领域】开关电源、电力电子基础课程
{_GLOSSARY_TEXT}
{ctx}
【翻译要求】
1. 逐条对应：输入有多少条，输出就有多少条；每条必须以相同编号的标记开头，例如 @@1@@。
2. 标记必须原样保留，不得翻译、删除、合并、拆分或改变顺序。
3. 只输出译文和标记，不要解释、不要加引号、不要输出原文。
4. 保持数字、单位、符号、公式、型号不变；技术数值用阿拉伯数字和国际单位，例如 3.3V、5A、100kHz、90%。
5. 数字和单位之间不加空格，也不要写成中文单位：1.5安→1.5A，3.3伏→3.3V，100千赫兹→100kHz，15毫安→15mA，0.6伏→0.6V，10微秒→10µs。
6. 术语严格按上面的术语表；同一概念全文统一。IC 保留为 IC，不要音译。
7. 译文要符合中文字幕习惯：口语化、准确、简洁；单条尽量不超过 40 个汉字，最多两行。
8. 原文是语音识别结果，可能有同音或技术词识别错误；请结合电力电子语境理解真实含义，不要直译明显错误的词。
9. 保留说话人的语气、疑问、强调和指代关系；跨条句子要连贯。
10. 专有名词、芯片型号、缩写（如 LM2596、MOSFET、PWM、ESR、VIN、VOUT）保持原样。
11. 每条尽量以完整短语收尾，不要把“的、它、这个、以及、和”等悬空词单独留在条尾；必要时调整语序或补全省略成分。
12. 严格按本条原文翻译：不要补充原文没有的信息，不要提前翻译下一条，也不要重复上一条已表达的内容。
13. include、contain、have 等表示“包含/内置”，不要译为“由……构成”；如 include internal frequency compensation 译为“内置频率补偿”。
{note}
【待翻译字幕】
{_marked(sources)}"""


def build_review_prompt(title: str, sources: list[str], drafts: list[str],
                        extra_note: str = "") -> str:
    note = f"\n【本次特别注意】\n{extra_note}\n" if extra_note else ""
    return f"""你是资深中文字幕审校。下面是一段电力电子课程字幕的英文原文和中文初译。请对照原文逐条修订中文译文。

【视频主题】{title}
{_GLOSSARY_TEXT}
【审校要求】
1. 输出条数、编号与输入完全一致，每条以 @@n@@ 开头，标记不得改动。
2. 只修改错译、漏译、术语不一致、数字或单位错误、生硬或过长的地方；不要润色过度，不要改变原意。
3. 保持数字、单位、型号、缩写不变；术语严格按术语表。数字和单位紧排：3.3V、1.5A、100kHz、15mA、10µs，不写“3.3伏”“1.5安”。IC 保留为 IC。
4. 只输出修订后的中文和标记，不要解释，不要输出英文原文。
5. 单条尽量不超过 40 个汉字，最多两行；口语化、自然、简洁。
6. 每条尽量以完整短语收尾，不要把“的、它、这个、以及、和”等悬空词单独留在条尾。
7. 严格按本条原文修订：不要补充原文没有的信息，不要提前翻译下一条，也不要重复上一条已表达的内容。
{note}
【英文原文】
{_marked(sources)}

【中文初译】
{_marked(drafts)}"""


# --------------------------------------------------------------------------- parsing / checks
def _strip_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_marked(text: str, n: int) -> list[str] | None:
    text = _strip_fence(text)
    matches = list(MARK_RE.finditer(text))
    if matches:
        got: dict[int, str] = {}
        for i, m in enumerate(matches):
            idx = int(m.group(1))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            val = text[m.end():end].strip()
            val = re.sub(r"^\s*[:：\-–]\s*", "", val)
            val = val.strip().strip("“”\"'")
            val = re.sub(r"\s+", " ", val).strip()
            if idx in got and val:
                got[idx] += " " + val
            elif val:
                got[idx] = val
        if sorted(got) == list(range(1, n + 1)):
            return [got[i] for i in range(1, n + 1)]
        return None
    # fallback: exactly n non-empty lines
    lines = [re.sub(r"^\s*(?:\d+[.、)]|[-*])\s*", "", x).strip()
             for x in text.splitlines() if x.strip()]
    lines = [x for x in lines if x]
    if len(lines) == n:
        return lines
    return None


def _nums_with_units(text: str) -> list[str]:
    out = []
    for m in NUM_UNIT_RE.finditer(text):
        out.append(m.group(1))
    return out


def check_chunk(sources: list[str], targets: list[str]) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    term_misses: list[str] = []
    if len(sources) != len(targets):
        errors.append(f"条数不一致 {len(sources)}->{len(targets)}")
        return {"errors": errors, "warnings": warnings, "term_misses": term_misses}

    for i, (s, t) in enumerate(zip(sources, targets), 1):
        if not t.strip():
            errors.append(f"第{i}条为空")
            continue
        # numbers with units must survive
        for num in _nums_with_units(s):
            if num not in t:
                errors.append(f"第{i}条缺少数值 {num}")
        # part numbers must survive
        for pn in PARTNO_RE.findall(s):
            if pn not in t:
                errors.append(f"第{i}条缺少型号 {pn}")
        # length
        cjk = len(re.findall(r"[\u4e00-\u9fff]", t))
        if cjk > MAX_CUE_CHARS:
            warnings.append(f"第{i}条过长({cjk}字)")
        # residual latin
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9+.\-]*", t):
            if tok in LATIN_WHITELIST or tok.upper() in {x.upper() for x in LATIN_WHITELIST}:
                continue
            if PARTNO_RE.fullmatch(tok) or re.fullmatch(r"[A-Z]{2,}\d+[A-Z0-9-]*", tok):
                continue
            if len(tok) >= 3:
                warnings.append(f"第{i}条残留英文 {tok}")
        if t.count("@@") > 0:
            errors.append(f"第{i}条残留标记")

    # duplicated target lines for different sources
    ct = Counter(t.strip() for t in targets)
    for line, cnt in ct.items():
        if cnt >= 3 and len(line) > 3:
            srcs = {s.strip() for s, t in zip(sources, targets) if t.strip() == line}
            if len(srcs) >= 2:
                errors.append(f"疑似重复译文：{line[:20]}")

    # glossary consistency (chunk level)
    joined_src = " ".join(sources)
    joined_tgt = " ".join(targets)
    for src, variants in GLOSSARY.items():
        pat = r"(?<![A-Za-z0-9])" + re.escape(src).replace(" ", r"[\s\-]+") + r"(?:s|es)?(?![A-Za-z0-9])"
        if re.search(pat, joined_src, flags=re.I):
            if not any(v in joined_tgt for v in variants):
                term_misses.append(f"{src} => {'/'.join(variants)}")
    if term_misses:
        errors.append("术语未按表翻译：" + "; ".join(term_misses[:6]))

    return {"errors": errors, "warnings": warnings, "term_misses": term_misses}


def _is_ascii_token_char(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in "._-%+µΩ")


def _word_boundaries(text: str) -> list[int]:
    if jieba is None:
        return [i for i in range(2, len(text) - 1)]
    out: list[int] = []
    pos = 0
    try:
        tokens = list(jieba.cut(text, HMM=False))
    except Exception:  # noqa: BLE001
        tokens = list(text)
    for tok in tokens:
        pos += len(tok)
        out.append(pos)
    return out


def polish_target(text: str) -> str:
    """Small deterministic cleanups for recurring Chinese grammar artifacts."""
    t = text.strip()
    t = re.sub(r"比值得控制", "比值控制", t)
    t = re.sub(r"固定的恒定", "固定不变的", t)
    t = re.sub(r"恒定固定", "固定不变", t)
    t = re.sub(r"输出电压电平", "输出电压", t)
    t = re.sub(r"固定电压电平的电源", "电压固定的电源", t)
    t = re.sub(r"名为\s*on\s*和\s*off\s*的引脚", "名为 on/off 的引脚", t, flags=re.I)
    t = re.sub(r"会始终", "总是会", t)
    t = re.sub(r"是种", "是一种", t)
    t = re.sub(r"([，。；！？、：])[，。；！？、：]+", r"\1", t)
    t = re.sub(r"\s+([，。；！？、：,;!?])", r"\1", t)
    return t.strip()


def wrap_target(text: str, max_line: int = MAX_LINE_CHARS) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    text = re.sub(r"\s+([，。；！？、：,;!?])", r"\1", text)
    text = re.sub(r"([，。；！？、：,;!?])\s+", r"\1", text)
    if len(text) <= max_line:
        return text

    n = len(text)
    target = n // 2
    punct = "，。；！？、：,;!?"
    dangling = set("的地得该这那其此每各把被将对为与和或而但就也都更最太很取使让给向于由")
    closing = "，。；！？、：,;!?）】”’"
    # 1) punctuation or Chinese word boundaries near the middle; reject breaks
    #    that strand a particle/determiner at the end of line 1
    raw = [i + 1 for i, ch in enumerate(text) if ch in punct]
    raw += _word_boundaries(text)
    bounds = []
    for b in raw:
        if not (4 <= b <= n - 4):
            continue
        if text[b - 1] in dangling:
            continue
        if text[b] in closing:
            continue
        if text[b] in "的地得了着过吗呢吧啊呀":
            continue
        if _is_ascii_token_char(text[b - 1]) and _is_ascii_token_char(text[b]):
            continue
        bounds.append(b)
    bounds = sorted(set(bounds))
    if not bounds:
        bounds = [b for b in _word_boundaries(text) if 4 <= b <= n - 4] or [target]
    # prefer a punctuation candidate if it is reasonably close to the middle
    punct_bounds = [b for b in bounds if text[b - 1] in punct
                    and abs(b - target) <= max_line // 2 + 2]
    if punct_bounds:
        split = min(punct_bounds, key=lambda x: (abs(x - target), -x))
    else:
        split = min(bounds, key=lambda x: abs(x - target))
    a, b = text[:split].strip(), text[split:].strip()
    if not a or not b:
        return text
    # if one side is very short, rebalance using the nearest usable boundary
    if len(b) < 5 or len(a) < 5:
        split = min(bounds, key=lambda x: abs(x - target))
        a, b = text[:split].strip(), text[split:].strip()
        if not a or not b:
            return text
    return a + "\n" + b


# --------------------------------------------------------------------------- chunking
def chunk_cues(cues: list[Cue], max_cues: int = MAX_SOURCE_CUES_PER_CHUNK,
               max_chars: int = MAX_SOURCE_CHARS_PER_CHUNK) -> list[dict]:
    chunks: list[dict] = []
    cur: list[Cue] = []
    cur_chars = 0
    for c in cues:
        ln = len(c.text)
        if cur and (len(cur) >= max_cues or cur_chars + ln > max_chars):
            chunks.append({"cues": cur, "context": []})
            cur, cur_chars = [], 0
        cur.append(c)
        cur_chars += ln
    if cur:
        chunks.append({"cues": cur, "context": []})
    # attach previous 2 lines as context
    for i, ch in enumerate(chunks):
        if i == 0:
            ch["context"] = []
        else:
            prev = chunks[i - 1]["cues"][-2:]
            ch["context"] = [p.text.replace("\n", " ") for p in prev]
    return chunks


# --------------------------------------------------------------------------- server
class LlamaServer:
    def __init__(self, model_path: Path = MT_MODEL_PATH, jobs: int = MT_JOBS, port: int = 18080):
        self.model_path = Path(model_path)
        self.jobs = jobs
        self.port = port
        self.proc: subprocess.Popen | None = None
        self.log_path = Path("/workspace/pyvideotrans/subtitle_pipeline/work/llama_server.log")
        self._client: httpx.AsyncClient | None = None
        self._raw_mode = False

    def _binary(self) -> Path:
        from config import LLAMA_DIR
        cands = sorted(LLAMA_DIR.rglob("llama-server"))
        if not cands:
            raise FileNotFoundError(f"llama-server not found under {LLAMA_DIR}")
        return cands[0]

    def start(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(self.log_path, "ab", buffering=0)
        args = [
            str(self._binary()),
            "-m", str(self.model_path),
            "-ngl", "99",
            "-c", str(4096 * self.jobs),
            "--parallel", str(self.jobs),
            "--cont-batching",
            "--jinja",
            "-ctk", "q8_0", "-ctv", "q8_0",
            "--cache-reuse", "256",
            "-t", "6",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--no-webui",
        ]
        log.info("starting llama-server: %s", " ".join(args))
        import os
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/workspace/tools/llama.cpp/cudart-llama-b12018-bin-ubuntu-cuda-12.8-x64:"
            "/workspace/tools/llama.cpp/llama-b12018:" + env.get("LD_LIBRARY_PATH", "")
        )
        self.proc = subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT,
                                     cwd=str(self._binary().parent), env=env)
        deadline = time.time() + 300
        last_err = None
        while time.time() < deadline:
            if self.proc.poll() is not None:
                tail = self._tail_log()
                raise RuntimeError(f"llama-server exited early:\n{tail}")
            try:
                r = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    log.info("llama-server ready after %.1fs", 300 - (deadline - time.time()))
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            time.sleep(1.0)
        raise RuntimeError(f"llama-server not ready: {last_err}\n{self._tail_log()}")

    def _tail_log(self, n: int = 40) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception:  # noqa: BLE001
            return "<no log>"

    async def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{self.port}",
                timeout=httpx.Timeout(600.0, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    async def _post(self, path: str, payload: dict, retries: int = 3) -> dict:
        c = await self.client()
        last = None
        for k in range(retries):
            try:
                r = await c.post(path, json=payload)
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                last = e
                await asyncio.sleep(1.0 + k * 2)
        raise RuntimeError(f"llama request failed: {last}")

    async def chat(self, prompt: str, temperature: float, max_tokens: int = 2048) -> str:
        if not self._raw_mode:
            try:
                data = await self._post("/v1/chat/completions", {
                    "model": MT_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": TOP_P,
                    "max_tokens": max_tokens,
                    "repeat_penalty": REPEAT_PENALTY,
                })
                ch = data.get("choices", [{}])[0]
                content = (ch.get("message") or {}).get("content", "")
                if content and content.strip():
                    return content.strip()
                log.warning("empty chat response, falling back to raw completion")
                self._raw_mode = True
            except Exception as e:  # noqa: BLE001
                log.warning("chat endpoint failed (%s), falling back to raw completion", e)
                self._raw_mode = True
        # raw completion using the model's own prompt format
        raw = "<|startoftext|>" + prompt + "<|extra_0|>"
        data = await self._post("/completion", {
            "prompt": raw,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": TOP_P,
            "repeat_penalty": REPEAT_PENALTY,
            "stop": ["<|eos|>", "<|extra_4|>", "<|startoftext|>"],
            "cache_prompt": True,
        })
        return (data.get("content") or "").strip()


# --------------------------------------------------------------------------- translation
class Translator:
    def __init__(self, server: LlamaServer, title: str, sem: asyncio.Semaphore | None = None):
        self.server = server
        self.title = title
        self.sem = sem or asyncio.Semaphore(MT_JOBS)
        self.stats = {"chunks": 0, "retries": 0, "split_fallbacks": 0, "review_kept": 0, "errors": 0}

    def _cache_key(self, sources: list[str], pass_name: str) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(PROMPT_VERSION.encode())
        h.update(MT_MODEL_NAME.encode())
        h.update(pass_name.encode())
        h.update(self.title.encode())
        for s in sources:
            h.update(b"\x00")
            h.update(s.encode())
        return h.hexdigest()[:32]

    async def _call(self, prompt: str, temperature: float) -> str:
        return await self.server.chat(prompt, temperature=temperature)

    async def _translate_once(self, sources: list[str], context: list[str],
                              extra: str, temperature: float) -> list[str]:
        prompt = build_translate_prompt(self.title, context, sources, extra)
        raw = await self._call(prompt, temperature)
        parsed = parse_marked(raw, len(sources))
        if parsed is None:
            raise ValueError("无法解析输出标记")
        return parsed

    async def translate_chunk(self, chunk: dict, on_progress=None) -> list[str]:
        sources = [c.text.replace("\n", " ").strip() for c in chunk["cues"]]
        context = chunk.get("context", [])
        self.stats["chunks"] += 1
        key = self._cache_key(sources, "translate")
        cache_file = MT_CACHE / f"{key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                if data.get("sources") == sources:
                    return data["targets"]
            except Exception:  # noqa: BLE001
                pass

        async with self.sem:
            targets, issues = await self._translate_with_retry(sources, context)
            # ---- review pass
            if targets:
                try:
                    reviewed = await self._review(sources, targets, issues)
                    if reviewed:
                        targets = reviewed
                        self.stats["review_kept"] += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("review failed: %s", e)
            if targets:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(json.dumps({
                    "sources": sources, "targets": targets, "issues": issues,
                }, ensure_ascii=False), encoding="utf-8")
        if on_progress:
            on_progress()
        return targets or ["" for _ in sources]

    async def _translate_with_retry(self, sources: list[str], context: list[str],
                                    depth: int = 0) -> tuple[list[str] | None, dict]:
        issues = {"errors": [], "warnings": [], "term_misses": []}
        extra = ""
        for attempt in range(3):
            try:
                temp = TEMP_TRANSLATE if attempt == 0 else TEMP_RETRY
                targets = await self._translate_once(sources, context, extra, temp)
                issues = check_chunk(sources, targets)
                if not issues["errors"]:
                    if issues["warnings"]:
                        log.debug("warnings: %s", issues["warnings"][:3])
                    return targets, issues
                self.stats["retries"] += 1
                log.warning("chunk check failed (%d/%d): %s", attempt + 1, 3,
                            "; ".join(issues["errors"][:3]))
                extra = "上一次输出存在以下问题，请修正后重新输出：\n" + "\n".join(issues["errors"][:8])
            except Exception as e:  # noqa: BLE001
                self.stats["retries"] += 1
                log.warning("chunk attempt %d failed: %s", attempt + 1, e)
                extra = "请严格按编号逐条输出，不要遗漏或合并任何一条。"

        # last resort: split the chunk
        if len(sources) > 1 and depth < 3:
            self.stats["split_fallbacks"] += 1
            mid = len(sources) // 2
            log.warning("splitting chunk of %d into %d+%d", len(sources), mid, len(sources) - mid)
            left, _ = await self._translate_with_retry(sources[:mid], context + sources[max(0, mid - 2):mid], depth + 1)
            right, _ = await self._translate_with_retry(sources[mid:], context + sources[max(0, mid - 2):mid], depth + 1)
            if left and right:
                return left + right, issues

        # final: one cue at a time
        out: list[str] = []
        for i, s in enumerate(sources):
            t = ""
            for attempt in range(2):
                try:
                    res = await self._translate_once([s], context + sources[max(0, i - 2):i],
                                                     "", TEMP_RETRY)
                    if res and res[0].strip():
                        t = res[0]
                        break
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(1)
            out.append(t)
        self.stats["errors"] += sum(1 for x in out if not x)
        return out, issues

    async def _review(self, sources: list[str], drafts: list[str], issues: dict) -> list[str] | None:
        extra = ""
        if issues.get("errors"):
            extra = "初译存在这些问题：" + "；".join(issues["errors"][:6])
        prompt = build_review_prompt(self.title, sources, drafts, extra)
        raw = await self._call(prompt, TEMP_REVIEW)
        revised = parse_marked(raw, len(sources))
        if not revised:
            return None
        chk = check_chunk(sources, revised)
        draft_chk = check_chunk(sources, drafts)
        # do not accept a review that introduces more errors or rewrites everything
        if len(chk["errors"]) > len(draft_chk["errors"]):
            return None
        import difflib
        ratio = difflib.SequenceMatcher(None, "".join(drafts), "".join(revised)).ratio()
        if ratio < REVIEW_MIN_RATIO and not draft_chk["errors"]:
            log.warning("review rejected (rewrite ratio %.2f)", ratio)
            return None
        return revised
