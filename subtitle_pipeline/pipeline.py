# -*- coding: utf-8 -*-
"""End-to-end subtitle translation pipeline.

Usage:
  python pipeline.py --pilot            # quick test: file 001, first 25 cues
  python pipeline.py                    # all six videos
  python pipeline.py --only 003         # one file
  python pipeline.py --asr-only         # transcribe + align only
  python pipeline.py --mt-only          # reuse cached ASR, translate only
  python pipeline.py --force            # ignore caches
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

import config as C
from align import align
from asr import load_model, transcribe_file
from mt import LlamaServer, Translator, check_chunk, chunk_cues, polish_target, wrap_target
from srt_utils import Cue, parse_srt, write_srt

log = logging.getLogger("pipeline")


def setup_logging() -> None:
    C.WORK.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(C.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def find_files(only: str | None) -> list[Path]:
    files = sorted(C.SRC_DIR.glob("*_en.srt"))
    if only:
        files = [f for f in files if only in f.name]
    if not files:
        raise SystemExit(f"no *_en.srt found in {C.SRC_DIR}")
    return files


def base_and_title(srt: Path) -> tuple[str, str]:
    stem = srt.stem
    base = stem[:-3] if stem.endswith("_en") else stem
    m = re.search(r"[（(]([^（）()]*)[）)]\s*$", base)
    title = m.group(1).strip() if m else base
    return base, title


def video_for(srt: Path) -> Path | None:
    stem = srt.stem[:-3] if srt.stem.endswith("_en") else srt.stem
    for ext in (".mp4", ".mkv", ".webm", ".mov", ".avi"):
        p = srt.with_name(stem + ext)
        if p.exists():
            return p
    return None


def run_asr(files: list[Path], force: bool) -> dict[str, dict]:
    model = load_model()
    out: dict[str, dict] = {}
    bar = tqdm(files, desc="ASR", unit="file", ncols=80, ascii=True, dynamic_ncols=False, leave=False, bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}")
    for i, srt in enumerate(bar):
        video = video_for(srt)
        if not video:
            log.error("no video next to %s, skip ASR", srt.name)
            continue
        # no postfix (prevents line wrap from long filenames)
        try:
            out[srt.name] = transcribe_file(video, model, force=force, position=1)
        except Exception as e:  # noqa: BLE001
            log.exception("ASR failed for %s: %s", srt.name, e)
            raise
    bar.close()
    return out


def load_or_align(srt: Path, asr_data: dict, force: bool) -> tuple[list[Cue], dict]:
    cache = C.CACHE / "aligned" / f"{srt.stem}.json"
    orig_sig = f"{srt.stat().st_size}:{int(srt.stat().st_mtime)}"
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (data.get("orig_sig") == orig_sig and data.get("asr_sig") == asr_data.get("sig")
                    and data.get("align_version") == C.ALIGN_VERSION):
                cues = [Cue(c["start"], c["end"], c["text"], c.get("index", i + 1),
                            c.get("flags", []), c.get("orig_index")) for i, c in enumerate(data["cues"])]
                log.info("align cache hit: %s", srt.name)
                return cues, data.get("stats", {})
        except Exception:  # noqa: BLE001
            pass
    orig = parse_srt(srt)
    cues, stats = align(asr_data, orig, duration=float(asr_data.get("duration", 0)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "orig_sig": orig_sig, "asr_sig": asr_data.get("sig"),
        "align_version": C.ALIGN_VERSION, "stats": stats,
        "cues": [{"start": c.start, "end": c.end, "text": c.text, "index": c.index,
                  "flags": c.flags, "orig_index": c.orig_index} for c in cues],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return cues, stats


async def translate_all(prepared: list[dict], server: LlamaServer, jobs: int,
                        limit: int | None) -> None:
    total = 0
    for f in prepared:
        chunks = chunk_cues(f["cues"])
        if limit:
            chunks = chunks[: max(1, (limit + C.MAX_SOURCE_CUES_PER_CHUNK - 1) // C.MAX_SOURCE_CUES_PER_CHUNK)]
            for ch in chunks:
                ch["cues"] = ch["cues"][:limit]
        f["chunks"] = chunks
        total += len(chunks)

    bar = tqdm(total=total, desc="MT", unit="chunk", ncols=80, ascii=True, dynamic_ncols=False, position=0, leave=False, bar_format="{l_bar}{bar} {n_fmt}/{total_fmt}")
    sem = asyncio.Semaphore(jobs)

    async def one(f: dict, ch: dict) -> None:
        tr = Translator(server, f["title"], sem=sem)
        try:
            targets = await tr.translate_chunk(ch)
        except Exception as e:  # noqa: BLE001
            log.exception("chunk failed in %s: %s", f["srt"].name, e)
            targets = ["" for _ in ch["cues"]]
        for c, t in zip(ch["cues"], targets):
            f["targets"][c.index] = t
        for k, v in tr.stats.items():
            f["stats"][k] = f["stats"].get(k, 0) + v
        bar.update(1)
        done = sum(1 for x in f["targets"].values() if x)
        # minimal postfix
        bar.set_postfix_str("chunk")

    tasks = []
    for f in prepared:
        f["targets"] = {}
        for ch in f["chunks"]:
            tasks.append(asyncio.create_task(one(f, ch)))
    await asyncio.gather(*tasks)
    bar.close()


def write_outputs(prepared: list[dict], limit: int | None) -> None:
    for f in prepared:
        cues: list[Cue] = f["cues"]
        en_cues = [Cue(c.start, c.end, c.text, c.index, list(c.flags), c.orig_index) for c in cues]
        out: list[Cue] = []
        for c in cues:
            t = f["targets"].get(c.index, "").strip()
            if not t:
                t = c.text
                c.flags.append("untranslated")
            c.text = wrap_target(polish_target(t))
            out.append(c)
        if limit:
            out = out[:limit]
        dst = f["out_path"]
        write_srt(out, dst)
        log.info("wrote %s (%d cues)", dst, len(out))
        f["final_cues"] = out

        en_path = C.CORRECTED_DIR / f"{f['base']}_en_corrected.srt"
        write_srt(en_cues, en_path)
        log.info("wrote corrected English %s", en_path)


def qa_report(prepared: list[dict], limit: int | None) -> None:
    lines = ["# 字幕翻译 QA 报告", "", f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 文件 | 原条数 | 新条数 | ASR词数 | 低置信 | 回退 | 分块 | 重试 | 错误 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for f in prepared:
        st = f["stats"]
        s = f["align_stats"]
        lines.append(
            f"| {f['base']} | {s.get('orig_cues','')} | {len(f['final_cues'])} | "
            f"{s.get('asr_words','')} | {s.get('low_conf',0)} | {s.get('fallback',0)} | "
            f"{st.get('chunks',0)} | {st.get('retries',0)} | {st.get('errors',0)} |"
        )
    lines.append("")
    lines.append("## 警告明细")
    for f in prepared:
        warns: list[str] = []
        errs: list[str] = []
        for ch in f["chunks"]:
            src = [c.text.replace("\n", " ").strip() for c in ch["cues"]]
            tgt = [f["targets"].get(c.index, "") for c in ch["cues"]]
            chk = check_chunk(src, tgt)
            for w in chk["warnings"]:
                warns.append(f"第{ch['cues'][0].index}条起：{w}")
            for e in chk["errors"]:
                errs.append(f"第{ch['cues'][0].index}条起：{e}")
        lines.append(f"### {f['base']}")
        if not warns and not errs:
            lines.append("无")
        for e in errs[:30]:
            lines.append(f"- 错误：{e}")
        for w in warns[:30]:
            lines.append(f"- 警告：{w}")
        lines.append("")
    C.WORK.mkdir(parents=True, exist_ok=True)
    (C.WORK / "qa_report.md").write_text("\n".join(lines), encoding="utf-8")
    (C.WORK / "run_stats.json").write_text(json.dumps(
        [{k: f[k] for k in ("base", "stats", "align_stats")} for f in prepared],
        ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("QA report: %s", C.WORK / "qa_report.md")


async def main_async(args) -> None:
    files = find_files(args.only)
    log.info("files: %s", [f.name for f in files])

    if args.asr_only:
        asr_data = run_asr(files, force=args.force)
        for srt in files:
            if srt.name in asr_data:
                cues, stats = load_or_align(srt, asr_data[srt.name], force=True)
                base, _ = base_and_title(srt)
                write_srt(cues, C.CORRECTED_DIR / f"{base}_en_corrected.srt")
                log.info("%s: %s", srt.name, stats)
        return

    # cache ASR results to disk so --mt-only can run without the ASR model
    asr_files = {}
    for srt in files:
        p = C.ASR_CACHE / f"{srt.stem}.json"
        if p.exists():
            try:
                asr_files[srt.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                pass
    if not args.mt_only or len(asr_files) < len(files):
        asr_data = run_asr(files, force=args.force)
        asr_files.update(asr_data)

    prepared = []
    for srt in files:
        data = asr_files.get(srt.name)
        if not data:
            log.error("no ASR data for %s, skip", srt.name)
            continue
        cues, stats = load_or_align(srt, data, force=args.force)
        base, title = base_and_title(srt)
        out_path = (C.WORK / f"pilot_{base}_zh.srt") if args.pilot else (C.SRC_DIR / f"{base}_zh.srt")
        prepared.append({
            "srt": srt, "base": base, "title": title, "cues": cues,
            "align_stats": stats, "out_path": out_path, "stats": {}, "targets": {},
        })
        log.info("%s: %d cues, title=%s", srt.name, len(cues), title)

    if args.pilot:
        for f in prepared:
            f["cues"] = f["cues"][: max(1, args.limit)]

    server = LlamaServer(jobs=args.jobs, port=args.port)
    t0 = time.time()
    try:
        server.start()
        # smoke test
        try:
            smoke = await server.chat("把下面这句话翻译成简体中文，只输出译文：The buck converter is efficient.", 0.1, 64)
            log.info("smoke test: %s", smoke.replace("\n", " ")[:160])
        except Exception as e:  # noqa: BLE001
            log.error("smoke test failed: %s", e)
            raise
        await translate_all(prepared, server, args.jobs, args.limit if args.pilot else None)
    finally:
        await server.close()
        if not args.keep_server:
            server.stop()
    log.info("translation finished in %.1fs", time.time() - t0)

    write_outputs(prepared, args.limit if args.pilot else None)
    qa_report(prepared, args.limit if args.pilot else None)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="English SRT -> Chinese SRT with ASR repair + Hy-MT2")
    ap.add_argument("--only", default=None, help="only file names containing this string")
    ap.add_argument("--pilot", action="store_true", help="write output under work/ and only first --limit cues")
    ap.add_argument("--limit", type=int, default=25, help="cue limit for pilot")
    ap.add_argument("--force", action="store_true", help="ignore caches")
    ap.add_argument("--jobs", type=int, default=C.MT_JOBS, help="concurrent MT requests / server slots")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--keep-server", action="store_true", help="leave llama-server running")
    ap.add_argument("--asr-only", action="store_true", help="only ASR + alignment")
    ap.add_argument("--mt-only", action="store_true", help="reuse cached ASR, skip ASR")
    args = ap.parse_args()
    if args.limit <= 0:
        args.limit = 10**9
    C.WORK.mkdir(parents=True, exist_ok=True)
    C.CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    C.MT_CACHE.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
