# -*- coding: utf-8 -*-
"""Audio extraction + faster-whisper transcription with cache and progress."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path

from tqdm import tqdm

from config import ASR_CACHE, ASR_MODEL_DIR, ASR_MODEL_NAME, AUDIO_DIR, HOTWORDS, INITIAL_PROMPT

log = logging.getLogger("asr")


def _sig(video: Path) -> str:
    st = video.stat()
    payload = f"{video.name}|{st.st_size}|{int(st.st_mtime)}|{ASR_MODEL_NAME}|whisper-cache-v1"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def extract_audio(video: Path, force: bool = False) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    wav = AUDIO_DIR / (video.stem + ".wav")
    if wav.exists() and wav.stat().st_size > 1000 and not force:
        return wav
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(wav),
    ]
    t0 = time.time()
    subprocess.run(cmd, check=True)
    log.info("audio extracted in %.1fs -> %s", time.time() - t0, wav)
    return wav


def load_model(device: str = "cuda"):
    from faster_whisper import WhisperModel

    attempts = [
        ("cuda", "float16"),
        ("cuda", "int8_float16"),
        ("cpu", "int8"),
    ]
    last = None
    for dev, ct in attempts:
        try:
            log.info("loading faster-whisper %s on %s/%s", ASR_MODEL_NAME, dev, ct)
            model = WhisperModel(
                str(ASR_MODEL_DIR), device=dev, compute_type=ct,
                cpu_threads=6, num_workers=1,
            )
            return model
        except Exception as e:  # noqa: BLE001
            last = e
            log.warning("ASR load failed on %s/%s: %s", dev, ct, e)
    raise RuntimeError(f"cannot load ASR model: {last}")


def transcribe_file(
    video: Path,
    model,
    force: bool = False,
    show_progress: bool = True,
    position: int = 0,
) -> dict:
    ASR_CACHE.mkdir(parents=True, exist_ok=True)
    cache = ASR_CACHE / f"{video.stem}.json"
    sig = _sig(video)
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("sig") == sig:
                log.info("ASR cache hit: %s", video.name)
                return data
        except Exception:  # noqa: BLE001
            pass

    wav = extract_audio(video, force=force)
    t0 = time.time()
    kwargs = dict(
        language="en",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=[0.0, 0.2, 0.4],
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=100),
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=INITIAL_PROMPT,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    )
    try:
        segments_iter, info = model.transcribe(str(wav), **kwargs)
    except TypeError:
        segments_iter, info = model.transcribe(str(wav), **kwargs)

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    bar = tqdm(
        total=duration or None, desc=f"ASR {video.stem[:28]}", unit="s",
        position=position, leave=False, dynamic_ncols=True,
        disable=not show_progress,
    )
    segments: list[dict] = []
    last_end = 0.0
    try:
        for seg in segments_iter:
            words = []
            for w in (seg.words or []):
                words.append({
                    "start": round(float(w.start), 3),
                    "end": round(float(w.end), 3),
                    "word": w.word,
                    "prob": round(float(getattr(w, "probability", 0.0) or 0.0), 4),
                })
            segments.append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text,
                "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0) or 0.0), 4),
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0) or 0.0), 4),
                "compression_ratio": round(float(getattr(seg, "compression_ratio", 0.0) or 0.0), 4),
                "words": words,
            })
            if show_progress and duration:
                p = min(max(seg.end, last_end), duration)
                bar.update(max(0.0, p - last_end))
                last_end = p
    finally:
        if show_progress:
            if duration:
                bar.update(max(0.0, duration - last_end))
            bar.close()

    data = {
        "sig": sig,
        "file": str(video),
        "model": ASR_MODEL_NAME,
        "duration": duration,
        "language": getattr(info, "language", "en"),
        "language_probability": round(float(getattr(info, "language_probability", 0.0) or 0.0), 4),
        "segments": segments,
        "elapsed": round(time.time() - t0, 2),
    }
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info(
        "ASR done in %.1fs: %s (%d segments, %d words)",
        time.time() - t0, video.name, len(segments),
        sum(len(s["words"]) for s in segments),
    )
    return data
