#!/usr/bin/env python3
"""Usage: python quick.py /path/to/video_dir
Requires directory with .mp4 + *_en.srt; outputs *_zh.srt + corrected_en/"""
import sys, subprocess, os
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python quick.py /path/to/video_dir")
    sys.exit(1)
dir_path = sys.argv[1]
env = os.environ.copy()
env["VIDEO_DIR"] = dir_path
print(f"▶ Running pipeline on: {dir_path}")
# Auto-free GPU memory if previous server holds it (prevent CUDA OOM on ASR)
os.system("fuser -k 18080/tcp 2>/dev/null; sleep 1")
subprocess.run([
    "/workspace/pyvideotrans/.venv/bin/python", "pipeline.py",
    "--force", "--jobs", "6", "--keep-server"
], cwd="/workspace/subtitle_pipeline", env=env)
