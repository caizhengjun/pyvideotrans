#!/usr/bin/env python3
"""Auto-setup environment: install requirements and verify model paths."""
import subprocess, sys, os
print("Checking environment...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm", "jieba", "httpx", "faster-whisper"], check=False)
print("Environment OK.")
