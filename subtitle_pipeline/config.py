# -*- coding: utf-8 -*-
"""Configuration: paths, glossary, prompt templates."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path("/workspace/pyvideotrans/subtitle_pipeline")
SRC_DIR = Path(os.environ.get("VIDEO_DIR", "/workspace/31dc"))
WORK = ROOT / "work"
CACHE = WORK / "cache"
ASR_CACHE = CACHE / "asr"
MT_CACHE = CACHE / "mt"
FINAL_CACHE = CACHE / "final"
AUDIO_DIR = WORK / "audio"
CORRECTED_DIR = SRC_DIR / "corrected_en"
LOG_FILE = WORK / "pipeline.log"

ASR_MODEL_DIR = Path("/workspace/models/faster-whisper-large-v3")
MT_MODEL_PATH = Path("/workspace/models/Hy-MT2-7B-GGUF/HY-MT2-7B-Q8_0.gguf")
LLAMA_DIR = Path("/workspace/tools/llama.cpp")

ASR_MODEL_NAME = "large-v3"
MT_MODEL_NAME = "Hy-MT2-7B-Q8_0"
PROMPT_VERSION = "v6"
ALIGN_VERSION = "a4"

# Domains for the six videos in /workspace/31dc
DOMAIN = "开关电源 / 电力电子基础课程"
INITIAL_PROMPT = (
    "A power electronics lecture about switching voltage regulators, buck converters, "
    "boost converters, linear regulators, duty cycle, inductors, capacitors, MOSFETs, "
    "bipolar junction transistors, diodes, Zener diodes, ripple, efficiency, simulation, "
    "oscilloscope measurements and datasheets."
)
HOTWORDS = [
    "switching regulator", "buck converter", "boost converter", "linear regulator",
    "duty cycle", "duty ratio", "inductor", "capacitor", "MOSFET", "BJT",
    "bipolar junction transistor", "diode", "Zener diode", "Schottky diode",
    "ripple", "efficiency", "datasheet", "simulation", "op-amp", "operational amplifier",
    "feedback", "voltage divider", "reference voltage", "back EMF", "freewheeling diode",
    "continuous conduction mode", "discontinuous conduction mode", "switching frequency",
    "on-resistance", "switching loss", "conduction loss", "millivolt", "milliohm",
    "microsecond", "nanosecond", "kilohertz", "megahertz", "drain", "source", "gate",
    "collector", "emitter", "base", "oscilloscope", "probe", "waveform",
]

# source term -> accepted Chinese variants. First variant is the canonical one.
try:
    from pathlib import Path
    _AUTO_GLOSSARY = Path(__file__).with_name("glossary_auto.json")
    if _AUTO_GLOSSARY.exists():
        _AUTO_DICT = json.load(open(_AUTO_GLOSSARY, encoding="utf-8"))
    else:
        _AUTO_DICT = {}
except Exception:
    _AUTO_DICT = {}

GLOSSARY: dict[str, list[str]] = {
    "switching regulator": ["开关稳压器"],
    "linear regulator": ["线性稳压器"],
    "voltage regulator": ["稳压器"],
    "regulator": ["稳压器"],
    "buck converter": ["降压转换器"],
    "boost converter": ["升压转换器"],
    "buck-boost converter": ["升降压转换器"],
    "flyback converter": ["反激式转换器"],
    "forward converter": ["正激式转换器"],
    "inverting converter": ["反相转换器"],
    "converter": ["转换器"],
    "duty cycle": ["占空比"],
    "duty ratio": ["占空比"],
    "inductor": ["电感"],
    "capacitor": ["电容"],
    "output capacitor": ["输出电容"],
    "input capacitor": ["输入电容"],
    "MOSFET": ["MOSFET"],
    "BJT": ["双极型晶体管"],
    "bipolar junction transistor": ["双极型晶体管"],
    "transistor": ["晶体管"],
    "diode": ["二极管"],
    "Zener diode": ["齐纳二极管", "稳压二极管"],
    "Schottky diode": ["肖特基二极管"],
    "freewheeling diode": ["续流二极管"],
    "catch diode": ["续流二极管"],
    "ripple": ["纹波"],
    "efficiency": ["效率"],
    "datasheet": ["数据手册"],
    "simulation": ["仿真"],
    "op-amp": ["运算放大器", "运放"],
    "operational amplifier": ["运算放大器", "运放"],
    "feedback": ["反馈"],
    "voltage divider": ["分压电路", "分压器"],
    "reference voltage": ["参考电压"],
    "back EMF": ["反电动势"],
    "switching frequency": ["开关频率"],
    "on-resistance": ["导通电阻"],
    "switching loss": ["开关损耗"],
    "conduction loss": ["导通损耗"],
    "continuous conduction mode": ["连续导通模式"],
    "discontinuous conduction mode": ["断续导通模式"],
    "voltage": ["电压"],
    "current": ["电流"],
    "load": ["负载"],
    "power supply": ["电源"],
    "battery": ["电池"],
    "waveform": ["波形"],
    "oscilloscope": ["示波器"],
    "probe": ["探头"],
    "drain": ["漏极"],
    "gate": ["栅极", "门极"],
    "collector": ["集电极"],
    "emitter": ["发射极"],
    "PWM": ["PWM"],
    "CCM": ["CCM"],
    "DCM": ["DCM"],
    "ESR": ["ESR"],
    "millivolt": ["毫伏"],
    "milliohm": ["毫欧"],
    "microsecond": ["微秒"],
    "nanosecond": ["纳秒"],
    "kilohertz": ["千赫兹"],
    "megahertz": ["兆赫兹"],
    "step-up": ["升压"],
    "step-down": ["降压"],
    "IC": ["IC", "芯片", "集成电路"],
    "integrated circuit": ["集成电路"],
    "resistor": ["电阻"],
    "variable resistor": ["可变电阻"],
    "voltage drop": ["压降"],
    "forward voltage": ["正向压降"],
    "oscillator": ["振荡器"],
    "frequency": ["频率"],
    "pin": ["引脚", "管脚"],
    "ground": ["接地", "地"],
    "power consumption": ["功耗"],
    "compensation": ["补偿"],
    "switch": ["开关"],
    "switching element": ["开关元件"],
    "microcontroller": ["微控制器"],
    "rectifier": ["整流器"],
    "capacitance": ["电容"],
    "fixed constant regulated voltage": ["固定不变的稳定电压"],
    "regulated voltage": ["稳定电压", "稳压"],
    "voltage level": ["电压电平", "电压等级"],
    "switching device": ["开关器件"],
    "power source": ["电源"],
    "fixed voltage": ["固定电压"],
    "linear and switching converters": ["线性稳压器和开关稳压器"],
    "linear converter": ["线性稳压器"],
    "switching converter": ["开关稳压器"],
}
# Merge auto-detected terms (no manual confirmation needed)
GLOSSARY = {**GLOSSARY, **_AUTO_DICT}

# acronyms / part numbers that may legitimately remain in Latin letters
LATIN_WHITELIST = {
    "MOSFET", "MOSFETs", "BJT", "IGBT", "PWM", "DC", "AC", "ESR", "EMI", "EMC",
    "PCB", "IC", "ICs", "LED", "CCM", "DCM", "ADC", "DAC", "PID", "RMS", "CPU",
    "USB", "OK", "NC", "NO", "mV", "mA", "mW", "kW", "kHz", "MHz", "Hz", "kΩ",
    "Ω", "V", "A", "W", "F", "H", "s", "ms", "us", "ns", "µF", "nF", "pF",
    "µH", "mH", "mΩ", "°C", "DC-DC", "AC-DC", "VIN", "VOUT", "N-channel",
    "P-channel", "R1", "R2", "R3", "R4", "C1", "C2", "C3", "L1", "L2",
    "D1", "D2", "Q1", "Q2", "LM317", "LM2596", "NE555", "IC", "ICs",
}

MAX_CUE_CHARS = 48          # Chinese characters per cue before warning
MAX_LINE_CHARS = 20         # wrap target
MAX_SOURCE_CUES_PER_CHUNK = 16
MAX_SOURCE_CHARS_PER_CHUNK = 1000
REVIEW_MIN_RATIO = 0.35     # reject review if it rewrites more than this

# translation sampling
TEMP_TRANSLATE = 0.25
TEMP_RETRY = 0.0
TEMP_REVIEW = 0.1
TOP_P = 0.9
REPEAT_PENALTY = 1.05

MT_JOBS = int(os.environ.get("MT_JOBS", "4"))
ASR_WORKERS = int(os.environ.get("ASR_WORKERS", "2"))
MAX_RETRIES = 3
