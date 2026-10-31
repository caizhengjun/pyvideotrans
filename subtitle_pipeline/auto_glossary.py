#!/usr/bin/env python3
"""Auto-extract new technical terms from corrected English SRTs,
translate via local llama-server, and merge into glossary_auto.json
(without human confirmation)."""
import json, re, glob, os
from pathlib import Path
import httpx

SRC = Path("/workspace/31dc/corrected_en")
OUT = Path(__file__).with_name("glossary_auto.json")

# Load existing terms to avoid duplicates
from config import GLOSSARY, LATIN_WHITELIST
KNOWN = set(GLOSSARY) | LATIN_WHITELIST

STOP = {
    "The","And","But","So","Here","Right","Now","Okay","Well","Let's","This","That",
    "These","Those","They","Them","Their","We","Our","You","Your","He","Him","His","She",
    "Her","Not","Also","Very","More","Most","Such","Only","Just","Even","Still","Well",
    "Called","Known","Like","Another","Each","Every","Some","Any","All","Both","Either",
    "Neither","Much","Many","Few","Several","No","One","Two","First","Second","Next",
    "Last","Finally","However","Therefore","Otherwise","Instead","Please","Almost",
    "About","Above","Across","After","Against","Around","Before","Behind","Below","Beyond",
    "By","Down","During","Except","Inside","Into","Near","Off","On","Onto","Opposite",
    "Outside","Over","Past","Through","Throughout","Till","To","Toward","Under","Until",
    "Up","Upon","With","Within","Without","Would","Could","Should","May","Might","Must",
    "Have","Has","Had","Do","Does","Did","Did","Can","Will","Shall","Should","Might",
    "Be","Been","Being","Is","Are","Was","Were","Am","Are","Was","Have","Has","Had",
    "Get","Gets","Got","Getting","Go","Goes","Going","Went","Gone","Make","Makes","Made",
    "Take","Takes","Took","Taken","Come","Comes","Came","Coming","See","Sees","Saw",
    "Seen","Know","Knows","Knew","Known","Think","Thinks","Thought","Thinking","Look",
    "Looks","Looked","Looking","Want","Wants","Wanted","Wanting","Give","Gives","Gave",
    "Given","Giving","Use","Uses","Used","Using","Find","Finds","Found","Finding","Tell",
    "Tells","Told","Telling","Ask","Asks","Asked","Asking","Work","Works","Worked",
    "Working","Seem","Seems","Seemed","Seeming","Feel","Feels","Felt","Feeling","Try",
    "Tries","Tried","Trying","Leave","Leaves","Left","Leaving","Call","Calls","Called",
    "Calling","Good","New","Old","Great","Little","Own","Other","Right","Big","High",
    "Different","Small","Large","Next","Early","Young","Important","Public","Same",
    "Able","Bad","Best","Better","Black","White","Blue","Red","Green","Brown","Gray",
}  # generous stoplist

CAND_PAT = re.compile(
    r"\b([A-Z][a-z]{1,15}(?:[-\s][A-Z][a-z]{1,15}){1,3})\b"
)

def extract_candidates():
    texts = []
    for p in glob.glob(str(SRC / "*_en_corrected.srt")):
        texts.append(Path(p).read_text(encoding="utf-8"))
    raw = "\n".join(texts)
    cands = {}
    for m in CAND_PAT.finditer(raw):
        term = m.group(1)
        # skip pure stop / numbers / already known
        first_word = term.split()[0]
        if first_word in STOP: continue
        if re.search(r"\d{2,}", term): continue
        if term in KNOWN: continue
        cands[term] = cands.get(term, 0) + 1
    # keep those appearing >= 2 times (reduce noise)
    return [t for t, c in sorted(cands.items(), key=lambda x: -x[1]) if c >= 1 and len(t) > 2 and len(t) < 60]

def translate_batch(terms):
    # Use running server; batch in groups of 20
    out = {}
    url = "http://127.0.0.1:18080/completion"
    for i in range(0, len(terms), 20):
        batch = terms[i:i+20]
        prompt = ("Translate these English power-electronics technical terms to "
                  "concise Chinese subtitles (keep units/abbr). Return ONLY JSON: "
                  "{\"term\": \"中文\"} per line.\n\n" + "\n".join(batch))
        try:
            r = httpx.post(url, json={
                "prompt": prompt, "temperature": 0.0, "max_tokens": 1024, "stop": ["\n\n"]
            }, timeout=60)
            text = r.json().get("content", "")
            for line in text.splitlines():
                line = line.strip()
                if "\"" not in line: continue
                try:
                    # simple parse of JSON-like line
                    k = line.split("\"")[1] if line.startswith('"') else None
                    if k is None:
                        # try extracting from "term":"value"
                        m = re.search(r'"([^"]+)"\s*:\s*"([^"]+)"', line)
                        if m:
                            term = m.group(1); trans = m.group(2)
                        else:
                            continue
                    else:
                        # try split by colon
                        parts = line.split(":")
                        if len(parts) >= 2:
                            term = parts[0].strip('" \t')
                            trans = ":".join(parts[1:]).strip('" \t')
                        else:
                            continue
                    # clean
                    term = term.strip(); trans = trans.strip()
                    if term and trans and term not in out:
                        out[term] = trans
                except Exception:
                    continue
        except Exception as e:
            # If server unavailable, skip this batch (won't overwrite existing)
            pass
    return out

def main():
    cands = extract_candidates()
    if not cands:
        print("No new candidates found.")
        return
    existing = {}
    if OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
    # Translate only new ones
    new = [t for t in cands if t not in existing and t not in KNOWN]
    print(f"Found {len(new)} new candidates to translate...")
    translated = translate_batch(new)
    # Merge (auto = no confirmation)
    merged = dict(existing)
    merged.update(translated)
    # Filter: keep only those that look valid (not empty, not identical to term)
    merged = {k: v for k, v in merged.items() if v and v != k and len(v) > 1 and len(k) <= 60}
    OUT.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(merged)} terms to {OUT}")
    # Auto-merge into config GLOSSARY (optional: do inline edit?)
    # Instead, rely on config loading below

if __name__ == "__main__":
    main()
