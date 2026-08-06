"""Merge the pitch diagnostic + correction logs into one JSON the table generator
reads, keyed by (voice, start-second). Parses the printed tables so the data in
MIX_TABLE always matches what was actually measured/applied."""
import json, re, sys
from pathlib import Path
TASKS = Path("/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/tasks")
DIAG = TASKS/"b21plu59a.output"          # trustworthy diagnostic
CORR_LOG = Path("/tmp/correct.log")      # correction run
out = {}

def tsec(m, s): return int(m)*60 + float(s)

# --- diagnostic: usable rows "0:06.0-0:07.6 kotori          +30 ct     80"
for ln in DIAG.read_text(encoding="utf-8", errors="replace").splitlines():
    # trailing "  <-- correctable" marker is optional — earlier regex required EOL and dropped 19 rows
    m = re.match(r"\s*(\d+):([\d.]+)-\d+:[\d.]+\s+(\S+)\s+([+-]\s*\d+) ct\s+(\d+)\s*(?:<--.*)?$", ln)
    if m:
        t = tsec(m.group(1), m.group(2)); v = m.group(3)
        out[f"{v}|{int(t)}"] = {"dev": int(m.group(4).replace(" ", "")),
                                "jitter": int(m.group(5)), "usable": True}
        continue
    # skipped rows: "  0:09.1 kotori       raw  +1610 ct (+16.1 semitones)"
    m = re.match(r"\s*(\d+):([\d.]+)\s+(\S+)\s+raw\s+([+-]\s*\d+) ct", ln)
    if m:
        t = tsec(m.group(1), m.group(2)); v = m.group(3)
        out[f"{v}|{int(t)}"] = {"dev": int(m.group(4).replace(" ", "")),
                                "jitter": None, "usable": False}

# --- corrections: "0:30.3-0:35.3 honoka         +40 ct    -40 ct    90"
if CORR_LOG.exists():
    for ln in CORR_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\s*(\d+):([\d.]+)-\d+:[\d.]+\s+(\S+)\s+([+-]\s*\d+) ct\s+([+-]\s*\d+) ct", ln)
        if m:
            t = tsec(m.group(1), m.group(2)); v = m.group(3)
            k = f"{v}|{int(t)}"
            out.setdefault(k, {})["shift"] = int(m.group(5).replace(" ", ""))

P = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/pitch_data.json"
json.dump(out, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_dev = sum(1 for v in out.values() if "dev" in v)
n_use = sum(1 for v in out.values() if v.get("usable"))
n_fix = sum(1 for v in out.values() if "shift" in v)
print(f"{len(out)} phrases | dev measured {n_dev} | usable ref {n_use} | corrected {n_fix}")
print(f"-> {P}")
