"""Sync pitch_data.json to the FINAL v8 state:
- the 17 curve-corrected phrases: shift = clip(-dev, ±80) (v2 cap)
- add the 2:09.4 umi-male dialogue line (dev -80 -> curve corrected, ~+70)
- 0:55.9 umi-male stays uncorrected + flagged ear-vetoed (correction lost the A/B)
"""
import json
P = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/pitch_data.json"
d = json.load(open(P, encoding="utf-8"))
n = 0
for k, v in d.items():
    if "shift" in v and v.get("usable"):
        v["shift"] = int(max(-80, min(80, -v["dev"])))
        v["method"] = "curve"
        n += 1
d["umi-male|129"]["shift"] = 70          # 2:09.4 dialogue line, curve-corrected in v8
d["umi-male|129"]["method"] = "curve"
d["umi-male|55"]["veto"] = True          # 0:55.9 correction rejected by ear (kept original)
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"updated: {n} curve shifts + 2:09 added + 0:55 ear-veto flag")
