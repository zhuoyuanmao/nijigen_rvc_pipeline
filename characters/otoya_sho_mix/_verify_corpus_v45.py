"""Phase-1 acceptance for the v4.5 corpus. Gate before spending 20h of GPU.

Checks:
  1. inventory — file count, per-voice duration, the 2:1 blend ratio
  2. dirt      — re-run the v4 chain on samples; how much is still removable?
                 (old corpus measured -18.7dB worst case on the sho half)
  3. breaths   — did the median gate actually preserve them? compare the
                 quiet-frame energy against the old corpus
  4. F0        — new corpus register vs the old one (drives transpose policy)

    python _verify_corpus_v45.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _prep_source_v4 import (separator_pass, min_intersection,
                             VOCALS_MODELS, DEREVERB_MODEL, DEREVERB_STEM)

CHAR = Path(__file__).resolve().parent
NEW = CHAR / "data/v45_corpus"
OLD = CHAR / "data/character_clean.wav"
WORK = CHAR / "data/v45_verify"
WORK.mkdir(parents=True, exist_ok=True)
OUTJ = CHAR / "data/v45_corpus_report.json"

res = {}

# ---------------------------------------------------------- 1. inventory
print("=== 1. inventory ===", flush=True)
inv = {"otoya": [], "sho": []}
for p in sorted(NEW.glob("*.wav")):
    voice = "otoya" if p.name.startswith("otoya") else "sho"
    info = sf.info(str(p))
    inv[voice].append(dict(name=p.name, dur_s=round(info.frames / info.samplerate, 1),
                           sr=info.samplerate))
tot = {v: round(sum(f["dur_s"] for f in inv[v]) / 60, 1) for v in inv}
uniq = {v: round(sum(f["dur_s"] for f in inv[v] if not f["name"].endswith("_b.wav"))
                 / 60, 1) for v in inv}
ratio = round(tot["otoya"] / max(tot["sho"], 1e-9), 2)
res["inventory"] = dict(n_files={v: len(inv[v]) for v in inv},
                        total_min=tot, unique_min=uniq, blend_ratio=ratio)
print(f"  otoya: {len(inv['otoya'])} files, {tot['otoya']}min "
      f"({uniq['otoya']}min unique)")
print(f"  sho  : {len(inv['sho'])} files, {tot['sho']}min")
print(f"  blend ratio otoya:sho = {ratio}:1   grand total {tot['otoya']+tot['sho']}min")

srs = {f["sr"] for v in inv for f in inv[v]}
res["inventory"]["sample_rates"] = sorted(srs)
if len(srs) > 1:
    print(f"  !! mixed sample rates: {srs}")

# ---------------------------------------------------------- 2. dirt
print("\n=== 2. residual dirt (v4 chain on samples) ===", flush=True)


def rel_db(removed, ref):
    return round(float(10 * np.log10((np.mean(removed ** 2) + 1e-20)
                                     / (np.mean(ref ** 2) + 1e-20))), 1)


dirt = {}
samples = [("otoya", NEW / "otoya_02.wav"), ("otoya", NEW / "otoya_08.wav"),
           ("sho", NEW / "sho_02.wav"), ("sho", NEW / "sho_08.wav")]
for voice, p in samples:
    if not p.exists():
        continue
    y, sr = librosa.load(str(p), sr=44100, mono=True, offset=45.0, duration=60.0)
    if len(y) < sr:
        continue
    tag = p.stem
    sp = WORK / f"{tag}.wav"
    sf.write(str(sp), y, sr)
    drv = separator_pass(DEREVERB_MODEL, DEREVERB_STEM, sp, WORK / f"{tag}_d.wav")
    outs = [separator_pass(mf, st, drv, WORK / f"{tag}_{t}.wav")
            for mf, st, t in VOCALS_MODELS]
    itc = min_intersection(outs, WORK / f"{tag}_i.wav")
    d, _ = librosa.load(str(drv), sr=44100, mono=True)
    it, _ = librosa.load(str(itc), sr=44100, mono=True)
    n = min(len(y), len(d), len(it))
    dirt[tag] = dict(dereverb_removed=rel_db(y[:n] - d[:n], y[:n]),
                     intersect_removed=rel_db(d[:n] - it[:n], y[:n]))
    print(f"  {tag}: dereverb {dirt[tag]['dereverb_removed']}dB   "
          f"intersect {dirt[tag]['intersect_removed']}dB", flush=True)
res["dirt"] = dirt

# ---------------------------------------------------------- 3. breaths
print("\n=== 3. breath preservation ===", flush=True)


def quiet_frame_stats(y, sr):
    """Raw digital-silence % is NOT the right test: this chain correctly
    zeroes instrumental interludes (the old corpus left instrument bleed
    there, which the model then learned). What matters is WHERE the silence
    is — long runs = interludes (fine), sub-0.5s runs = erased breaths (bad)
    — plus whether breath-level events still exist at all."""
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-10)
    loud = np.percentile(db, 90)
    quiet = (db > loud - 45) & (db < loud - 22)

    sil = db < -80
    runs, cur = [], None
    for i, s in enumerate(sil):
        if s and cur is None:
            cur = i
        elif not s and cur is not None:
            runs.append((i - cur) * hop / sr)
            cur = None
    if cur is not None:
        runs.append((len(sil) - cur) * hop / sr)
    d = np.array(runs) if runs else np.array([0.0])

    # breath-level events: 80ms+ of -60..-35dB content
    br = (db > -60) & (db < -35)
    n_breath, cur = 0, None
    for i, b in enumerate(br):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            if (i - cur) * hop / sr > 0.08:
                n_breath += 1
            cur = None

    return dict(quiet_frac=round(float(quiet.mean()), 3),
                quiet_mean_db=round(float(db[quiet].mean()), 1) if quiet.any() else None,
                digital_silence_frac=round(float(sil.mean()), 4),
                short_silence_s=round(float(d[d < 0.5].sum()), 2),
                long_silence_s=round(float(d[d >= 2.0].sum()), 1),
                breath_events=n_breath,
                breath_per_min=round(n_breath / (len(y) / sr / 60), 1))


br = {}
for tag, p in [("new_otoya", NEW / "otoya_02.wav"), ("new_sho", NEW / "sho_02.wav")]:
    if not p.exists():
        continue
    y, sr = librosa.load(str(p), sr=None, mono=True, offset=30.0, duration=90.0)
    br[tag] = quiet_frame_stats(y, sr)
if OLD.exists():
    y, sr = librosa.load(str(OLD), sr=None, mono=True, offset=600.0, duration=90.0)
    br["old_corpus"] = quiet_frame_stats(y, sr)
for k, v in br.items():
    print(f"  {k}: breaths {v['breath_events']} ({v['breath_per_min']}/min)  "
          f"quiet {v['quiet_frac']*100:.0f}% @ {v['quiet_mean_db']}dB  |  "
          f"silence: short {v['short_silence_s']}s / long {v['long_silence_s']}s")
res["breaths"] = br

# ---------------------------------------------------------- 4. F0
print("\n=== 4. F0 register ===", flush=True)


def f0_stats(paths_or_file, n_win=24):
    vals = []
    if isinstance(paths_or_file, list):
        for p in paths_or_file:
            y, sr = librosa.load(str(p), sr=None, mono=True,
                                 offset=30.0, duration=25.0)
            if len(y) < sr:
                continue
            f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=65.0, fmax=1000.0,
                                    sr=sr, frame_length=2048, hop_length=1024)
            v = f0[np.isfinite(f0)]
            vals.append(v[(v > 66) & (v < 990)])
    else:
        info = sf.info(str(paths_or_file))
        dur = info.frames / info.samplerate
        for i in range(n_win):
            y, sr = librosa.load(str(paths_or_file), sr=None, mono=True,
                                 offset=(i + 0.5) * dur / n_win, duration=6.0)
            if len(y) < sr or np.sqrt(np.mean(y ** 2)) < 1e-3:
                continue
            f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=65.0, fmax=1000.0,
                                    sr=sr, frame_length=2048, hop_length=1024)
            v = f0[np.isfinite(f0)]
            vals.append(v[(v > 66) & (v < 990)])
    a = np.concatenate(vals) if vals else np.array([])
    if a.size < 100:
        return None
    return {k: round(float(x), 1) for k, x in
            zip(("p05", "p25", "median", "p75", "p95"),
                np.percentile(a, [5, 25, 50, 75, 95]))}


new_files = sorted([p for p in NEW.glob("*.wav") if not p.name.endswith("_b.wav")])
res["f0_new"] = f0_stats(new_files[:14])
res["f0_old"] = f0_stats(OLD) if OLD.exists() else None
print(f"  new corpus: {res['f0_new']}")
print(f"  old corpus: {res['f0_old']}")

OUTJ.write_text(json.dumps(res, indent=2))
print(f"\n-> {OUTJ}")

# ---------------------------------------------------------- verdict
print("\n=== VERDICT ===")
ok = True
worst_dirt = max((v["intersect_removed"] for v in dirt.values()), default=-99)
if worst_dirt > -30:
    print(f"  !! still dirty: worst intersect_removed {worst_dirt}dB (want < -30)")
    ok = False
else:
    print(f"  clean: worst residual {worst_dirt}dB")
old_bpm = br.get("old_corpus", {}).get("breath_per_min")
for k, v in br.items():
    if not k.startswith("new"):
        continue
    if v["short_silence_s"] > 3.0:
        print(f"  !! {k}: {v['short_silence_s']}s of sub-0.5s silence "
              f"— gate erased breaths")
        ok = False
    if old_bpm and v["breath_per_min"] < old_bpm * 0.6:
        print(f"  !! {k}: {v['breath_per_min']} breaths/min vs old {old_bpm} "
              f"— breath loss")
        ok = False
    else:
        print(f"  {k}: {v['breath_events']} breath events kept, "
              f"{v['long_silence_s']}s of interlude correctly zeroed")
if abs(ratio - 2.0) > 0.35:
    print(f"  !! blend ratio {ratio} far from 2.0")
    ok = False
print("  => PASS, safe to train" if ok else "  => REVIEW before training")
sys.exit(0 if ok else 1)
