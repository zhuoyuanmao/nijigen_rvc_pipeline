"""v4.5 training-corpus re-prep for otoya_sho_mix (Phase 1 of the retrain).

Why (all measured, see KNOWHOW §8):
  * the old corpus was cleaned with the v3-era chain; the sho half carries
    backing/instrument residue up to −18.7dB — noise the model then learns.
  * the plain v4 min-|x| intersection erases breaths (−64dB) — fine for an
    inference source, harmful for TRAINING data where breaths are genuine
    voice samples.

Chain per raw track (raw/{otoya,sho}_raw/*.wav):
  ffmpeg 44.1k stereo
    -> 3x Roformer vocals models applied DIRECTLY to the mix
       (they are full-mix separators, higher SDR than htdemucs; this also
        drops the demucs dependency, whose torchaudio is broken in-venv)
    -> gated blend: loud frames = per-sample min-|x| (conservative,
       backing removed); quiet frames = per-sample MEDIAN (breaths survive
       if >=2 of 3 models kept them); sigmoid gate on frame RMS
    -> anvuew dereverb (less-aggressive) on the blended vocal
    -> head/tail trim + LUFS -20
    -> data/v45_corpus/<voice>_<idx>[ _b].wav
otoya tracks are written twice (suffix _b) => the 2:1 blend happens at the
file level instead of one giant concat (no cross-song slices).

    python _prep_corpus_v45.py            # all tracks (~30 min on 3090)
    python _prep_corpus_v45.py --limit 1  # smoke test
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

from _prep_source_v4 import (separator_pass, VOCALS_MODELS,
                             DEREVERB_MODEL, DEREVERB_STEM)

CHAR = Path(__file__).resolve().parent
RAW = {
    "otoya": Path("raw/otoya_raw"),
    "sho":   Path("raw/sho_raw"),
}
OUT = CHAR / "data/v45_corpus"
WORK = CHAR / "data/v45_work"

GATE_LOW_DB = -38.0     # below: median blend (breaths protected)
GATE_HIGH_DB = -28.0    # above: min-|x| (backing removed)
TARGET_LUFS = -20.0


def run(cmd):
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], check=True)


def gated_min_median(stems: list[Path], out_p: Path) -> Path:
    """loud -> min-|x| (conservative), quiet -> median (breath-friendly)."""
    arrays, srs = [], []
    for p in stems:
        a, sr = sf.read(str(p), always_2d=True)
        arrays.append(a.astype(np.float32))
        srs.append(sr)
    assert len(set(srs)) == 1, f"sr mismatch: {srs}"
    sr = srs[0]
    n = min(a.shape[0] for a in arrays)
    ch = max(a.shape[1] for a in arrays)
    fixed = []
    for a in arrays:
        a = a[:n]
        if a.shape[1] == 1 and ch > 1:
            a = np.tile(a, (1, ch))
        fixed.append(a)
    stack = np.stack(fixed, axis=0)                 # (3, N, C)

    absv = np.abs(stack)
    idx_min = np.argmin(absv, axis=0)
    row = np.arange(n)[:, None]
    col = np.arange(ch)[None, :]
    take_min = stack[idx_min, row, col]
    # sign-preserving median of 3 = the value with middle |x|
    idx_med = np.argsort(absv, axis=0)[1]
    take_med = stack[idx_med, row, col]

    med_mono = take_med.mean(axis=1)
    hop = 512
    rms = librosa.feature.rms(y=med_mono, frame_length=2048, hop_length=hop)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    mid = (GATE_LOW_DB + GATE_HIGH_DB) / 2
    width = (GATE_HIGH_DB - GATE_LOW_DB) / 2
    w = 1.0 / (1.0 + np.exp(-(rms_db - mid) / (width / 2.2)))
    k = 5                                            # ~64ms smoothing
    w = np.convolve(w, np.ones(k) / k, mode="same")
    w_full = np.repeat(w, hop)[:n]
    if len(w_full) < n:
        w_full = np.pad(w_full, (0, n - len(w_full)), mode="edge")
    w_full = w_full[:, None]

    out = w_full * take_min + (1.0 - w_full) * take_med
    sf.write(str(out_p), out.astype(np.float32), sr, subtype="FLOAT")
    print(f"  gate: {float(np.mean(w_full < 0.5))*100:.0f}% frames in "
          f"median (breath) zone", flush=True)
    return out_p


def trim_lufs(src_p: Path, dst_p: Path) -> dict:
    y, sr = sf.read(str(src_p), always_2d=False)
    mono = (y.mean(axis=1) if y.ndim > 1 else y).astype(np.float32)
    _, (a, b) = librosa.effects.trim(mono, top_db=40,
                                     frame_length=2048, hop_length=512)
    keep = int(0.2 * sr)
    a, b = max(0, a - keep), min(len(y), b + keep)
    out = y[a:b]
    meter = pyln.Meter(sr)
    m = out.mean(axis=1) if out.ndim > 1 else out
    lufs = meter.integrated_loudness(np.asarray(m, dtype=np.float64))
    if np.isfinite(lufs):
        out = pyln.normalize.loudness(out, lufs, TARGET_LUFS)
    peak = float(np.max(np.abs(out)))
    if peak > 10 ** (-1 / 20):
        out = out * (10 ** (-1 / 20) / peak)
    sf.write(str(dst_p), out.astype(np.float32), sr, subtype="PCM_16")
    return dict(dur_s=round(len(out) / sr, 1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N tracks per voice (smoke test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    total = 0.0

    for voice, raw_dir in RAW.items():
        tracks = sorted(raw_dir.glob("*.wav"))
        if args.limit:
            tracks = tracks[:args.limit]
        print(f"\n########## {voice}: {len(tracks)} tracks ##########",
              flush=True)
        for i, src in enumerate(tracks):
            tag = f"{voice}_{i:02d}"
            final = OUT / f"{tag}.wav"
            if final.exists() and final.stat().st_size > 0:
                print(f"[skip] {tag}", flush=True)
                if voice == "otoya":
                    dup = OUT / f"{tag}_b.wav"
                    if not dup.exists():
                        shutil.copy2(final, dup)
                continue
            t0 = time.time()
            print(f"\n--- {tag}: {src.name[:60]} ---", flush=True)
            w = WORK / tag
            w.mkdir(parents=True, exist_ok=True)

            src44 = w / "src44.wav"
            if not src44.exists():
                run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                     "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
                     str(src44)])

            stems = [separator_pass(mf, stem, src44, w / f"voc_{t}.wav")
                     for mf, stem, t in VOCALS_MODELS]
            blend = gated_min_median(stems, w / "blend.wav")
            drv = separator_pass(DEREVERB_MODEL, DEREVERB_STEM, blend,
                                 w / "dereverb.wav")
            info = trim_lufs(drv, final)
            total += info["dur_s"]
            if voice == "otoya":
                shutil.copy2(final, OUT / f"{tag}_b.wav")
            print(f"  => {final.name}  {info['dur_s']}s   "
                  f"({time.time()-t0:.0f}s)", flush=True)

    n_files = len(list(OUT.glob("*.wav")))
    print(f"\n=== DONE in {(time.time()-t_all)/60:.1f} min ===")
    print(f"{n_files} files ({total:.0f}s unique) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
