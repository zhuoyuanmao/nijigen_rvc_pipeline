"""Attenuate/remove phrase-end breaths in a finished honoka build.

The opposite of _fix_breaths.py (which splices SOURCE breaths back in to mask
RMVPE whistle). Here we detect breath regions on the source (low-energy AND
unvoiced) and DUCK them in the build by a chosen amount. This is a MIX-stage
taste control, never a preprocessing step (removing breaths upstream makes RVC
hallucinate whistle — see KNOWHOW §9.5).

Levels: -8dB (subtle), -18dB (strong), mute (gone). Sung frames untouched.

    python _remove_breaths.py --build stage8_v2_ab/titan_..._breathfix.wav --duck-db -18
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

CHAR = Path(__file__).resolve().parent
HOP = 512
FADE_S = 0.02


def breath_regions(src, sr):
    rms = librosa.feature.rms(y=src, frame_length=2048, hop_length=HOP)[0]
    db = 20 * np.log10(rms + 1e-10)
    loud = np.percentile(db, 90)
    f0, vflag, _ = librosa.pyin(src.astype(np.float32), fmin=200, fmax=1100,
                                sr=sr, frame_length=2048, hop_length=HOP)
    voiced = np.isfinite(f0)
    n = min(len(db), len(voiced))
    mask = (db[:n] > loud - 45) & (db[:n] < loud - 12) & (~voiced[:n])
    regions, cur = [], None
    for i, b in enumerate(mask):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            if (i - cur) * HOP / sr >= 0.12:
                regions.append((cur * HOP, i * HOP))
            cur = None
    if cur is not None:
        regions.append((cur * HOP, n * HOP))
    return regions


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="tokyo_summer_v3")
    ap.add_argument("--build", required=True, help="path rel to output/<song>/")
    ap.add_argument("--src", default="stage1_40k/song_clean_lead_40k.wav")
    ap.add_argument("--duck-db", type=float, default=-18.0,
                    help="attenuation in dB (use -99 for mute)")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    src, sr = librosa.load(str(base / args.src), sr=None, mono=True)
    y, sr2 = librosa.load(str(base / args.build), sr=None, mono=True)
    assert sr == sr2
    n = min(len(y), len(src))
    y = y.copy()
    g = 10 ** (args.duck_db / 20)
    fade = int(FADE_S * sr)
    regions = breath_regions(src[:n], sr)
    print(f"{len(regions)} breath regions, duck {args.duck_db}dB:")
    for s0, e0 in regions:
        e0 = min(e0, n)
        if e0 - s0 <= 2 * fade:
            continue
        env = np.full(e0 - s0, g)
        t = np.linspace(0, np.pi / 2, fade)
        env[:fade] = 1 - (1 - g) * np.sin(t) ** 2
        env[-fade:] = 1 - (1 - g) * np.cos(t) ** 2
        y[s0:e0] *= env
        print(f"  {s0/sr:5.2f}-{e0/sr:5.2f}s")

    tag = "mute" if args.duck_db <= -40 else f"{int(abs(args.duck_db))}db"
    out = (base / args.build).with_name(
        (base / args.build).stem.replace("_breathfix", "") + f"_nobreath_{tag}.wav")
    sf.write(str(out), y.astype(np.float32), sr)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
