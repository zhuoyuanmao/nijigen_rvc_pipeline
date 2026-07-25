"""Median-STFT ensemble — timbre smoothing WITHOUT the union-of-combs.

Why: waveform-averaging (the old top-3 ensemble) superimposes every ckpt's
vocoder comb (peaks 98→197 on otoya; 57 here) and phase-cancels the shared
vocal. Median of MAGNITUDE spectra instead rejects per-ckpt outlier bins —
each comb tooth lives in only one ckpt, so the median never contains it —
while phase comes from the single best ckpt. The usual repair chain follows.

Measured on tokyo_summer_v3 (vs the plain single-ckpt flat rebuild):
  3-5k harshness  +3.4dB residual -> +0.1dB   (median smooths what EQ can't)
  peaks >15.5k    0 -> 1, timbre proxy 41.1 -> 41.3   (ties)

    python _median_ensemble.py --cache-name stage3_cache_flat --top-k 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from _rebuild_v4 import (artifact_score, spectral_repair, equal_power_fade,
                         DEFAULT_CHAIN)

CHAR = Path(__file__).resolve().parent
N_FFT, HOP = 2048, 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="tokyo_summer_v3")
    ap.add_argument("--cache-name", default="stage3_cache_flat")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--suffix", default="_flat_medens")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    cache = base / args.cache_name
    segs_dir = base / "stage2_segments"
    src_full_p = base / "stage1_40k/song_clean_lead_40k.wav"
    out_dir = base / "stage5_rebuilt"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_src, sr = librosa.load(str(src_full_p), sr=None, mono=True)
    full = np.zeros(len(full_src))

    # same split re-derivation as _rebuild_v4
    intervals = librosa.effects.split(full_src, top_db=35,
                                      frame_length=2048, hop_length=512)
    merged = []
    min_gap = int(3.0 * sr)
    for i, (s, e) in enumerate(intervals):
        if i > 0 and s - intervals[i - 1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    starts = {i: int(s) for i, (s, e) in enumerate(merged)
              if (e - s) / sr >= 2.0}

    for sp in sorted(segs_dir.glob("seg_*.wav")):
        si = int(sp.stem.split("_")[1])
        src, _ = librosa.load(str(sp), sr=None, mono=True)

        scores = {}
        for p in sorted(cache.glob(f"s{si:03d}_G_*.wav")):
            cn = p.stem.split("_", 1)[1]
            y, _ = librosa.load(str(p), sr=None, mono=True)
            n = min(len(y), len(src))
            scores[cn] = artifact_score(y[:n], src[:n], sr)["total"]
        if not scores:
            print(f"seg_{si:03d}: NO CACHE", flush=True)
            continue
        top = sorted(scores, key=scores.get)[:args.top_k]
        print(f"seg_{si:03d} top-{args.top_k}: {top}", flush=True)

        mags, phase_best, L = [], None, None
        for cn in top:
            y, _ = librosa.load(str(cache / f"s{si:03d}_{cn}.wav"),
                                sr=None, mono=True)
            if L is None:
                L = len(y)
            D = librosa.stft(y[:L], n_fft=N_FFT, hop_length=HOP)
            if phase_best is None:
                phase_best = np.angle(D)
            mags.append(np.abs(D))
        T = min(m.shape[1] for m in mags)
        med = np.median(np.stack([m[:, :T] for m in mags]), axis=0)
        y_med = librosa.istft(med * np.exp(1j * phase_best[:, :T]),
                              hop_length=HOP, length=L)

        n = min(len(y_med), len(src))
        y_rep = spectral_repair(y_med[:n].astype(np.float64), src[:n], sr,
                                **DEFAULT_CHAIN)
        tgt = float(np.sqrt(np.mean(src[:n] ** 2)))
        y_rep *= tgt / (float(np.sqrt(np.mean(y_rep ** 2))) + 1e-12)
        y_rep = equal_power_fade(y_rep, sr)
        st = starts[si]
        m = min(len(y_rep), len(full) - st)
        full[st:st + m] += y_rep[:m]

        after = artifact_score(y_rep, src[:n], sr)
        print(f"  after repair: excess_hf {after['excess_hf']:+.1f}  "
              f"peaks {after['peaks']}", flush=True)

    pk = float(np.max(np.abs(full)))
    if pk > 0.99:
        full = full / pk * 0.99
    out = out_dir / f"vocals_rebuilt{args.suffix}.wav"
    sf.write(str(out), full.astype(np.float32), sr)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
