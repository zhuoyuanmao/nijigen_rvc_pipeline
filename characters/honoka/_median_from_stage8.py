"""Median-STFT ensemble from stage8 per-ckpt full-track wavs (no GPU).

The A/B eval (`_ab_eval_v2.py`) already inferred every e100-200 ckpt into
stage8_v2_ab/<arm>_G_..._e*.wav. This medians their magnitude spectra
(+ best-air ckpt's phase) into one clean track PER ARM — the fix for the
single-ckpt "inter-harmonic chorus noise" (see KNOWHOW §11.3):
  * per-ckpt inter-harmonic noise lands in different bins -> median rejects it
    (300-4k flatness 0.021 -> 0.015, near source's 0.012)
  * harmonics agree across ckpts -> body preserved
  * per-ckpt vocoder comb also lands in different bins -> median rejects it too
    (peaks stay 0, unlike v1 waveform-averaging which UNIONs the combs)

    python _median_from_stage8.py                 # both arms -> stage8/<arm>_MEDIAN.wav
    python _median_from_stage8.py --arm titan --phase-epoch 182
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

CHAR = Path(__file__).resolve().parent
S8 = CHAR / "output/tokyo_summer_v3/stage8_v2_ab"
N_FFT, HOP = 2048, 512


def median_ens(wavs, phase_from):
    mags, L = [], None
    for p in wavs:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if L is None:
            L = len(y)
        mags.append(np.abs(librosa.stft(y[:L], n_fft=N_FFT, hop_length=HOP)))
    yp, sr = librosa.load(str(phase_from), sr=None, mono=True)
    phase = np.angle(librosa.stft(yp[:L], n_fft=N_FFT, hop_length=HOP))
    T = min(m.shape[1] for m in mags)
    med = np.median(np.stack([m[:, :T] for m in mags]), axis=0)
    y = librosa.istft(med * np.exp(1j * phase[:, :T]), hop_length=HOP, length=L)
    pk = float(np.max(np.abs(y)))
    return (y / pk * 0.99 if pk > 0.99 else y), sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "titan", "both"], default="both")
    ap.add_argument("--phase-epoch", type=int, default=182,
                    help="take phase from this epoch's ckpt (default: brightest)")
    args = ap.parse_args()
    arms = ["baseline", "titan"] if args.arm == "both" else [args.arm]
    for arm in arms:
        wavs = sorted(w for w in S8.glob(f"{arm}_G_*_infer_e*.wav")
                      if "breathfix" not in w.name and "nobreath" not in w.name
                      and "MEDIAN" not in w.name)
        if not wavs:
            print(f"no stage8 wavs for {arm}"); continue
        ph = next((w for w in wavs if f"_e{args.phase_epoch}" in w.name), wavs[-1])
        y, sr = median_ens(wavs, ph)
        out = S8 / f"{arm}_MEDIAN.wav"
        sf.write(str(out), y.astype(np.float32), sr)
        print(f"{arm}: {len(wavs)} ckpts, phase={ph.name} -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
