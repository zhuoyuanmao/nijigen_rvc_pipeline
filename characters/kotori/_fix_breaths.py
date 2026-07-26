"""Replace converted breaths with the source's real breaths.

Why: RMVPE occasionally hallucinates an F0 track during breaths, and the
vocoder then synthesises a faint moving whistle riding on the breath noise
(audible; invisible to pooled spectral metrics because it shifts frequency
per breath). Breath is essentially speaker-neutral noise, so the cleanest
fix is structural: splice the SOURCE's genuine breaths back in, level-
matched, with short crossfades. Sung frames are untouched by construction.

Breath detection runs on the SOURCE (not the build): frames that are
low-energy (1.5%..25% of the loud level) AND unvoiced (pyin).

    python _fix_breaths.py --build stage5_rebuilt/vocals_rebuilt_flat_medens.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

CHAR = Path(__file__).resolve().parent
N_FFT, HOP = 2048, 512
FADE_S = 0.02
PAD_S = 0.03            # widen each region so whistle onset/decay is covered
MIN_REGION_S = 0.12


def breath_regions(src: np.ndarray, sr: int) -> list[tuple[int, int]]:
    S = np.abs(librosa.stft(src, n_fft=N_FFT, hop_length=HOP))
    rms = librosa.feature.rms(S=S, frame_length=N_FFT)[0]
    loud = np.percentile(rms, 75)
    quiet = (rms > loud * 0.015) & (rms < loud * 0.25)

    f0, vflag, _ = librosa.pyin(src.astype(np.float32), fmin=65.0,
                                fmax=1000.0, sr=sr,
                                frame_length=N_FFT, hop_length=HOP)
    T = min(len(quiet), len(vflag))
    unvoiced = ~(vflag[:T] > 0)
    mask = quiet[:T] & unvoiced

    regions, cur = [], None
    for i, b in enumerate(mask):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            if (i - cur) * HOP / sr >= MIN_REGION_S:
                regions.append((cur, i))
            cur = None
    if cur is not None and (T - cur) * HOP / sr >= MIN_REGION_S:
        regions.append((cur, T))

    pad = int(PAD_S * sr)
    out = []
    for a, b in regions:
        s = max(0, a * HOP - pad)
        e = min(len(src), b * HOP + pad)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="tokyo_summer_v3")
    ap.add_argument("--build", required=True,
                    help="path relative to output/<song>/ of the build to fix")
    ap.add_argument("--src", default="stage1_40k/song_clean_lead_40k.wav")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    build_p = base / args.build
    src, sr = librosa.load(str(base / args.src), sr=None, mono=True)
    y, sr2 = librosa.load(str(build_p), sr=None, mono=True)
    assert sr == sr2, (sr, sr2)
    n = min(len(y), len(src))

    regions = breath_regions(src[:n], sr)
    print(f"{len(regions)} breath region(s):")
    fade = int(FADE_S * sr)
    y = y.copy()
    for s, e in regions:
        e = min(e, n)
        if e - s <= 2 * fade:
            continue
        # level: keep the build's local energy so mix balance is unchanged;
        # fall back to source level if the build is near-silent here
        rb = float(np.sqrt(np.mean(y[s:e] ** 2)))
        rs = float(np.sqrt(np.mean(src[s:e] ** 2))) + 1e-12
        g = (rb / rs) if rb > 1e-5 else 1.0
        g = float(np.clip(g, 0.25, 4.0))
        rep = src[s:e] * g
        w = np.ones(e - s)
        t = np.linspace(0, np.pi / 2, fade)
        w[:fade] = np.sin(t) ** 2
        w[-fade:] = np.cos(t) ** 2
        y[s:e] = y[s:e] * (1 - w) + rep * w
        print(f"  {s/sr:6.2f}-{e/sr:6.2f}s  gain {20*np.log10(g):+5.1f} dB")

    out = build_p.with_name(build_p.stem + "_breathfix.wav")
    sf.write(str(out), y.astype(np.float32), sr)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
