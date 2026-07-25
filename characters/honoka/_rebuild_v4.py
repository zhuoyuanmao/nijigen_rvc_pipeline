"""Honoka: rebuild the vocal from a stage3 cache (port of otoya_sho_mix's
_rebuild_v4.py with HONOKA-MEASURED repair params — do not copy otoya's).

Measured on her tokyo_summer_v3 cache (scratch honoka_measure.py):
  * her source is a bright female vocal: real content to ~15.5kHz
    (rolloff95 14.5k; 13-15k bands at −16..−20dB rel core)
  * conversions ADD junk only ABOVE 16kHz (+21dB excess at 17k)
    -> LPF at 15.8k, NOT otoya's 12k (which would cut 3kHz of real air)
  * harshness excess at 2-5k (+7.6..+9.6dB)  -> tame bell 3.2k, −5dB
  * low end clean (−0.4dB excess)            -> no de-mud shelf
  * 6-8k is DEFICIENT (−4..−11dB)            -> never cut there
Ensemble vs single (round-1): her top-3 ensemble buzz −22.7 vs best single
−27.2 — same union-of-combs failure as otoya; single-ckpt selection here.

Usage:
    python _rebuild_v4.py --cache-name stage3_cache_flat --suffix _flat
    python _rebuild_v4.py --variants ...     # write A/B set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import firwin2, filtfilt

CHAR = Path(__file__).resolve().parent

# --- spectral repair (HONOKA-tuned; see module docstring) ------------------
LPF_HZ = 15800.0
MATCH_LO_HZ = 8000.0
MATCH_STRENGTH = 0.40
MAX_CUT_DB = 6.0
MAX_BOOST_DB = 1.5
TAME_HZ = 3200.0
TAME_OCT = 0.65
TAME_DB = 5.0
DEMUD_DB = 0.0
FADE_MS = 10.0
N_FIR = 2047

DEFAULT_CHAIN = dict(match=False, tame=True, lpf=True)

# split params — identical to the original segmentation of her source
SILENCE_TOP_DB = 35
MERGE_GAP_S = 3.0
MIN_SEGMENT_S = 2.0


# --------------------------------------------------------------------------
def avg_spectrum(y: np.ndarray, sr: int, n_fft: int = 4096) -> tuple:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    if rms.size > 8:
        keep = rms > np.percentile(rms, 70) * 0.4
        if keep.sum() > 4:
            S = S[:, keep]
    return S.mean(axis=1) + 1e-12, freqs


def band_db(P, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(20 * np.log10(P[m].mean() + 1e-12))


def tonal_peak_count(y, sr, lo=15500.0, hi=20000.0):
    """Her comb lives above ~16k (source is dead there)."""
    P, freqs = avg_spectrum(y, sr, n_fft=8192)
    logP = 20 * np.log10(P)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    prom = logP - local
    return int(np.sum((freqs >= lo) & (freqs < hi) & (prom > 8)))


def artifact_score(conv, src, sr):
    """Lower = better. Bands adapted to honoka's measured excess profile."""
    Pc, f = avg_spectrum(conv, sr)
    Ps, _ = avg_spectrum(src, sr)

    # invented top end: >16k, where her source is silent
    excess_hf = ((band_db(Pc, f, 16000, 20000) - band_db(Pc, f, 1000, 3000))
                 - (band_db(Ps, f, 16000, 20000) - band_db(Ps, f, 1000, 3000)))
    # 2-5k harshness
    harsh = ((band_db(Pc, f, 2000, 5000) - band_db(Pc, f, 300, 2000))
             - (band_db(Ps, f, 2000, 5000) - band_db(Ps, f, 300, 2000)))
    # air fidelity: how much of the source's real 12-15.5k is missing
    air_loss = ((band_db(Ps, f, 12000, 15500) - band_db(Ps, f, 1000, 3000))
                - (band_db(Pc, f, 12000, 15500) - band_db(Pc, f, 1000, 3000)))
    peaks = tonal_peak_count(conv, sr)
    Sc = np.abs(librosa.stft(conv, n_fft=2048, hop_length=512))
    flat = float(librosa.feature.spectral_flatness(S=Sc)[0].mean())

    total = (1.00 * max(excess_hf, 0)
             + 0.60 * max(harsh, 0)
             + 0.30 * max(air_loss, 0)
             + 0.05 * peaks
             + 300.0 * flat)
    return dict(total=total, excess_hf=excess_hf, harsh=harsh,
                air_loss=air_loss, peaks=peaks, flat=flat)


def _apply_curve(y, sr, f, gain_lin):
    fn = np.clip(f / (sr / 2), 0, 1)
    fn[0], fn[-1] = 0.0, 1.0
    gain_lin = gain_lin.copy()
    gain_lin[-1] = 0.0
    taps = firwin2(N_FIR, fn, gain_lin)
    if len(y) <= len(taps) * 3:
        return y
    return filtfilt(taps, [1.0], y)


def spectral_repair(conv, src, sr, match=True, tame=True, lpf=True):
    Pc, f = avg_spectrum(conv, sr)
    Ps, _ = avg_spectrum(src, sr)
    gain_db = np.zeros_like(f)

    if match:
        def smooth(P):
            out = np.empty_like(P)
            for i, fc in enumerate(f):
                m = (f >= fc / 1.26) & (f <= fc * 1.26)
                out[i] = P[m].mean() if m.any() else P[i]
            return out
        g = 20 * np.log10((smooth(Ps) + 1e-12) / (smooth(Pc) + 1e-12))
        g = np.clip(g * MATCH_STRENGTH, -MAX_CUT_DB, MAX_BOOST_DB)
        gain_db += g * np.clip((f - MATCH_LO_HZ) / 2000.0, 0.0, 1.0)

    if tame:
        bell = np.exp(-0.5 * (np.log2((f + 1e-9) / TAME_HZ) / TAME_OCT) ** 2)
        gain_db -= TAME_DB * bell
        if DEMUD_DB:
            shelf = 1.0 / (1.0 + (f / 300.0) ** 4)
            gain_db -= DEMUD_DB * shelf

    gain_lin = 10 ** (gain_db / 20)
    if lpf:
        gain_lin[f >= LPF_HZ] = 0.0
        trans = (f >= LPF_HZ * 0.94) & (f < LPF_HZ)
        if trans.any():
            w = (f[trans] - LPF_HZ * 0.94) / (LPF_HZ * 0.06)
            gain_lin[trans] *= 0.5 * (1 + np.cos(np.pi * w))
    return _apply_curve(conv, sr, f, gain_lin)


def equal_power_fade(y, sr, ms=FADE_MS):
    n = int(sr * ms / 1000)
    if n * 2 >= len(y) or n < 2:
        return y
    y = y.copy()
    t = np.linspace(0, np.pi / 2, n)
    y[:n] *= np.sin(t)
    y[-n:] *= np.cos(t)
    return y


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="tokyo_summer_v3")
    ap.add_argument("--cache-name", default="stage3_cache")
    ap.add_argument("--variants", action="store_true")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    segs_dir, cache_dir = base / "stage2_segments", base / args.cache_name
    src_40k = base / "stage1_40k/song_clean_lead_40k.wav"
    out_dir = base / "stage5_rebuilt"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_src, sr = librosa.load(str(src_40k), sr=None, mono=True)
    seg_ids = sorted(int(p.stem.split("_")[1]) for p in segs_dir.glob("seg_*.wav"))
    ckpts = sorted({p.stem.split("_", 1)[1] for p in cache_dir.glob("s*_G_*.wav")},
                   key=lambda s: int(s.replace("G_", "")))
    print(f"{len(seg_ids)} segments x {len(ckpts)} ckpts  @ {sr}Hz  "
          f"{len(full_src)/sr:.1f}s", flush=True)

    intervals = librosa.effects.split(full_src, top_db=SILENCE_TOP_DB,
                                      frame_length=2048, hop_length=512)
    merged = []
    min_gap = int(MERGE_GAP_S * sr)
    for i, (s, e) in enumerate(intervals):
        if i > 0 and s - intervals[i - 1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    starts = {}
    for idx, (s, e) in enumerate(merged):
        dur = (e - s) / sr
        if dur < MIN_SEGMENT_S:
            print(f"  !! dropped short interval {s/sr:.2f}s +{dur:.2f}s "
                  f"(recover manually if audible)", flush=True)
            continue
        starts[idx] = int(s)

    missing = [si for si in seg_ids if si not in starts]
    if missing:
        raise SystemExit(f"split mismatch, no offset for segments {missing}")
    for si in seg_ids:
        s, _ = librosa.load(str(segs_dir / f"seg_{si:03d}.wav"), sr=None, mono=True)
        want = merged[si][1] - merged[si][0]
        flag = "" if abs(want - len(s)) < 64 else f"  !! length {len(s)} vs {want}"
        print(f"  seg_{si:03d} @ {starts[si]/sr:7.2f}s  dur {len(s)/sr:5.1f}s{flag}",
              flush=True)

    VARIANTS = {
        "A_raw_best":  dict(match=False, tame=False, lpf=False),
        "B_lpf_only":  dict(match=False, tame=False, lpf=True),
        "C_lpf_tame":  dict(match=False, tame=True,  lpf=True),
    }
    report, chosen = {}, {}
    full = np.zeros(len(full_src), dtype=np.float64)
    variants = {k: np.zeros(len(full_src), dtype=np.float64)
                for k in VARIANTS} if args.variants else {}

    for si in seg_ids:
        src_seg, _ = librosa.load(str(segs_dir / f"seg_{si:03d}.wav"),
                                  sr=None, mono=True)
        scores = {}
        for cn in ckpts:
            p = cache_dir / f"s{si:03d}_{cn}.wav"
            if not p.exists():
                continue
            y, _ = librosa.load(str(p), sr=None, mono=True)
            n = min(len(y), len(src_seg))
            scores[cn] = artifact_score(y[:n], src_seg[:n], sr)
        if not scores:
            print(f"  seg_{si:03d}: NO CACHE, skipped", flush=True)
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1]["total"])
        win = ranked[0][0]
        chosen[si] = win
        y, _ = librosa.load(str(cache_dir / f"s{si:03d}_{win}.wav"),
                            sr=None, mono=True)
        n = min(len(y), len(src_seg))
        y, ref = y[:n].astype(np.float64), src_seg[:n]
        tgt = float(np.sqrt(np.mean(ref ** 2)))
        st = starts[si]

        def place(sig, dest):
            v = np.asarray(sig, dtype=np.float64)
            v = v * (tgt / (float(np.sqrt(np.mean(v ** 2))) + 1e-12))
            v = equal_power_fade(v, sr)
            m = min(len(v), len(dest) - st)
            dest[st:st + m] += v[:m]

        y_eq = spectral_repair(y, ref, sr, **DEFAULT_CHAIN)
        place(y_eq, full)
        if args.variants:
            for vname, cfg in VARIANTS.items():
                place(spectral_repair(y, ref, sr, **cfg), variants[vname])

        after = artifact_score(y_eq, ref, sr)
        report[f"seg_{si:03d}"] = dict(
            winner=win, winner_epoch=int(win.replace("G_", "")) // 234,
            before={k: round(v, 3) for k, v in scores[win].items()},
            after={k: round(v, 3) for k, v in after.items()},
            runner_up=ranked[1][0] if len(ranked) > 1 else None,
            worst=ranked[-1][0],
        )
        print(f"  seg_{si:03d}: {win} (e{int(win.replace('G_',''))//234})  "
              f"excess_hf {scores[win]['excess_hf']:+.1f} -> "
              f"{after['excess_hf']:+.1f} dB   "
              f"peaks {scores[win]['peaks']} -> {after['peaks']}", flush=True)

    peak = float(np.max(np.abs(full)))
    if peak > 0.99:
        full = full / peak * 0.99
    out = out_dir / f"vocals_rebuilt{args.suffix}.wav"
    sf.write(str(out), full.astype(np.float32), sr)
    print(f"\n-> {out}", flush=True)

    if args.variants:
        for k, v in variants.items():
            pk = float(np.max(np.abs(v)))
            if pk > 0.99:
                v = v / pk * 0.99
            sf.write(str(out_dir / f"variant_{k}.wav"), v.astype(np.float32), sr)

    (out_dir / "rebuild_report.json").write_text(
        json.dumps({"chosen": chosen, "segments": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
