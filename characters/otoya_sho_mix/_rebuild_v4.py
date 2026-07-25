"""Tier-1 rebuild: regenerate the vocal from the existing stage3_cache.

No GPU, no re-inference. Fixes the four defects measured in the v3 output:

  1. top-3 softmax averaging  -> artifact-aware single-ckpt selection.
     Different ckpts put their vocoder comb teeth on different frequencies,
     so averaging takes the UNION of every ckpt's artifacts (measured: 98
     tonal peaks in the best single -> 197 in the ensemble) while the real
     vocal partially cancels (1-3k presence -2.2dB, 5-6k -2.6dB).
  2. >10kHz synthetic comb    -> tame EQ (3.8k bell, low shelf) plus a
     linear-phase 12k lowpass. The source's 95% rolloff is 8.95kHz, so every
     dB above ~12k is vocoder invention and can be removed for free.
     (A stronger >5k match-EQ was tried and REJECTED: it over-corrected,
     rolloff95 9017 -> 5595Hz, audibly dull. Kept as the D variant only.)
  3. dynamics flattened       -> per-segment RMS matched to that segment's
     own source RMS (the old code matched every segment to the whole file's
     global RMS, silence included).
  4. hard-spliced boundaries  -> 10ms equal-power fades at every edge.
  5. dropped short vocals     -> _recover_short_segs.py writes snippets the
     segmenter discarded (<2s); the manifest is placed back here.

Usage:
    python _rebuild_v4.py --song tokyo_summer_v3
    python _rebuild_v4.py --song tokyo_summer_v3 --variants   # write A/B set
    # flat-index caches + transposed verse + recovered snippets:
    python _rebuild_v4.py --cache-name stage3_cache_flat \
        --tcache-name stage3_cache_flat_key+12 --transposed-segs 0 1 2 \
        --shorts --suffix _flat_t12
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

# --- spectral repair -------------------------------------------------------
# Tuned against measurement, not taste. A first pass matched the whole >5kHz
# envelope back to the source at 0.85 strength; that over-corrected badly
# (8-12k ended up 7.4dB BELOW the source and rolloff95 fell 9017 -> 5595Hz,
# i.e. audibly dull). The lowpass alone already takes the 10-20k tonal peak
# count to zero, so the match-EQ is now gentle and confined to 8-12k.
LPF_HZ = 12000.0        # above this the source has nothing; all invention
MATCH_LO_HZ = 8000.0    # below this we keep the character's own timbre
MATCH_STRENGTH = 0.40
MAX_CUT_DB = 6.0
MAX_BOOST_DB = 1.5
TAME_HZ = 3800.0        # measured +7.1dB vs source here — harshness band
TAME_OCT = 0.55
TAME_DB = 4.0
DEMUD_DB = 2.5          # measured +2.9dB of 80-300Hz mud vs source
FADE_MS = 10.0
N_FIR = 2047            # zero-phase FIR length (filtfilt -> no pre-ringing)

# What the default output uses. `--variants` writes A/B files for each stage.
DEFAULT_CHAIN = dict(match=False, tame=True, lpf=True)


# --------------------------------------------------------------------------
def avg_spectrum(y: np.ndarray, sr: int, n_fft: int = 4096) -> tuple:
    """Loud-frame average magnitude spectrum."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    if rms.size > 8:
        keep = rms > np.percentile(rms, 70) * 0.4
        if keep.sum() > 4:
            S = S[:, keep]
    return S.mean(axis=1) + 1e-12, freqs


def band_db(P: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    m = (freqs >= lo) & (freqs < hi)
    return float(20 * np.log10(P[m].mean() + 1e-12))


def tonal_peak_count(y: np.ndarray, sr: int, lo=10000.0, hi=20000.0) -> int:
    """Count narrow, prominent spectral ridges — the audible 'buzz'."""
    P, freqs = avg_spectrum(y, sr, n_fft=8192)
    logP = 20 * np.log10(P)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    prom = logP - local
    band = (freqs >= lo) & (freqs < hi)
    return int(np.sum(band & (prom > 8)))


def artifact_score(conv: np.ndarray, src: np.ndarray, sr: int) -> dict:
    """Lower total = better. Purely artifact-driven, unlike the old
    HNR/flatness score which had no high-frequency term at all."""
    Pc, f = avg_spectrum(conv, sr)
    Ps, _ = avg_spectrum(src, sr)

    junk = band_db(Pc, f, 12000, 20000) - band_db(Pc, f, 1000, 3000)
    junk_src = band_db(Ps, f, 12000, 20000) - band_db(Ps, f, 1000, 3000)
    excess_hf = junk - junk_src                       # dB of invented top end

    excess_mid_hf = ((band_db(Pc, f, 8000, 12000) - band_db(Pc, f, 1000, 3000))
                     - (band_db(Ps, f, 8000, 12000) - band_db(Ps, f, 1000, 3000)))

    mud = ((band_db(Pc, f, 80, 300) - band_db(Pc, f, 300, 1000))
           - (band_db(Ps, f, 80, 300) - band_db(Ps, f, 300, 1000)))

    peaks = tonal_peak_count(conv, sr)

    Sc = np.abs(librosa.stft(conv, n_fft=2048, hop_length=512))
    flat = float(librosa.feature.spectral_flatness(S=Sc)[0].mean())

    total = (1.00 * max(excess_hf, 0)
             + 0.60 * max(excess_mid_hf, 0)
             + 0.40 * abs(mud)
             + 0.05 * peaks
             + 300.0 * flat)
    return dict(total=total, excess_hf=excess_hf, excess_mid_hf=excess_mid_hf,
                mud=mud, peaks=peaks, flat=flat)


def _apply_curve(y: np.ndarray, sr: int, f: np.ndarray,
                 gain_lin: np.ndarray) -> np.ndarray:
    """Zero-phase FIR from an arbitrary magnitude curve."""
    fn = np.clip(f / (sr / 2), 0, 1)
    fn[0], fn[-1] = 0.0, 1.0
    gain_lin = gain_lin.copy()
    gain_lin[-1] = 0.0
    taps = firwin2(N_FIR, fn, gain_lin)
    if len(y) <= len(taps) * 3:
        return y
    return filtfilt(taps, [1.0], y)


def spectral_repair(conv: np.ndarray, src: np.ndarray, sr: int,
                    match: bool = True, tame: bool = True,
                    lpf: bool = True) -> np.ndarray:
    """Repair chain. Each stage is independently switchable so the three
    fixes can be A/B'd in isolation."""
    Pc, f = avg_spectrum(conv, sr)
    Ps, _ = avg_spectrum(src, sr)
    gain_db = np.zeros_like(f)

    if match:
        def smooth(P):                              # 1/3-octave smoothing
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
        shelf = 1.0 / (1.0 + (f / 300.0) ** 4)      # low shelf below ~300Hz
        gain_db -= DEMUD_DB * shelf

    gain_lin = 10 ** (gain_db / 20)

    if lpf:
        gain_lin[f >= LPF_HZ] = 0.0
        trans = (f >= LPF_HZ * 0.92) & (f < LPF_HZ)
        if trans.any():
            w = (f[trans] - LPF_HZ * 0.92) / (LPF_HZ * 0.08)
            gain_lin[trans] *= 0.5 * (1 + np.cos(np.pi * w))

    return _apply_curve(conv, sr, f, gain_lin)


def equal_power_fade(y: np.ndarray, sr: int, ms: float = FADE_MS) -> np.ndarray:
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
    ap.add_argument("--variants", action="store_true",
                    help="also write no-EQ / no-LPF variants for A/B")
    ap.add_argument("--transposed-segs", type=int, nargs="*", default=[],
                    help="segments to take from the transposed cache instead")
    ap.add_argument("--transpose-key", type=int, default=12)
    ap.add_argument("--cache-name", default="stage3_cache",
                    help="cache dir for key-0 segments")
    ap.add_argument("--tcache-name", default=None,
                    help="cache dir for transposed segments "
                         "(default: stage3_cache_key<key>)")
    ap.add_argument("--shorts", action="store_true",
                    help="place recovered snippets from stage3_cache_short/")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filename")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    segs_dir, cache_dir = base / "stage2_segments", base / args.cache_name
    tcache_dir = base / (args.tcache_name
                         or f"stage3_cache_key{args.transpose_key:+d}")
    tset = set(args.transposed_segs)
    src_40k = base / "stage1_40k/source_40k.wav"
    out_dir = base / "stage5_rebuilt"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_src, sr = librosa.load(str(src_40k), sr=None, mono=True)
    seg_ids = sorted(int(p.stem.split("_")[1]) for p in segs_dir.glob("seg_*.wav"))
    ckpts = sorted({p.stem.split("_", 1)[1] for p in cache_dir.glob("s*_G_*.wav")},
                   key=lambda s: int(s.replace("G_", "")))
    print(f"{len(seg_ids)} segments x {len(ckpts)} ckpts  @ {sr}Hz  "
          f"{len(full_src)/sr:.1f}s", flush=True)

    # Segment start offsets. Replicate _infer_single.py's split verbatim rather
    # than correlating — it is deterministic and reproduces the exact indices
    # the cache was written under (including its merge quirk: the gap test
    # compares against intervals[i-1][1], not merged[-1][1]).
    intervals = librosa.effects.split(full_src, top_db=35,
                                      frame_length=2048, hop_length=512)
    merged = []
    min_gap = int(3.0 * sr)                       # MERGE_GAP_S
    for i, (s, e) in enumerate(intervals):
        if i > 0 and s - intervals[i - 1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    starts = {}
    for idx, (s, e) in enumerate(merged):
        if (e - s) / sr < 2.0:                    # MIN_SEGMENT_S
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

    report, chosen = {}, {}
    full = np.zeros(len(full_src), dtype=np.float64)
    VARIANTS = {
        "A_raw_best":       dict(match=False, tame=False, lpf=False),
        "B_lpf_only":       dict(match=False, tame=False, lpf=True),
        "C_lpf_tame":       dict(match=False, tame=True,  lpf=True),
        "D_lpf_tame_match": dict(match=True,  tame=True,  lpf=True),
    }
    variants = {k: np.zeros(len(full_src), dtype=np.float64)
                for k in VARIANTS} if args.variants else {}

    for si in seg_ids:
        src_seg, _ = librosa.load(str(segs_dir / f"seg_{si:03d}.wav"), sr=None, mono=True)
        cdir = tcache_dir if si in tset else cache_dir
        key = args.transpose_key if si in tset else 0
        scores = {}
        for cn in ckpts:
            p = cdir / f"s{si:03d}_{cn}.wav"
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
        chosen[si] = f"{win}@key{key:+d}" if key else win
        y, _ = librosa.load(str(cdir / f"s{si:03d}_{win}.wav"), sr=None, mono=True)
        n = min(len(y), len(src_seg))
        y, ref = y[:n].astype(np.float64), src_seg[:n]

        tgt = float(np.sqrt(np.mean(ref ** 2)))     # per-segment -> dynamics
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
            winner=win,
            winner_epoch=int(win.replace("G_", "")) // 126,
            old_ensemble_rank=None,
            before=({k: round(v, 3) for k, v in scores[win].items()}),
            after=({k: round(v, 3) for k, v in after.items()}),
            runner_up=ranked[1][0] if len(ranked) > 1 else None,
            worst=ranked[-1][0],
        )
        print(f"  seg_{si:03d}: {win} (e{int(win.replace('G_',''))//126})  "
              f"excess_hf {scores[win]['excess_hf']:+.1f} -> {after['excess_hf']:+.1f} dB   "
              f"peaks {scores[win]['peaks']} -> {after['peaks']}", flush=True)

    # --- recovered short snippets (see _recover_short_segs.py) -------------
    if args.shorts:
        man_p = base / "stage3_cache_short/manifest.json"
        if not man_p.exists():
            print("!! --shorts given but no manifest; run _recover_short_segs.py")
        else:
            for m in json.loads(man_p.read_text()):
                yv, sro = librosa.load(str(man_p.parent / m["path"]),
                                       sr=sr, mono=True)
                s0, e0 = m["start_sample"], m["end_sample"]
                ref = full_src[s0:e0]
                if len(yv) < int(0.15 * sr) or not len(ref):
                    continue
                v = spectral_repair(yv.astype(np.float64), ref, sr,
                                    **DEFAULT_CHAIN)
                tgt = float(np.sqrt(np.mean(ref ** 2)))
                v *= tgt / (float(np.sqrt(np.mean(v ** 2))) + 1e-12)
                v = equal_power_fade(v, sr)
                n = min(len(v), len(full) - s0)
                full[s0:s0 + n] += v[:n]
                for arr in variants.values():
                    arr[s0:s0 + n] += v[:n]
                chosen[f"short_{m['idx']}"] = (f"{m['ckpt']}@key{m['key']:+d}"
                                               f" @{m['start_s']}s")
                print(f"  short_{m['idx']:03d} @ {m['start_s']}s "
                      f"+{m['dur_s']}s key{m['key']:+d}  placed", flush=True)

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
    print(json.dumps({k: v["winner"] for k, v in report.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
