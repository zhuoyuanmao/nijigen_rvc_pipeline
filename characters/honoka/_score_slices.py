"""Slice quality scoring + selection for honoka v2 (corpus 278min -> ~180min).

Runs AFTER RVC preprocess + F0 extraction, so it can reuse the RVC-computed
f0 (2b-f0nsf/*.npy, 0 = unvoiced) instead of re-running pyin (~40min saved).

Per slice ({song}_{idx}.wav in 0_gt_wavs):
  components (z-normalised across corpus, then weighted):
    0.40  HNR proxy        harmonic/percussive variance ratio (hpss)
    0.25  voiced_ratio     fraction of frames with f0 > 0
    0.20  1 - flatness     spectral flatness (low = voice-like)
    0.15  1 - silence_frac frames below (max - 40dB)
  hard rejects (definitely-bad slices):
    voiced_ratio < 0.15 | octave-jump ratio > 0.30 | dur < 1.0s

Selection to --target-min with two diversity guards (METHODOLOGY §5):
  * per-song cap: <= 1.4x the song's proportional share of the target
  * F0 coverage: 6 quantile bins over slice median F0; per-bin budget
    proportional to the bin's share of eligible duration

Outputs (into the RVC experiment dir):
  selected_slices.txt      basenames ("12_034"), one per line
  slice_score_report.json  distributions + per-song admission stats

    python _score_slices.py --exp honoka_v2 --target-min 180
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

RVC = Path(__file__).resolve().parents[2] / "Retrieval-based-Voice-Conversion-WebUI"

W_HNR, W_VOICED, W_FLAT, W_SIL = 0.40, 0.25, 0.20, 0.15
REJECT_VOICED = 0.15
REJECT_OCT = 0.30
REJECT_DUR = 1.0
SONG_CAP_X = 1.4
N_F0_BINS = 6


def slice_features(wav_p: Path, f0_p: Path) -> dict | None:
    try:
        y, sr = sf.read(str(wav_p), dtype="float32")
    except Exception:
        return None
    if y.ndim > 1:
        y = y.mean(axis=1)
    dur = len(y) / sr
    if dur < 0.3:
        return None

    # f0-based (RVC f0nsf: Hz, 0 = unvoiced)
    f0 = np.load(str(f0_p)) if f0_p.exists() else np.zeros(8)
    voiced = f0 > 0
    voiced_ratio = float(voiced.mean()) if f0.size else 0.0
    v = f0[voiced]
    if v.size >= 3:
        r = np.abs(np.log2(v[1:] / v[:-1]))
        oct_jump = float(np.mean(r > 0.45))
        f0_med = float(np.median(v))
    else:
        oct_jump, f0_med = 1.0, 0.0

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    flat = float(librosa.feature.spectral_flatness(S=S)[0].mean())
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    db = 20 * np.log10(rms + 1e-10)
    silence_frac = float(np.mean(db < db.max() - 40))

    try:
        h = librosa.effects.harmonic(y, margin=3.0)
        p = librosa.effects.percussive(y, margin=3.0)
        hnr = float(np.log10(h.var() / (p.var() + 1e-9) + 1))
    except Exception:
        hnr = 0.0

    return dict(dur=dur, voiced_ratio=voiced_ratio, oct_jump=oct_jump,
                f0_med=f0_med, flat=flat, silence=silence_frac, hnr=hnr)


def zscore(a: np.ndarray) -> np.ndarray:
    s = a.std()
    return (a - a.mean()) / (s + 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="honoka_v2")
    ap.add_argument("--target-min", type=float, default=180.0)
    args = ap.parse_args()

    exp = RVC / "logs" / args.exp
    gt, f0d = exp / "0_gt_wavs", exp / "2b-f0nsf"
    wavs = sorted(gt.glob("*.wav"))
    print(f"{len(wavs)} slices in {gt}", flush=True)

    rows = []
    for i, p in enumerate(wavs):
        ft = slice_features(p, f0d / f"{p.name}.npy")
        if ft is None:
            continue
        ft["name"] = p.stem
        ft["song"] = int(p.stem.split("_")[0])
        rows.append(ft)
        if (i + 1) % 400 == 0:
            print(f"  scored {i+1}/{len(wavs)}", flush=True)

    total_min = sum(r["dur"] for r in rows) / 60
    print(f"scored {len(rows)} slices, {total_min:.1f} min total", flush=True)

    # hard rejects
    eligible, rejected = [], []
    for r in rows:
        if (r["voiced_ratio"] < REJECT_VOICED or r["oct_jump"] > REJECT_OCT
                or r["dur"] < REJECT_DUR):
            rejected.append(r)
        else:
            eligible.append(r)
    el_min = sum(r["dur"] for r in eligible) / 60
    print(f"hard-rejected {len(rejected)} ({sum(r['dur'] for r in rejected)/60:.1f} min)"
          f"  -> eligible {len(eligible)} ({el_min:.1f} min)", flush=True)

    # composite quality (z-normalised)
    z_hnr = zscore(np.array([r["hnr"] for r in eligible]))
    z_vc = zscore(np.array([r["voiced_ratio"] for r in eligible]))
    z_fl = zscore(np.array([-r["flat"] for r in eligible]))
    z_si = zscore(np.array([-r["silence"] for r in eligible]))
    for i, r in enumerate(eligible):
        r["q"] = float(W_HNR * z_hnr[i] + W_VOICED * z_vc[i]
                       + W_FLAT * z_fl[i] + W_SIL * z_si[i])

    target_s = args.target_min * 60
    if el_min * 60 <= target_s:
        print("eligible <= target; keeping everything")
        selected = eligible
    else:
        # per-song caps
        song_dur = defaultdict(float)
        for r in eligible:
            song_dur[r["song"]] += r["dur"]
        el_total = sum(song_dur.values())
        cap = {s: SONG_CAP_X * (d / el_total) * target_s
               for s, d in song_dur.items()}

        # F0 quantile bins over voiced slices
        f0s = np.array([r["f0_med"] for r in eligible if r["f0_med"] > 0])
        edges = np.quantile(f0s, np.linspace(0, 1, N_F0_BINS + 1))
        edges[0], edges[-1] = 0.0, 1e9

        def bin_of(r):
            return int(np.searchsorted(edges, max(r["f0_med"], 1e-6),
                                       side="right") - 1)

        bins = defaultdict(list)
        for r in eligible:
            bins[bin_of(r)].append(r)
        bin_budget = {b: target_s * sum(x["dur"] for x in v) / el_total
                      for b, v in bins.items()}

        used_song = defaultdict(float)
        selected, leftovers = [], []
        for b, members in bins.items():
            members.sort(key=lambda r: -r["q"])
            acc = 0.0
            for r in members:
                if acc + r["dur"] <= bin_budget[b] and \
                   used_song[r["song"]] + r["dur"] <= cap[r["song"]]:
                    selected.append(r)
                    acc += r["dur"]
                    used_song[r["song"]] += r["dur"]
                else:
                    leftovers.append(r)
        # top up to target from leftovers (still respecting song caps)
        leftovers.sort(key=lambda r: -r["q"])
        sel_s = sum(r["dur"] for r in selected)
        for r in leftovers:
            if sel_s >= target_s:
                break
            if used_song[r["song"]] + r["dur"] <= cap[r["song"]]:
                selected.append(r)
                sel_s += r["dur"]
                used_song[r["song"]] += r["dur"]

    sel_min = sum(r["dur"] for r in selected) / 60
    songs_in = len({r["song"] for r in selected})
    print(f"\nSELECTED {len(selected)} slices, {sel_min:.1f} min, "
          f"{songs_in} songs represented", flush=True)

    out = exp / "selected_slices.txt"
    out.write_text("\n".join(sorted(r["name"] for r in selected)))
    q_all = sorted(r["q"] for r in eligible)
    report = dict(
        total_slices=len(rows), total_min=round(total_min, 1),
        rejected=len(rejected), eligible_min=round(el_min, 1),
        selected=len(selected), selected_min=round(sel_min, 1),
        songs_represented=songs_in,
        per_song_selected_min={
            str(s): round(sum(r["dur"] for r in selected if r["song"] == s) / 60, 2)
            for s in sorted({r["song"] for r in rows})},
        q_quartiles=[round(float(np.percentile(q_all, p)), 3)
                     for p in (5, 25, 50, 75, 95)],
    )
    (exp / "slice_score_report.json").write_text(json.dumps(report, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
