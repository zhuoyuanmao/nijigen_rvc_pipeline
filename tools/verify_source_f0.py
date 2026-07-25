"""Pre-flight gate: is an inference source inside a model's trained F0 range?

Lesson from otoya_sho_mix / tokyo_summer_v3 (2026-07-25): the corpus sings at
median 248Hz (p05 173Hz) but the song's verse sits at ~127Hz — an octave below
anything the model saw. Result: 12dB worse vocoder buzz on those segments,
unfixable by post-processing alone; +12 transpose recovered it entirely.
This check catches that BEFORE burning GPU time.

Usage:
    # one-time per character: compute + cache corpus stats
    python tools/verify_source_f0.py --corpus characters/otoya_sho_mix/data/character_clean.wav \
        --stats characters/otoya_sho_mix/models/corpus_f0_stats.json

    # per song: verify a source (whole file, or a stage2_segments dir)
    python tools/verify_source_f0.py --stats characters/otoya_sho_mix/models/corpus_f0_stats.json \
        --input characters/otoya_sho_mix/output/tokyo_summer_v3/stage2_segments

Verdicts per segment:
    OK        median F0 inside corpus [p25, p75] — sweet spot
    EDGE      inside [p05, p95] but outside the quartile band
    OOD       outside [p05, p95] — expect artifacts; use the recommended key
The recommended transpose is the key in {-12, 0, +12} whose shifted median
lands closest to the corpus median (log-F0 distance).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

CANDIDATE_KEYS = (-12, 0, 12)


def f0_frames(y: np.ndarray, sr: int) -> np.ndarray:
    f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=65.0, fmax=1000.0,
                            sr=sr, frame_length=2048, hop_length=1024)
    v = f0[np.isfinite(f0)]
    return v[(v > 66) & (v < 990)]          # drop pyin floor/ceiling failures


def corpus_stats(corpus_wav: Path, n_windows: int = 40) -> dict:
    """Sample n 6s windows spread across the corpus; robust and fast."""
    info = sf.info(str(corpus_wav))
    dur = info.frames / info.samplerate
    vals = []
    for i in range(n_windows):
        start = (i + 0.5) * dur / n_windows
        try:
            y, sr = librosa.load(str(corpus_wav), sr=None, mono=True,
                                 offset=start, duration=6.0)
        except Exception:
            continue
        if len(y) < sr or np.sqrt(np.mean(y ** 2)) < 1e-3:
            continue
        vals.append(f0_frames(y, sr))
    v = np.concatenate(vals) if vals else np.array([])
    if v.size < 200:
        raise SystemExit(f"too few voiced frames in corpus ({v.size})")
    return {
        "corpus": str(corpus_wav),
        "n_frames": int(v.size),
        "p05": round(float(np.percentile(v, 5)), 1),
        "p25": round(float(np.percentile(v, 25)), 1),
        "median": round(float(np.median(v)), 1),
        "p75": round(float(np.percentile(v, 75)), 1),
        "p95": round(float(np.percentile(v, 95)), 1),
    }


def verdict(med: float, st: dict) -> tuple[str, int]:
    if st["p25"] <= med <= st["p75"]:
        band = "OK"
    elif st["p05"] <= med <= st["p95"]:
        band = "EDGE"
    else:
        band = "OOD"
    best = min(CANDIDATE_KEYS,
               key=lambda k: abs(np.log2(med * 2 ** (k / 12) / st["median"])))
    return band, best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    help="training corpus wav; computes stats and writes --stats")
    ap.add_argument("--stats", type=Path, required=True,
                    help="stats JSON (written with --corpus, read otherwise)")
    ap.add_argument("--input", type=Path,
                    help="source wav or a directory of segment wavs")
    ap.add_argument("--json-out", type=Path, help="also write the report here")
    args = ap.parse_args()

    if args.corpus:
        st = corpus_stats(args.corpus)
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(st, indent=2))
        print(f"corpus stats -> {args.stats}\n{json.dumps(st, indent=2)}")
        if not args.input:
            return 0

    if not args.stats.exists():
        raise SystemExit(f"stats file missing: {args.stats} (run with --corpus first)")
    st = json.loads(args.stats.read_text())

    if not args.input:
        raise SystemExit("nothing to verify: pass --input")

    targets: list[Path]
    if args.input.is_dir():
        targets = sorted(args.input.glob("*.wav"))
    else:
        targets = [args.input]

    print(f"corpus band: p25-p75 = {st['p25']}-{st['p75']}Hz  "
          f"(p05-p95 = {st['p05']}-{st['p95']}Hz, median {st['median']}Hz)\n")
    print(f"{'file':<22}{'dur_s':>7}{'f0_med':>9}{'band':>7}{'rec.key':>9}")

    report, worst = [], "OK"
    order = {"OK": 0, "EDGE": 1, "OOD": 2}
    for p in targets:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        v = f0_frames(y, sr)
        if v.size < 30:
            print(f"{p.name:<22}{len(y)/sr:>7.1f}{'(unvoiced)':>9}")
            continue
        med = float(np.median(v))
        band, key = verdict(med, st)
        worst = band if order[band] > order[worst] else worst
        report.append(dict(file=p.name, dur_s=round(len(y) / sr, 1),
                           f0_median=round(med, 1), band=band, rec_key=key))
        mark = "" if band == "OK" else ("  <-- transpose!" if band == "OOD" else "")
        print(f"{p.name:<22}{len(y)/sr:>7.1f}{med:>9.1f}{band:>7}{key:>+9d}{mark}")

    if args.json_out:
        args.json_out.write_text(json.dumps(
            {"stats": st, "segments": report}, indent=2))
    print(f"\noverall: {worst}")
    return 0 if worst != "OOD" else 1


if __name__ == "__main__":
    sys.exit(main())
