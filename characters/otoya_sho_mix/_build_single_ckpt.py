"""Build a whole-track vocal from ONE optimally-chosen ckpt.

Rationale: after the pretrain-warm-started v2 retrain the model converged by
~epoch 20 and every ckpt is uniformly clean (0 tonal peaks everywhere), so
the per-segment single-ckpt selection in _rebuild_v4 is choosing between
near-equivalent candidates and its winners scatter across e20-e260 with no
epoch trend. This asks the simpler question: which SINGLE ckpt is best on
average across the whole song, and how does using it everywhere compare?

Ranks all ckpts by mean artifact score, picks the best (ties broken toward
mid-epochs, which generalise better than the very first/last), then assembles
the track with that one ckpt — RAW (no spectral repair, matching
FINAL_v2_vocals): per-segment RMS + equal-power crossfades only.

    python _build_single_ckpt.py --cache-name stage3_cache_v2 --suffix _v2_single
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from _rebuild_v4 import artifact_score, equal_power_fade

CHAR = Path(__file__).resolve().parent
STEPS_PER_EPOCH = 127
MID_LO, MID_HI = 80, 240        # "mid-epoch" preference band for tie-breaking


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="tokyo_summer_v3")
    ap.add_argument("--cache-name", default="stage3_cache_v2")
    ap.add_argument("--suffix", default="_v2_single")
    ap.add_argument("--force-ckpt", default=None,
                    help="e.g. G_17780 to override the automatic pick")
    args = ap.parse_args()

    base = CHAR / "output" / args.song
    segs_dir = base / "stage2_segments"
    cache = base / args.cache_name
    src_40k = base / "stage1_40k/source_40k.wav"
    out_dir = base / "stage5_rebuilt"

    full_src, sr = librosa.load(str(src_40k), sr=None, mono=True)
    seg_ids = sorted(int(p.stem.split("_")[1]) for p in segs_dir.glob("seg_*.wav"))
    ckpts = sorted({p.stem.split("_", 1)[1] for p in cache.glob("s*_G_*.wav")},
                   key=lambda s: int(s.replace("G_", "")))

    # split offsets (same as _rebuild_v4)
    iv = librosa.effects.split(full_src, top_db=35, frame_length=2048, hop_length=512)
    merged = []
    for i, (s, e) in enumerate(iv):
        if i > 0 and s - iv[i - 1][1] < int(3.0 * sr):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    starts = {i: int(s) for i, (s, e) in enumerate(merged) if (e - s) / sr >= 2.0}

    # score every ckpt on every segment
    print(f"scoring {len(ckpts)} ckpts x {len(seg_ids)} segments ...", flush=True)
    scores = {cn: {} for cn in ckpts}
    src_segs = {}
    for si in seg_ids:
        s, _ = librosa.load(str(segs_dir / f"seg_{si:03d}.wav"), sr=None, mono=True)
        src_segs[si] = s
        for cn in ckpts:
            p = cache / f"s{si:03d}_{cn}.wav"
            if not p.exists():
                continue
            y, _ = librosa.load(str(p), sr=None, mono=True)
            n = min(len(y), len(s))
            scores[cn][si] = artifact_score(y[:n], s[:n], sr)["total"]

    # mean score per ckpt (lower = better)
    ranking = []
    for cn in ckpts:
        vals = [scores[cn][si] for si in seg_ids if si in scores[cn]]
        ep = int(cn.replace("G_", "")) // STEPS_PER_EPOCH
        ranking.append((cn, ep, float(np.mean(vals)), float(np.std(vals))))
    ranking.sort(key=lambda r: r[2])

    print(f"\n{'ckpt':>9} {'epoch':>6} {'mean':>8} {'std':>7}")
    for cn, ep, m, sd in ranking:
        star = "  <-- mid" if MID_LO <= ep <= MID_HI else ""
        print(f"{cn:>9} {ep:>6} {m:>8.3f} {sd:>7.3f}{star}")

    if args.force_ckpt:
        pick = args.force_ckpt
    else:
        # best overall; if the top pick is an epoch extreme and a mid-epoch
        # ckpt is within 3% of it, prefer the mid-epoch one (better general.)
        best = ranking[0]
        pick = best[0]
        if not (MID_LO <= best[1] <= MID_HI):
            for cn, ep, m, sd in ranking:
                if MID_LO <= ep <= MID_HI and m <= best[2] * 1.03:
                    pick = cn
                    print(f"\n(top pick e{best[1]} is an extreme; mid-epoch {cn} "
                          f"(e{ep}) is within 3% -> preferred)")
                    break
    pick_ep = int(pick.replace("G_", "")) // STEPS_PER_EPOCH
    print(f"\n=> single ckpt: {pick} (e{pick_ep})")

    # assemble whole track from this one ckpt, RAW
    full = np.zeros(len(full_src))
    for si in seg_ids:
        p = cache / f"s{si:03d}_{pick}.wav"
        if not p.exists():
            print(f"  seg_{si:03d}: {pick} MISSING, skipped")
            continue
        y, _ = librosa.load(str(p), sr=None, mono=True)
        ref = src_segs[si]
        n = min(len(y), len(ref))
        y = y[:n].astype(np.float64)
        tgt = float(np.sqrt(np.mean(ref[:n] ** 2)))
        y *= tgt / (float(np.sqrt(np.mean(y ** 2))) + 1e-12)
        y = equal_power_fade(y, sr)
        st = starts[si]
        m = min(len(y), len(full) - st)
        full[st:st + m] += y[:m]

    peak = float(np.max(np.abs(full)))
    if peak > 0.99:
        full = full / peak * 0.99
    out = out_dir / f"vocals_rebuilt{args.suffix}.wav"
    sf.write(str(out), full.astype(np.float32), sr)
    (out_dir / f"single_ckpt_report{args.suffix}.json").write_text(json.dumps(
        {"pick": pick, "epoch": pick_ep,
         "ranking": [{"ckpt": c, "epoch": e, "mean": round(m, 3)}
                     for c, e, m, _ in ranking]}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
