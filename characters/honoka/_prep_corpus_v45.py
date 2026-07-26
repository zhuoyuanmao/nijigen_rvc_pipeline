"""honoka v4.5 training-corpus re-prep (58 raw tracks, single voice).

Same chain as otoya_sho_mix (METHODOLOGY §3): per raw track
  ffmpeg 44.1k -> 3x Roformer vocals on the mix -> gated min/median blend
  (loud = min-|x| removes backing, quiet = median keeps breaths)
  -> anvuew dereverb -> trim + LUFS -20 -> data/v45_corpus/honoka_<idx>.wav

Differences vs otoya: one voice only (no 2:1 duplication), 58 tracks.
Functions are imported from the otoya reference implementation.

    python _prep_corpus_v45.py            # all 58 tracks
    python _prep_corpus_v45.py --limit 1  # smoke test
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

CHAR = Path(__file__).resolve().parent
sys.path.insert(0, str(CHAR.parent / "otoya_sho_mix"))
from _prep_source_v4 import (separator_pass, VOCALS_MODELS,           # noqa: E402
                             DEREVERB_MODEL, DEREVERB_STEM)
from _prep_corpus_v45 import gated_min_median, trim_lufs, run         # noqa: E402

RAW_DIR = Path(os.environ.get(
    "RAW_DIR", str(Path(__file__).resolve().parents[2] / "raw"))) / "honoka_raw"
OUT = CHAR / "data/v45_corpus"
WORK = CHAR / "data/v45_work"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    tracks = sorted(RAW_DIR.glob("*.wav"))
    if args.limit:
        tracks = tracks[:args.limit]
    print(f"honoka: {len(tracks)} raw tracks from {RAW_DIR}", flush=True)

    t_all, total = time.time(), 0.0
    for i, src in enumerate(tracks):
        tag = f"honoka_{i:02d}"
        final = OUT / f"{tag}.wav"
        if final.exists() and final.stat().st_size > 0:
            print(f"[skip] {tag}", flush=True)
            continue
        t0 = time.time()
        print(f"\n--- {tag}: {src.name[:60]} ---", flush=True)
        w = WORK / tag
        w.mkdir(parents=True, exist_ok=True)

        src44 = w / "src44.wav"
        if not src44.exists():
            run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
                 "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", str(src44)])

        stems = [separator_pass(mf, stem, src44, w / f"voc_{t}.wav")
                 for mf, stem, t in VOCALS_MODELS]
        blend = gated_min_median(stems, w / "blend.wav")
        drv = separator_pass(DEREVERB_MODEL, DEREVERB_STEM, blend,
                             w / "dereverb.wav")
        info = trim_lufs(drv, final)
        total += info["dur_s"]
        print(f"  => {final.name}  {info['dur_s']}s   ({time.time()-t0:.0f}s)",
              flush=True)

    print(f"\n=== DONE in {(time.time()-t_all)/60:.1f} min ===")
    print(f"{len(list(OUT.glob('*.wav')))} files ({total:.0f}s new) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
