"""Flat FAISS index for honoka v2 — SELECTED slices' features only.

The index defines the retrieval timbre space; building it from the curated
slice set keeps rejected/dirty slices out of retrieval too. Shared by both
training arms (features are identical across arms).

    python _build_flat_index_v2.py --exp honoka_v2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import faiss
import numpy as np

RVC = Path(__file__).resolve().parents[2] / "Retrieval-based-Voice-Conversion-WebUI"
OUT = Path(__file__).resolve().parents[2] / "characters/honoka/models_v2/flat_full_src_feat.index"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="honoka_v2")
    args = ap.parse_args()

    exp = RVC / "logs" / args.exp
    feat_dir = exp / "3_feature768"
    keep_p = exp / "selected_slices.txt"
    keep = ({l.strip() for l in keep_p.read_text().splitlines() if l.strip()}
            if keep_p.exists() else None)

    t0 = time.time()
    files = sorted(feat_dir.glob("*.npy"))
    if keep:
        files = [f for f in files if f.stem in keep]
    print(f"{len(files)} feature files (selected only: {keep is not None})",
          flush=True)
    big = np.concatenate([np.load(str(p)) for p in files], axis=0).astype(np.float32)
    print(f"features: {big.shape} ({big.nbytes/1e9:.2f} GB)", flush=True)

    index = faiss.IndexFlatL2(big.shape[1])
    index.add(big)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT))
    print(f"-> {OUT}  ntotal={index.ntotal}  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
