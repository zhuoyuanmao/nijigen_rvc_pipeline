"""Build a full-resolution Flat FAISS index for honoka.

Same fix as otoya_sho_mix (KNOWHOW §7.1): the deployed index holds only
10k k-means centroids searched at IVF256/nprobe=1 — retrieval effectively
disabled. Flat = exact search over every training feature vector.

    python _build_flat_index.py
"""
from pathlib import Path
import time

import faiss
import numpy as np

import os
PROJECT = Path(__file__).resolve().parents[2]
EXT4_FEATS = Path(os.environ.get("RVC_DATA_DIR", Path.home() / "rvc_data")) / \
    "honoka_v1/logs/3_feature768"
OUT = PROJECT / "characters/honoka/models/flat_full_src_feat.index"

t0 = time.time()
files = sorted(EXT4_FEATS.glob("*.npy"))
print(f"{len(files)} feature files", flush=True)

chunks = []
for i, p in enumerate(files):
    chunks.append(np.load(str(p)))
    if (i + 1) % 500 == 0:
        print(f"  loaded {i+1}/{len(files)}", flush=True)
big = np.concatenate(chunks, axis=0).astype(np.float32)
del chunks
print(f"features: {big.shape}  ({big.nbytes/1e9:.2f} GB)  "
      f"load {time.time()-t0:.0f}s", flush=True)

index = faiss.IndexFlatL2(big.shape[1])
index.add(big)
faiss.write_index(index, str(OUT))
print(f"-> {OUT}  ntotal={index.ntotal}  "
      f"({OUT.stat().st_size/1e9:.2f} GB)  total {time.time()-t0:.0f}s")
