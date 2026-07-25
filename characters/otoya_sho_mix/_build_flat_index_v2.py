"""Full Flat FAISS index for otoya_sho_mix_v2 (from the v4.5-corpus features).

Same rationale as v1 (_build_flat_index.py): exact search over every training
feature vector, not the 10k-centroid IVF256/nprobe=1 that had retrieval
effectively off.

    python _build_flat_index_v2.py
"""
from pathlib import Path
import time

import faiss
import numpy as np

import os
PROJECT = Path(__file__).resolve().parents[2]
FEATS = Path(os.environ.get("RVC_DATA_DIR", Path.home() / "rvc_data")) / \
    "otoya_sho_mix_v2/logs/3_feature768"
OUT = PROJECT / "characters/otoya_sho_mix/models_v2/flat_full_src_feat.index"
OUT.parent.mkdir(parents=True, exist_ok=True)

t0 = time.time()
files = sorted(FEATS.glob("*.npy"))
print(f"{len(files)} feature files", flush=True)
chunks = [np.load(str(p)) for p in files]
big = np.concatenate(chunks, axis=0).astype(np.float32)
del chunks
print(f"features: {big.shape}  ({big.nbytes/1e9:.2f} GB)  "
      f"load {time.time()-t0:.0f}s", flush=True)

index = faiss.IndexFlatL2(big.shape[1])
index.add(big)
faiss.write_index(index, str(OUT))
print(f"-> {OUT}  ntotal={index.ntotal}  "
      f"({OUT.stat().st_size/1e9:.2f} GB)  {time.time()-t0:.0f}s")
