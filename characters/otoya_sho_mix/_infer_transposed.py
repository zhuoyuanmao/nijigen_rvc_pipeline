"""Batch re-inference of selected segments across all ckpts.

Originally written for the +12 verse transpose (see _transpose_probe.py:
the 0-48s verse sits an octave below the corpus register and +12 recovers
8.7-8.8dB of buzz). Generalised for the flat-index re-infer: the param
sweep (_param_sweep.py) showed the old IVF256/nprobe=1/10k-centroid index
had retrieval effectively disabled, and flat + ir=0.75 measurably pulls
timbre toward the corpus.

Model-major loop: load each ckpt once, run every segment through it. Per the
Liyuu KNOWHOW that is ~40x faster than reloading per segment.

    python _infer_transposed.py --segs 0 1 2 --key 12 \
        --index models/flat_full_src_feat.index --index-rate 0.75 \
        --cache-name stage3_cache_flat_key+12
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--song", default="tokyo_summer_v3")
ap.add_argument("--segs", type=int, nargs="+", required=True)
ap.add_argument("--key", type=int, required=True)
ap.add_argument("--index", default=None,
                help="index path relative to the character dir "
                     "(default: models/added_IVF256_...)")
ap.add_argument("--index-rate", type=float, default=0.50)
ap.add_argument("--protect", type=float, default=0.33)
ap.add_argument("--cache-name", default=None,
                help="cache dir name (default: stage3_cache_key<key>)")
CLI = ap.parse_args()

import numpy as np
import soundfile as sf
import torch

PROJECT = Path(__file__).resolve().parents[2]
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/otoya_sho_mix"
MODELS_DIR = CHAR / "models"
INDEX_PATH = (CHAR / CLI.index if CLI.index
              else MODELS_DIR / "added_IVF256_Flat_mi_baseline_src_feat.index")

BASE = CHAR / "output" / CLI.song
SEGS = BASE / "stage2_segments"
CACHE = BASE / (CLI.cache_name or f"stage3_cache_key{CLI.key:+d}")
CACHE.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(RVC_DIR))
os.chdir(str(RVC_DIR))
os.environ["weight_root"] = str(RVC_DIR / "assets" / "weights")
os.environ["index_root"] = str(RVC_DIR / "logs")
os.environ["rmvpe_root"] = str(RVC_DIR / "assets" / "rmvpe")
os.environ["weight_uvr5_root"] = str(RVC_DIR / "assets" / "uvr5_weights")
from dotenv import load_dotenv                                   # noqa: E402
load_dotenv(override=True)
from infer.modules.vc.modules import VC                          # noqa: E402
from configs.config import Config                                # noqa: E402

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

_saved = sys.argv.copy()
sys.argv = ["_infer_transposed.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _saved
vc = VC(config)

ckpts = sorted(MODELS_DIR.glob("G_*_infer.pth"),
               key=lambda p: int(p.stem.replace("G_", "").replace("_infer", "")))
seg_paths = [(si, SEGS / f"seg_{si:03d}.wav") for si in CLI.segs]
seg_paths = [(si, p) for si, p in seg_paths if p.exists()]
print(f"{len(ckpts)} ckpts x {len(seg_paths)} segments  key{CLI.key:+d}", flush=True)

done = 0
for ci, ck in enumerate(ckpts):
    cn = ck.stem.replace("_infer", "")
    wname = f"otoya_sho_{cn}.pth"
    wdst = RVC_DIR / "assets" / "weights" / wname
    if not wdst.exists():
        shutil.copy2(ck, wdst)
    try:
        vc.get_vc(wname)
    except Exception as e:
        print(f"  FAIL load {cn}: {e}", flush=True)
        continue
    t0 = time.time()
    for si, sp in seg_paths:
        outp = CACHE / f"s{si:03d}_{cn}.wav"
        if outp.exists():
            done += 1
            continue
        try:
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(sp), f0_up_key=CLI.key, f0_file=None,
                f0_method="rmvpe", file_index=str(INDEX_PATH),
                file_index2=str(INDEX_PATH), index_rate=CLI.index_rate,
                filter_radius=7, resample_sr=0, rms_mix_rate=0.25,
                protect=CLI.protect)
            sf.write(str(outp), ao, sro)
            done += 1
        except Exception as e:
            print(f"  FAIL seg={si} ckpt={cn}: {e}", flush=True)
    torch.cuda.empty_cache()
    print(f"  [{ci+1}/{len(ckpts)}] {cn}: {time.time()-t0:.1f}s", flush=True)

print(f"\n{done} files -> {CACHE}")
