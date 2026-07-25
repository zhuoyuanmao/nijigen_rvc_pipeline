"""Honoka: re-infer all ckpts against the flat index -> stage3_cache_flat.

Params from _param_sweep.py: flat index, ir=0.5, protect=0.33, key 0
(F0 verify: source is EDGE-high but −12 would leave the register; no shift).
Model-major loop, one segment.

    python _infer_flat.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import soundfile as sf
import torch

PROJECT = Path(__file__).resolve().parents[2]  # repo/project root
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/honoka"
MODELS_DIR = CHAR / "models"
INDEX_PATH = MODELS_DIR / "flat_full_src_feat.index"

BASE = CHAR / "output/tokyo_summer_v3"
SEGS = sorted((BASE / "stage2_segments").glob("seg_*.wav"))
CACHE = BASE / "stage3_cache_flat"
CACHE.mkdir(parents=True, exist_ok=True)

IR = 0.5
PROTECT = 0.33

sys.path.insert(0, str(RVC_DIR))
os.chdir(str(RVC_DIR))
os.environ["weight_root"] = str(RVC_DIR / "assets/weights")
os.environ["index_root"] = str(RVC_DIR / "logs")
os.environ["rmvpe_root"] = str(RVC_DIR / "assets/rmvpe")
os.environ["weight_uvr5_root"] = str(RVC_DIR / "assets/uvr5_weights")
from dotenv import load_dotenv                                   # noqa: E402
load_dotenv(override=True)
from infer.modules.vc.modules import VC                          # noqa: E402
from configs.config import Config                                # noqa: E402

warnings.filterwarnings("ignore")
torch.set_grad_enabled(False)

_s = sys.argv.copy()
sys.argv = ["_infer_flat.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _s
vc = VC(config)

ckpts = sorted(MODELS_DIR.glob("G_*_infer.pth"),
               key=lambda p: int(p.stem.split("_")[1]))
print(f"{len(ckpts)} ckpts x {len(SEGS)} segments  flat ir={IR}", flush=True)

for ci, ck in enumerate(ckpts):
    cn = ck.stem.replace("_infer", "")
    wname = f"honoka_{cn}.pth"
    wdst = RVC_DIR / "assets/weights" / wname
    if not wdst.exists():
        shutil.copy2(ck, wdst)
    try:
        vc.get_vc(wname)
    except Exception as e:
        print(f"  FAIL load {cn}: {e}", flush=True)
        continue
    t0 = time.time()
    for sp in SEGS:
        si = int(sp.stem.split("_")[1])
        outp = CACHE / f"s{si:03d}_{cn}.wav"
        if outp.exists():
            continue
        try:
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(sp), f0_up_key=0, f0_file=None,
                f0_method="rmvpe", file_index=str(INDEX_PATH),
                file_index2=str(INDEX_PATH), index_rate=IR, filter_radius=7,
                resample_sr=0, rms_mix_rate=0.25, protect=PROTECT)
            sf.write(str(outp), ao, sro)
        except Exception as e:
            print(f"  FAIL seg={si} ckpt={cn}: {e}", flush=True)
    torch.cuda.empty_cache()
    print(f"  [{ci+1}/{len(ckpts)}] {cn}: {time.time()-t0:.1f}s", flush=True)

print(f"\n-> {CACHE}")
