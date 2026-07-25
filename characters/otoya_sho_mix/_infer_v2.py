"""otoya_sho_mix_v2 batch inference: all 15 ckpts x all segments.

Uses models_v2/ (the pretrained-warm-started, clean-corpus model) and the
v2 flat index. Same batch structure as the v1 _infer_transposed.py.

    python _infer_v2.py --key 0  --cache-name stage3_cache_v2
    python _infer_v2.py --segs 0 1 2 --key 12 --cache-name stage3_cache_v2_key+12
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
ap.add_argument("--segs", type=int, nargs="*", default=None)
ap.add_argument("--key", type=int, default=0)
ap.add_argument("--index-rate", type=float, default=0.75)
ap.add_argument("--protect", type=float, default=0.33)
ap.add_argument("--cache-name", default="stage3_cache_v2")
CLI = ap.parse_args()

import soundfile as sf
import torch

PROJECT = Path(__file__).resolve().parents[2]  # repo/project root
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/otoya_sho_mix"
MODELS_DIR = CHAR / "models_v2"
INDEX_PATH = MODELS_DIR / "flat_full_src_feat.index"

BASE = CHAR / "output" / CLI.song
SEGS_DIR = BASE / "stage2_segments"
CACHE = BASE / CLI.cache_name
CACHE.mkdir(parents=True, exist_ok=True)

seg_ids = (CLI.segs if CLI.segs is not None
           else sorted(int(p.stem.split("_")[1])
                       for p in SEGS_DIR.glob("seg_*.wav")))
SEGS = [(si, SEGS_DIR / f"seg_{si:03d}.wav") for si in seg_ids]
SEGS = [(si, p) for si, p in SEGS if p.exists()]

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
sys.argv = ["_infer_v2.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _s
vc = VC(config)

ckpts = sorted(MODELS_DIR.glob("G_*_infer.pth"),
               key=lambda p: int(p.stem.split("_")[1]))
print(f"{len(ckpts)} ckpts x {len(SEGS)} segments  v2  key{CLI.key:+d}  "
      f"ir={CLI.index_rate}", flush=True)

for ci, ck in enumerate(ckpts):
    cn = ck.stem.replace("_infer", "")
    wname = f"osv2_{cn}.pth"
    wdst = RVC_DIR / "assets/weights" / wname
    if not wdst.exists():
        shutil.copy2(ck, wdst)
    try:
        vc.get_vc(wname)
    except Exception as e:
        print(f"  FAIL load {cn}: {e}", flush=True)
        continue
    t0 = time.time()
    for si, sp in SEGS:
        outp = CACHE / f"s{si:03d}_{cn}.wav"
        if outp.exists():
            continue
        try:
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(sp), f0_up_key=CLI.key, f0_file=None,
                f0_method="rmvpe", file_index=str(INDEX_PATH),
                file_index2=str(INDEX_PATH), index_rate=CLI.index_rate,
                filter_radius=7, resample_sr=0, rms_mix_rate=0.25,
                protect=CLI.protect)
            sf.write(str(outp), ao, sro)
        except Exception as e:
            print(f"  FAIL seg={si} ckpt={cn}: {e}", flush=True)
    torch.cuda.empty_cache()
    print(f"  [{ci+1}/{len(ckpts)}] {cn}: {time.time()-t0:.1f}s", flush=True)

print(f"\n-> {CACHE}")
