"""Honoka: index / index_rate sweep with objective metrics (single segment).

Same rationale as otoya_sho_mix KNOWHOW §7.2 — the deployed IVF256 index had
retrieval effectively disabled, so ir needs re-tuning against the new flat
index. Honoka additions: her corpus is 4.38h (652,863 vectors) and her
source sits at the EDGE of her register (508Hz vs corpus p75 452Hz).

    python _param_sweep.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import torch

PROJECT = Path("/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore")
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/honoka"
MODELS_DIR = CHAR / "models"
CORPUS = CHAR / "data/v3_corpus/character_clean.wav"

INDEXES = {
    "flat": MODELS_DIR / "flat_full_src_feat.index",
    "ivf":  MODELS_DIR / "added_IVF256_Flat_mi_baseline_src_feat.index",
}
BASE = CHAR / "output/tokyo_summer_v3"
SEG = BASE / "stage2_segments/seg_000.wav"
OUT = BASE / "stage7_param_sweep"
OUT.mkdir(parents=True, exist_ok=True)

CKPT = "G_25740"                    # cleanest by 10-20k peak count (e110)
IRS = [0.0, 0.5, 0.75, 0.9]
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

print("corpus MFCC reference ...", flush=True)
mfs = []
for off in (600.0, 3000.0, 7000.0, 11000.0):
    try:
        cy, _ = librosa.load(str(CORPUS), sr=40000, mono=True,
                             offset=off, duration=60.0)
    except Exception:
        continue
    if len(cy) < 40000:
        continue
    mfs.append(librosa.feature.mfcc(y=cy, sr=40000, n_mfcc=20,
                                    n_fft=2048, hop_length=512).mean(axis=1))
CM = np.mean(mfs, axis=0)


def metrics(y, sr):
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    P = (S ** 2).mean(axis=1)

    def b(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(10 * np.log10(P[m].sum() + 1e-20))

    logP = 10 * np.log10(P + 1e-20)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    hf = (f >= 15500) & (f < 20000)         # her comb band is 16k+, not 12k+
    peaks = int(np.sum(hf & (logP - local > 8)))
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                              n_fft=2048, hop_length=512).mean(axis=1)
    return dict(junk16k=round(b(16000, 20000) - b(1000, 3000), 2),
                peaks16k=peaks,
                timbre=round(float(np.linalg.norm(mf[1:] - CM[1:])), 1))


_s = sys.argv.copy()
sys.argv = ["_param_sweep.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _s
vc = VC(config)

wname = f"honoka_{CKPT}.pth"
wdst = RVC_DIR / "assets/weights" / wname
if not wdst.exists():
    shutil.copy2(MODELS_DIR / f"{CKPT}_infer.pth", wdst)
vc.get_vc(wname)

results = []
for iname, ipath in INDEXES.items():
    for ir in IRS:
        if ir == 0.0 and iname == "ivf":
            continue
        tag = f"{CKPT}_{iname}_ir{ir}"
        t0 = time.time()
        try:
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(SEG), f0_up_key=0, f0_file=None,
                f0_method="rmvpe", file_index=str(ipath),
                file_index2=str(ipath), index_rate=ir, filter_radius=7,
                resample_sr=0, rms_mix_rate=0.25, protect=PROTECT)
        except Exception as e:
            print(f"  FAIL {tag}: {e}", flush=True)
            continue
        sf.write(str(OUT / f"{tag}.wav"), ao, sro)
        y = ao.astype(np.float32)
        y /= max(np.max(np.abs(y)), 1e-9)
        m = metrics(y * 0.9, sro)
        m.update(index=iname, ir=ir, secs=round(time.time() - t0, 1))
        results.append(m)
        print(f"  {tag}: junk16k {m['junk16k']:+.1f}  peaks {m['peaks16k']:>3}  "
              f"timbre {m['timbre']}  ({m['secs']}s)", flush=True)

(OUT / "sweep_report.json").write_text(json.dumps(results, indent=2))
print(f"\n{'index':<6}{'ir':>6}{'junk16k':>9}{'peaks':>7}{'timbre':>8}")
for m in sorted(results, key=lambda r: r["timbre"]):
    print(f"{m['index']:<6}{m['ir']:>6}{m['junk16k']:>9}"
          f"{m['peaks16k']:>7}{m['timbre']:>8}")
