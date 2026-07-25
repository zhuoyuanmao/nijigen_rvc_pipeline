"""Sweep index / index_rate / protect with objective metrics, on the GPU.

Motivation: every stage3_cache inference so far ran against an index that
holds only 10k k-means centroids searched at nprobe=1 (~39 vectors/query) —
retrieval was effectively off. flat_full_src_feat.index (354,701 vectors,
exact search) may shift both timbre and artifact behaviour, so index_rate
and protect need re-tuning against it before the full re-infer.

Metrics per run:
  buzz       12-20k minus 1-3k energy, dB (lower = cleaner)
  peaks      narrowband tonal peaks 10-20k (the audible whine)
  timbre     MFCC L2 distance to the training corpus (lower = more otoya/sho)

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
CHAR = PROJECT / "characters/otoya_sho_mix"
MODELS_DIR = CHAR / "models"

INDEXES = {
    "flat": MODELS_DIR / "flat_full_src_feat.index",
    "ivf":  MODELS_DIR / "added_IVF256_Flat_mi_baseline_src_feat.index",
}
BASE = CHAR / "output/tokyo_summer_v3"
SEGS = BASE / "stage2_segments"
OUT = BASE / "stage7_param_sweep"
OUT.mkdir(parents=True, exist_ok=True)

# (segment, transpose, ckpt) — the measured winners for verse / chorus
CASES = [(0, 12, "G_17640"), (4, 0, "G_18900")]
IRS = [0.0, 0.25, 0.5, 0.75]
PROTECTS = [0.33, 0.5]

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


def metrics(y, sr, corpus_mfcc):
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    rms = librosa.feature.rms(S=S, frame_length=8192)[0]
    keep = rms > np.percentile(rms, 70) * 0.4
    Sv = S[:, keep] if keep.sum() > 5 else S
    P = (Sv ** 2).mean(axis=1)

    def b(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(10 * np.log10(P[m].sum() + 1e-20))

    logP = 10 * np.log10(P + 1e-20)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    hf = (f >= 10000) & (f < 20000)
    peaks = int(np.sum(hf & (logP - local > 8)))

    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20,
                              n_fft=2048, hop_length=512).mean(axis=1)
    timbre = float(np.linalg.norm(mf[1:] - corpus_mfcc[1:]))
    return dict(buzz=round(b(12000, 20000) - b(1000, 3000), 2),
                peaks=peaks, timbre=round(timbre, 1))


print("computing corpus MFCC reference ...", flush=True)
cy, _ = librosa.load(str(CHAR / "data/character_clean.wav"),
                     sr=40000, mono=True, offset=600.0, duration=120.0)
CORPUS_MFCC = librosa.feature.mfcc(y=cy, sr=40000, n_mfcc=20,
                                   n_fft=2048, hop_length=512).mean(axis=1)
del cy

_saved = sys.argv.copy()
sys.argv = ["_param_sweep.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _saved
vc = VC(config)

results = []
loaded = None
for si, key, cn in CASES:
    wname = f"otoya_sho_{cn}.pth"
    wdst = RVC_DIR / "assets" / "weights" / wname
    if not wdst.exists():
        shutil.copy2(MODELS_DIR / f"{cn}_infer.pth", wdst)
    if loaded != wname:
        vc.get_vc(wname)
        loaded = wname
    sp = SEGS / f"seg_{si:03d}.wav"
    for iname, ipath in INDEXES.items():
        for ir in IRS:
            for prot in PROTECTS:
                if ir == 0.0 and iname == "ivf":
                    continue                      # ir=0 makes index moot
                tag = f"s{si}_k{key}_{cn}_{iname}_ir{ir}_p{prot}"
                outp = OUT / f"{tag}.wav"
                t0 = time.time()
                try:
                    _, (sro, ao) = vc.vc_single(
                        sid=0, input_audio_path=str(sp), f0_up_key=key,
                        f0_file=None, f0_method="rmvpe",
                        file_index=str(ipath), file_index2=str(ipath),
                        index_rate=ir, filter_radius=7, resample_sr=0,
                        rms_mix_rate=0.25, protect=prot)
                except Exception as e:
                    print(f"  FAIL {tag}: {e}", flush=True)
                    continue
                sf.write(str(outp), ao, sro)
                y = ao.astype(np.float32)
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y)) * 0.9
                m = metrics(y, sro, CORPUS_MFCC)
                m.update(seg=si, key=key, ckpt=cn, index=iname,
                         ir=ir, protect=prot, secs=round(time.time() - t0, 1))
                results.append(m)
                print(f"  {tag}: buzz {m['buzz']:+.1f}  peaks {m['peaks']:>3}  "
                      f"timbre {m['timbre']:.1f}  ({m['secs']}s)", flush=True)
    torch.cuda.empty_cache()

(OUT / "sweep_report.json").write_text(json.dumps(results, indent=2))
print("\nseg key index   ir    prot   buzz  peaks timbre")
for m in sorted(results, key=lambda r: (r["seg"], r["timbre"])):
    print(f"{m['seg']:>3} {m['key']:>3} {m['index']:<6} {m['ir']:<5} "
          f"{m['protect']:<6} {m['buzz']:>6} {m['peaks']:>5} {m['timbre']:>6}")
