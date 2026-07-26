"""Tier-2 probe: does transposing the low male verse into the model's
trained register remove the artifacts?

Measured facts that motivate this:
  * otoya_sho_mix corpus F0: median 248.3Hz, p05 173.5, p95 363.5
  * this song's 0-48s verse: F0 median ~127Hz, MFCC distance to corpus 86.4
    (the 68s+ chorus sits at 245Hz, distance 30.9 — i.e. in-distribution)
  * seg_000 best achievable buzz -16.0dB vs seg_006's -27.9dB

So the verse is an octave below anything the model ever saw. This sweeps
f0_up_key over the low segments and reports the artifact metrics, so the
musical cost (a key/octave change) can be weighed against the quality gain.

    python _transpose_probe.py --segs 0 1 2 --keys 0 5 7 12
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import warnings
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--song", default="tokyo_summer_v3")
ap.add_argument("--segs", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--keys", type=int, nargs="+", default=[0, 5, 7, 12])
ap.add_argument("--ckpt", default="G_18900")
CLI = ap.parse_args()

import numpy as np
import soundfile as sf
import librosa
import torch

PROJECT = Path(__file__).resolve().parents[2]
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/otoya_sho_mix"
MODELS_DIR = CHAR / "models"
INDEX_PATH = MODELS_DIR / "added_IVF256_Flat_mi_baseline_src_feat.index"

BASE = CHAR / "output" / CLI.song
SEGS = BASE / "stage2_segments"
OUT = BASE / "stage6_transpose_probe"
OUT.mkdir(parents=True, exist_ok=True)

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


def avg_spectrum(y, sr, n_fft=4096):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4))
    f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    rms = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    if rms.size > 8:
        k = rms > np.percentile(rms, 70) * 0.4
        if k.sum() > 4:
            S = S[:, k]
    return S.mean(axis=1) + 1e-12, f


def report(y, sr):
    P, f = avg_spectrum(y, sr)

    def b(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(20 * np.log10(P[m].mean() + 1e-12))

    logP = 20 * np.log10(P)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    local = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    prom = logP - local
    hf = (f >= 10000) & (f < 20000)
    peaks = int(np.sum(hf & (prom > 8)))

    f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=65.0, fmax=1000.0,
                            sr=sr, frame_length=2048, hop_length=1024)
    v = f0[np.isfinite(f0)]
    v = v[(v > 66) & (v < 990)]
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    return dict(
        buzz=round(b(12000, 20000) - b(1000, 3000), 2),
        mid_hf=round(b(8000, 12000) - b(1000, 3000), 2),
        peaks=peaks,
        flat=round(float(librosa.feature.spectral_flatness(S=S)[0].mean()), 5),
        f0_median=round(float(np.median(v)), 1) if v.size else None,
    )


_saved = sys.argv.copy()
sys.argv = ["_transpose_probe.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _saved
vc = VC(config)

src_ck = MODELS_DIR / f"{CLI.ckpt}_infer.pth"
wname = f"otoya_sho_{CLI.ckpt}.pth"
wdst = RVC_DIR / "assets" / "weights" / wname
if not wdst.exists():
    shutil.copy2(src_ck, wdst)
vc.get_vc(wname)
print(f"loaded {wname}", flush=True)

results = {}
for si in CLI.segs:
    sp = SEGS / f"seg_{si:03d}.wav"
    if not sp.exists():
        print(f"  seg_{si:03d} missing"); continue
    ysrc, sr = librosa.load(str(sp), sr=None, mono=True)
    results[f"seg_{si:03d}"] = {"SOURCE": report(ysrc, sr)}
    for key in CLI.keys:
        t0 = time.time()
        try:
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(sp), f0_up_key=key, f0_file=None,
                f0_method="rmvpe", file_index=str(INDEX_PATH),
                file_index2=str(INDEX_PATH), index_rate=0.50,
                filter_radius=7, resample_sr=0, rms_mix_rate=0.25, protect=0.33)
        except Exception as e:
            print(f"  seg_{si:03d} key{key:+d}: FAIL {e}", flush=True)
            continue
        outp = OUT / f"s{si:03d}_key{key:+d}.wav"
        sf.write(str(outp), ao, sro)
        y = ao.astype(np.float64)
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y)) * 0.9
        r = report(y.astype(np.float32), sro)
        results[f"seg_{si:03d}"][f"key{key:+d}"] = r
        print(f"  seg_{si:03d} key{key:+d}: buzz {r['buzz']:+.2f}  "
              f"peaks {r['peaks']:>3}  f0 {r['f0_median']}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    torch.cuda.empty_cache()

(OUT / "probe_report.json").write_text(json.dumps(results, indent=2))
print("\n" + "=" * 70)
print(json.dumps(results, indent=2))
print(f"\n-> {OUT}")
