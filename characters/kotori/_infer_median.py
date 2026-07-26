"""kotori v2 inference: infer every e100-200 ckpt on the female source seg,
then median-STFT ensemble (bright female voice -> median is the right call,
see honoka KNOWHOW §11.3), then breath handling -> FINAL.

Mirrors honoka's flow but single-arm (TITAN only).

    python _infer_median.py                 # full: infer + median + breathmute
    python _infer_median.py --skip-infer    # reuse cached per-ckpt wavs
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[2]
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/kotori"
V3 = CHAR / "output/tokyo_summer_v3"
SEG = V3 / "stage2_segments/seg_000.wav"
SRC_FULL = V3 / "stage1_40k/song_clean_lead_40k.wav"
MODELS = CHAR / "models_v2"
INDEX = MODELS / "flat_full_src_feat.index"
CACHE = V3 / "stage8_infer"
N_FFT, HOP = 2048, 512
IR, PROTECT = 0.5, 0.33


def steps_per_epoch():
    fl = RVC_DIR / "logs/kotori_v2/filelist.txt"
    return int(np.ceil(len(fl.read_text().splitlines()) / 16))


def infer_all():
    CACHE.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RVC_DIR)); os.chdir(str(RVC_DIR))
    os.environ["weight_root"] = str(RVC_DIR / "assets/weights")
    os.environ["index_root"] = str(RVC_DIR / "logs")
    os.environ["rmvpe_root"] = str(RVC_DIR / "assets/rmvpe")
    os.environ["weight_uvr5_root"] = str(RVC_DIR / "assets/uvr5_weights")
    from dotenv import load_dotenv; load_dotenv(override=True)
    from infer.modules.vc.modules import VC
    from configs.config import Config
    import torch
    warnings.filterwarnings("ignore"); torch.set_grad_enabled(False)
    _s = sys.argv.copy(); sys.argv = ["x"]
    cfg = Config(); cfg.device = "cuda:0"; cfg.is_half = True
    sys.argv = _s
    vc = VC(cfg)

    spe = steps_per_epoch()
    src_full, sr = librosa.load(str(SRC_FULL), sr=None, mono=True)
    seg_src, _ = librosa.load(str(SEG), sr=None, mono=True)
    st = int(0.26 * sr)
    for ck in sorted(MODELS.glob("G_*_infer.pth"),
                     key=lambda p: int(p.stem.split("_")[1])):
        step = int(ck.stem.split("_")[1]); ep = round(step / spe)
        if ep < 95:
            continue
        out = CACHE / f"{ck.stem}_e{ep}.wav"
        if out.exists():
            continue
        wname = f"kotori_{ck.stem}.pth"
        wdst = RVC_DIR / "assets/weights" / wname
        if not wdst.exists():
            shutil.copy2(ck, wdst)
        vc.get_vc(wname)
        _, (sro, ao) = vc.vc_single(
            sid=0, input_audio_path=str(SEG), f0_up_key=0, f0_file=None,
            f0_method="rmvpe", file_index=str(INDEX), file_index2=str(INDEX),
            index_rate=IR, filter_radius=7, resample_sr=0, rms_mix_rate=0.25,
            protect=PROTECT)
        y = ao.astype(np.float64)
        nmin = min(len(y), len(seg_src)); y = y[:nmin]
        tgt = float(np.sqrt(np.mean(seg_src[:nmin] ** 2)))
        y *= tgt / (float(np.sqrt(np.mean(y ** 2))) + 1e-12)
        full = np.zeros(len(src_full)); m = min(len(y), len(full) - st)
        full[st:st + m] = y[:m]
        sf.write(str(out), full.astype(np.float32), sr)
        print(f"  {ck.stem} (e{ep})", flush=True)
        torch.cuda.empty_cache()


def median_ensemble():
    wavs = sorted(CACHE.glob("G_*_e*.wav"))
    mags, L = [], None
    for p in wavs:
        y, sr = librosa.load(str(p), sr=None, mono=True)
        if L is None:
            L = len(y)
        mags.append(np.abs(librosa.stft(y[:L], n_fft=N_FFT, hop_length=HOP)))
    ph = next((w for w in wavs if "_e182" in w.name or "_e180" in w.name), wavs[-1])
    yp, sr = librosa.load(str(ph), sr=None, mono=True)
    phase = np.angle(librosa.stft(yp[:L], n_fft=N_FFT, hop_length=HOP))
    T = min(m.shape[1] for m in mags)
    med = np.median(np.stack([m[:, :T] for m in mags]), axis=0)
    y = librosa.istft(med * np.exp(1j * phase[:, :T]), hop_length=HOP, length=L)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = y / pk * 0.99
    print(f"median of {len(wavs)} ckpts, phase={ph.name}")
    return y, sr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-infer", action="store_true")
    args = ap.parse_args()
    if not args.skip_infer:
        print("=== infer e100-200 ckpts ===")
        infer_all()
    print("=== median ensemble ===")
    y, sr = median_ensemble()
    reb = V3 / "stage5_rebuilt"
    reb.mkdir(parents=True, exist_ok=True)
    med_p = reb / "kotori_MEDIAN.wav"
    sf.write(str(med_p), y.astype(np.float32), sr)
    # breath mute (Kevin's honoka choice); keep a with-breath version too
    from importlib import import_module
    sys.path.insert(0, str(CHAR))
    print(f"-> {med_p}")
    print("run _remove_breaths.py --build stage5_rebuilt/kotori_MEDIAN.wav "
          "--duck-db -99  for the muted final")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
