"""otoya A/B: baseline vs TITAN, each built with MEDIAN-STFT ensemble
(the honoka lesson §11.3 — check if a male voice benefits too), plus single-
ckpt and v1 champion for reference. 8-segment source, flat index, ir=0.75.

    python _ab_titan.py                 # infer titan cache + build + measure
    python _ab_titan.py --skip-infer    # reuse caches
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
CHAR = PROJECT / "characters/otoya_sho_mix"
V3 = CHAR / "output/tokyo_summer_v3"
SEGS_DIR = V3 / "stage2_segments"
SRC40 = V3 / "stage1_40k/source_40k.wav"
INDEX = CHAR / "models_v2/flat_full_src_feat.index"
V1_FINAL = V3 / "stage5_rebuilt/FINAL_v2_vocals.wav"   # otoya v1-era single-ckpt final
OUT = V3 / "stage8_titan_ab"
N_FFT, HOP = 2048, 512
IR, PROTECT = 0.75, 0.33
NEW_CORPUS = CHAR / "data/v45_corpus"

ARMS = {"baseline": CHAR / "models_v2", "titan": CHAR / "models_v2_titan"}


def seg_offsets(full_src, sr):
    iv = librosa.effects.split(full_src, top_db=35, frame_length=2048, hop_length=512)
    merged, mg = [], int(3.0 * sr)
    for i, (s, e) in enumerate(iv):
        if i > 0 and s - iv[i-1][1] < mg:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return {i: int(s) for i, (s, e) in enumerate(merged) if (e - s) / sr >= 2.0}


def infer_arm(arm, mdir, seg_ids):
    cache = V3 / f"stage3_cache_v2_{arm}" if arm == "titan" else V3 / "stage3_cache_v2"
    if arm == "baseline" and cache.exists() and len(list(cache.glob("*.wav"))) >= 40:
        print(f"  {arm}: reuse existing cache ({len(list(cache.glob('*.wav')))} files)")
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(RVC_DIR)); os.chdir(str(RVC_DIR))
    for k, v in dict(weight_root="assets/weights", index_root="logs",
                     rmvpe_root="assets/rmvpe", weight_uvr5_root="assets/uvr5_weights").items():
        os.environ[k] = str(RVC_DIR / v)
    from dotenv import load_dotenv; load_dotenv(override=True)
    from infer.modules.vc.modules import VC
    from configs.config import Config
    import torch
    warnings.filterwarnings("ignore"); torch.set_grad_enabled(False)
    _s = sys.argv.copy(); sys.argv = ["x"]
    cfg = Config(); cfg.device = "cuda:0"; cfg.is_half = True
    sys.argv = _s
    vc = VC(cfg)
    for ck in sorted(mdir.glob("G_*_infer.pth"), key=lambda p: int(p.stem.split("_")[1])):
        cn = ck.stem.replace("_infer", "")
        wn = f"{arm}_{cn}.pth"; wd = RVC_DIR / "assets/weights" / wn
        if not wd.exists(): shutil.copy2(ck, wd)
        vc.get_vc(wn)
        for si in seg_ids:
            out = cache / f"s{si:03d}_{cn}.wav"
            if out.exists(): continue
            _, (sro, ao) = vc.vc_single(
                sid=0, input_audio_path=str(SEGS_DIR / f"seg_{si:03d}.wav"),
                f0_up_key=0, f0_file=None, f0_method="rmvpe",
                file_index=str(INDEX), file_index2=str(INDEX), index_rate=IR,
                filter_radius=7, resample_sr=0, rms_mix_rate=0.25, protect=PROTECT)
            sf.write(str(out), ao, sro)
        torch.cuda.empty_cache()
        print(f"  {arm} {cn}", flush=True)
    return cache


def spe(exp):
    return int(np.ceil(len((RVC_DIR / f"logs/{exp}/filelist.txt").read_text().splitlines()) / 16))


def median_track(cache, ckpts, seg_ids, starts, full_len, src_segs, sr):
    """Per-segment median-STFT ensemble, assembled into the full track."""
    full = np.zeros(full_len)
    for si in seg_ids:
        wavs = [cache / f"s{si:03d}_{cn}.wav" for cn in ckpts]
        wavs = [w for w in wavs if w.exists()]
        if not wavs: continue
        mags, L = [], None
        for w in wavs:
            y, _ = librosa.load(str(w), sr=None, mono=True)
            if L is None: L = len(y)
            mags.append(np.abs(librosa.stft(y[:L], n_fft=N_FFT, hop_length=HOP)))
        yp, _ = librosa.load(str(wavs[len(wavs)//2]), sr=None, mono=True)
        phase = np.angle(librosa.stft(yp[:L], n_fft=N_FFT, hop_length=HOP))
        T = min(m.shape[1] for m in mags)
        med = np.median(np.stack([m[:, :T] for m in mags]), axis=0)
        y = librosa.istft(med * np.exp(1j*phase[:, :T]), hop_length=HOP, length=L)
        ref = src_segs[si]; n = min(len(y), len(ref))
        y = y[:n] * (float(np.sqrt(np.mean(ref[:n]**2))) / (float(np.sqrt(np.mean(y[:n]**2)))+1e-12))
        fade = int(0.01*sr); t = np.linspace(0, np.pi/2, fade)
        y[:fade] *= np.sin(t); y[-fade:] *= np.cos(t)
        st = starts[si]; m = min(len(y), full_len - st)
        full[st:st+m] += y[:m]
    pk = float(np.max(np.abs(full)))
    return full/pk*0.99 if pk > 0.99 else full


def mfcc_mean(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048, hop_length=512).mean(axis=1)


def measure(y, sr, cm):
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    rms = librosa.feature.rms(S=S, frame_length=8192)[0]
    keep = rms > np.percentile(rms, 70)*0.4
    Sv = S[:, keep] if keep.sum() > 5 else S
    P = (Sv**2).mean(axis=1)
    logP = 10*np.log10(P+1e-20); k=12; pad=np.pad(logP,k,mode='edge')
    loc=np.array([np.median(pad[i:i+2*k+1]) for i in range(len(logP))])
    peaks=int(np.sum((f>=10000)&(f<20000)&(logP-loc>8)))
    S2=np.abs(librosa.stft(y,n_fft=4096,hop_length=1024))
    r2=librosa.feature.rms(S=S2,frame_length=4096)[0]
    vm=r2>np.percentile(r2,60)*0.4
    S2v=S2[:,vm] if vm.sum()>5 else S2
    ff=librosa.fft_frequencies(sr=sr,n_fft=4096); m=(ff>=300)&(ff<4000)
    ih=float(librosa.feature.spectral_flatness(S=S2v[m])[0].mean())
    roll=round(float(librosa.feature.spectral_rolloff(S=Sv,sr=sr,roll_percent=0.95).mean()))
    t=round(float(np.linalg.norm(mfcc_mean(y,sr)[1:]-cm[1:])),1)
    return dict(peaks=peaks, ih=round(ih,4), roll=roll, timbre=t)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-infer", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    full_src, sr = librosa.load(str(SRC40), sr=None, mono=True)
    seg_ids = sorted(int(p.stem.split("_")[1]) for p in SEGS_DIR.glob("seg_*.wav"))
    starts = seg_offsets(full_src, sr)
    src_segs = {si: librosa.load(str(SEGS_DIR/f"seg_{si:03d}.wav"), sr=None, mono=True)[0] for si in seg_ids}

    mfs = []
    for p in sorted(NEW_CORPUS.glob("*.wav"))[:12]:
        cy,_ = librosa.load(str(p), sr=40000, mono=True, offset=30, duration=40)
        if len(cy) > 40000: mfs.append(mfcc_mean(cy, 40000))
    cm = np.mean(mfs, axis=0)

    caches = {}
    for arm, mdir in ARMS.items():
        caches[arm] = (V3 / (f"stage3_cache_v2_titan" if arm=="titan" else "stage3_cache_v2")) \
            if args.skip_infer else infer_arm(arm, mdir, seg_ids)

    rows = {}
    for arm, mdir in ARMS.items():
        cache = caches[arm]
        exp = "otoya_sho_mix_v2" if arm=="baseline" else "otoya_sho_mix_v2_titan"
        s = spe(exp)
        ckpts = sorted({p.stem.split("_",1)[1] for p in cache.glob("s*_G_*.wav")},
                       key=lambda c:int(c.replace("G_","")))
        win = [c for c in ckpts if 95 <= int(c.replace("G_",""))//s <= 205]
        full = median_track(cache, win or ckpts, seg_ids, starts, len(full_src), src_segs, sr)
        sf.write(str(OUT/f"otoya_{arm}_MEDIAN.wav"), full.astype(np.float32), sr)
        rows[f"{arm} median"] = measure(full.astype(np.float32), sr, cm)
        print(f"{arm}: median of {len(win or ckpts)} ckpts -> otoya_{arm}_MEDIAN.wav", flush=True)

    for name, p in (("v1 final", V1_FINAL), ("source", SRC40)):
        if p.exists():
            y,_ = librosa.load(str(p), sr=None, mono=True)
            rows[name] = measure(y.astype(np.float32), sr, cm)

    print(f"\n{'':<16}{'peaks':>7}{'ihFlat':>8}{'roll':>7}{'timbre':>8}")
    for n, m in rows.items():
        print(f"{n:<16}{m['peaks']:>7}{m['ih']:>8}{m['roll']:>7}{m['timbre']:>8}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
