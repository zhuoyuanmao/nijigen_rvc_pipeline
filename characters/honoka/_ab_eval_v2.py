"""honoka v2 A/B: baseline vs TITAN arm vs v1 best — objective judgement.

For each arm: infer seg_000 with every deployed ckpt in the e100-200 window
(flat v2 index, ir=0.5), assemble the whole track per ckpt (single-ckpt,
raw), measure, pick the arm's best. Then compare arms + v1 champion.

Metrics (the ones that mattered for honoka v1):
  peaks>15.5k   narrowband whistle count       (v1 ensemble: 57)
  air12_15.5k   real-air band vs source        (v1 deficit: −30 vs −13)
  junk16_20k    invented top end
  rolloff95, dynR, timbre (MFCC vs NEW corpus)
  breath_whistle frames (per-frame tonal prominence in breath regions)

    python _ab_eval_v2.py
"""
from __future__ import annotations

import json
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
CHAR = PROJECT / "characters/honoka"
V3 = CHAR / "output/tokyo_summer_v3"
SRC_FULL = V3 / "stage1_40k/song_clean_lead_40k.wav"
SEG = V3 / "stage2_segments/seg_000.wav"
INDEX = CHAR / "models_v2/flat_full_src_feat.index"
OUT = V3 / "stage8_v2_ab"
V1_BEST = V3 / "stage5_rebuilt/vocals_rebuilt_flat_medens_breathfix.wav"

ARMS = {"baseline": CHAR / "models_v2", "titan": CHAR / "models_v2_titan"}
IR, PROTECT = 0.5, 0.33
NEW_CORPUS = CHAR / "data/v45_corpus"

# ---------------------------------------------------------------- metrics
def mfcc_mean(y, sr):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=2048,
                               hop_length=512).mean(axis=1)


def corpus_ref():
    mfs = []
    for p in sorted(NEW_CORPUS.glob("*.wav"))[:16]:
        y, _ = librosa.load(str(p), sr=40000, mono=True, offset=30, duration=40)
        if len(y) > 40000:
            mfs.append(mfcc_mean(y, 40000))
    return np.mean(mfs, axis=0)


def breath_regions(src, sr):
    S = np.abs(librosa.stft(src, n_fft=2048, hop_length=512))
    rms = librosa.feature.rms(S=S, frame_length=2048)[0]
    loud = np.percentile(rms, 75)
    quiet = (rms > loud * 0.015) & (rms < loud * 0.25)
    runs, cur = [], None
    for i, b in enumerate(quiet):
        if b and cur is None:
            cur = i
        elif not b and cur is not None:
            if (i - cur) * 512 / sr > 0.15:
                runs.append((cur * 512, i * 512))
            cur = None
    return runs


def measure(y, sr, cm, src, br_runs):
    n = min(len(y), len(src))
    y = y[:n]
    S = np.abs(librosa.stft(y, n_fft=8192, hop_length=2048))
    f = librosa.fft_frequencies(sr=sr, n_fft=8192)
    rms = librosa.feature.rms(S=S, frame_length=8192)[0]
    keep = rms > np.percentile(rms, 70) * 0.4
    Sv = S[:, keep] if keep.sum() > 5 else S
    P = (Sv ** 2).mean(axis=1)
    core = P[(f >= 300) & (f < 3000)].sum() + 1e-20

    def b(lo, hi):
        return float(10 * np.log10(P[(f >= lo) & (f < hi)].sum() / core + 1e-20))

    logP = 10 * np.log10(P + 1e-20)
    k = 12
    pad = np.pad(logP, k, mode="edge")
    loc = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(logP))])
    peaks = int(np.sum((f >= 15500) & (f < 20000) & (logP - loc > 8)))
    roll = float(librosa.feature.spectral_rolloff(S=Sv, sr=sr,
                                                  roll_percent=0.95).mean())
    fl = sr // 2
    nf = len(y) // fl
    r = np.sqrt(np.mean(y[:nf * fl].reshape(nf, fl) ** 2, axis=1) + 1e-14)
    act = r > np.max(r) * 10 ** (-30 / 20)
    dyn = float(np.std(20 * np.log10(r[act]))) if act.sum() > 3 else 0.0
    timbre = float(np.linalg.norm(mfcc_mean(y, sr)[1:] - cm[1:]))

    # breath whistle frames
    n_wh, tot = 0, 0
    band = None
    for s0, e0 in br_runs:
        seg = y[s0:min(e0, n)]
        if len(seg) < 2048:
            continue
        Sb = np.abs(librosa.stft(seg, n_fft=2048, hop_length=512))
        fb = librosa.fft_frequencies(sr=sr, n_fft=2048)
        if band is None:
            band = (fb >= 800) & (fb <= 10000)
        logSb = 20 * np.log10(Sb + 1e-10)
        for fr in range(Sb.shape[1]):
            col = logSb[:, fr]
            tot += 1
            if float(col[band].max() - np.median(col[band])) > 18:
                n_wh += 1
    return dict(peaks=peaks, air12=round(b(12000, 15500), 1),
                junk16=round(b(16000, 20000), 1), roll=round(roll),
                dyn=round(dyn, 2), timbre=round(timbre, 1),
                breath_wh=f"{n_wh}/{tot}")


# ---------------------------------------------------------------- RVC infer
def setup_vc():
    sys.path.insert(0, str(RVC_DIR))
    os.chdir(str(RVC_DIR))
    os.environ["weight_root"] = str(RVC_DIR / "assets/weights")
    os.environ["index_root"] = str(RVC_DIR / "logs")
    os.environ["rmvpe_root"] = str(RVC_DIR / "assets/rmvpe")
    os.environ["weight_uvr5_root"] = str(RVC_DIR / "assets/uvr5_weights")
    from dotenv import load_dotenv
    load_dotenv(override=True)
    from infer.modules.vc.modules import VC
    from configs.config import Config
    import torch
    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    _s = sys.argv.copy()
    sys.argv = ["_ab_eval_v2.py"]
    cfg = Config()
    cfg.device = "cuda:0"
    cfg.is_half = True
    sys.argv = _s
    return VC(cfg)


def epoch_of(step: int, steps_per_epoch: int) -> int:
    return round(step / steps_per_epoch)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cm = corpus_ref()
    src_full, sr = librosa.load(str(SRC_FULL), sr=None, mono=True)
    seg_src, _ = librosa.load(str(SEG), sr=None, mono=True)
    br_runs = breath_regions(src_full, sr)
    st_off = int(0.26 * sr)                       # seg_000 placement

    # steps/epoch from filelist length (selected slices / bs, ceil)
    n_lines = len((RVC_DIR / "logs/honoka_v2/filelist.txt")
                  .read_text().splitlines())
    spe = int(np.ceil(n_lines / 16))
    print(f"filelist {n_lines} -> ~{spe} steps/epoch", flush=True)

    vc = setup_vc()
    results = {}
    for arm, mdir in ARMS.items():
        cks = sorted(mdir.glob("G_*_infer.pth"),
                     key=lambda p: int(p.stem.split("_")[1]))
        rows = []
        for ck in cks:
            step = int(ck.stem.split("_")[1])
            ep = epoch_of(step, spe)
            if ep < 95:                            # e100-200 window (tolerant)
                continue
            wname = f"{arm}_{ck.stem}.pth"
            wdst = RVC_DIR / "assets/weights" / wname
            if not wdst.exists():
                shutil.copy2(ck, wdst)
            try:
                vc.get_vc(wname)
                _, (sro, ao) = vc.vc_single(
                    sid=0, input_audio_path=str(SEG), f0_up_key=0,
                    f0_file=None, f0_method="rmvpe",
                    file_index=str(INDEX), file_index2=str(INDEX),
                    index_rate=IR, filter_radius=7, resample_sr=0,
                    rms_mix_rate=0.25, protect=PROTECT)
            except Exception as e:
                print(f"  FAIL {arm} {ck.stem}: {e}", flush=True)
                continue
            y = ao.astype(np.float64)
            # place into full timeline, RMS-matched to source segment
            nmin = min(len(y), len(seg_src))
            y = y[:nmin]
            tgt = float(np.sqrt(np.mean(seg_src[:nmin] ** 2)))
            y *= tgt / (float(np.sqrt(np.mean(y ** 2))) + 1e-12)
            full = np.zeros(len(src_full))
            m = min(len(y), len(full) - st_off)
            full[st_off:st_off + m] = y[:m]
            met = measure(full.astype(np.float32), sro, cm, src_full, br_runs)
            met.update(ckpt=ck.stem, epoch=ep)
            rows.append(met)
            wav_p = OUT / f"{arm}_{ck.stem}_e{ep}.wav"
            sf.write(str(wav_p), full.astype(np.float32), sr)
            print(f"  {arm} {ck.stem} (e{ep}): peaks {met['peaks']}  "
                  f"air12 {met['air12']}  timbre {met['timbre']}  "
                  f"breath {met['breath_wh']}", flush=True)
        results[arm] = rows

    # source + v1 reference rows
    ref = {}
    ref["source"] = measure(src_full.astype(np.float32), sr, cm, src_full, br_runs)
    if V1_BEST.exists():
        yv1, _ = librosa.load(str(V1_BEST), sr=None, mono=True)
        ref["v1_best"] = measure(yv1.astype(np.float32), sr, cm, src_full, br_runs)

    def rank_key(m):
        # fewer peaks, closer air12 to source, lower timbre
        air_pen = abs(m["air12"] - ref["source"]["air12"])
        return (m["peaks"], round(air_pen, 1), m["timbre"])

    print("\n" + "=" * 74)
    print(f"{'arm':<10}{'ckpt':>9}{'ep':>5}{'peaks':>6}{'air12':>8}"
          f"{'junk16':>8}{'roll':>7}{'dyn':>6}{'timbre':>8}{'breath':>10}")
    for name, m in ref.items():
        print(f"{name:<10}{'—':>9}{'—':>5}{m['peaks']:>6}{m['air12']:>8}"
              f"{m['junk16']:>8}{m['roll']:>7}{m['dyn']:>6}{m['timbre']:>8}"
              f"{m['breath_wh']:>10}")
    best = {}
    for arm, rows in results.items():
        rows.sort(key=rank_key)
        for m in rows:
            print(f"{arm:<10}{m['ckpt']:>9}{m['epoch']:>5}{m['peaks']:>6}"
                  f"{m['air12']:>8}{m['junk16']:>8}{m['roll']:>7}{m['dyn']:>6}"
                  f"{m['timbre']:>8}{m['breath_wh']:>10}")
        if rows:
            best[arm] = rows[0]

    (OUT / "ab_report.json").write_text(json.dumps(
        {"reference": ref, "arms": results,
         "best": {a: m for a, m in best.items()}}, indent=2, default=str))
    print(f"\nbest per arm: "
          + "  ".join(f"{a}={m['ckpt']}(e{m['epoch']})" for a, m in best.items()))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
