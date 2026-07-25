"""Recover the vocal snippets that the segmenter silently discarded.

_infer_single.py's split drops any merged interval shorter than
MIN_SEGMENT_S=2.0s. On tokyo_summer_v3 that lost ~1.8s of real vocals
(around 118.9-119.5s and 125.6-126.8s — measured by the coverage analysis:
2.4% of active source frames had no ensemble output at all).

This re-derives the split, finds every dropped interval, and infers it with
0.35s of surrounding context (RVC's F0 tracking is unreliable on sub-second
clips; the pad is trimmed back off afterwards). Transpose is chosen per
snippet from corpus_f0_stats.json — same rule as tools/verify_source_f0.py.

Writes stage3_cache_short/short_XXX.wav + manifest.json for _rebuild_v4.py.

    python _recover_short_segs.py --ckpt G_18900
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--song", default="tokyo_summer_v3")
ap.add_argument("--ckpt", default="G_18900",
                help="ckpt to convert the snippets with (regional winner)")
ap.add_argument("--index", default="models/flat_full_src_feat.index")
ap.add_argument("--index-rate", type=float, default=0.75)
ap.add_argument("--min-keep-s", type=float, default=0.25,
                help="ignore dropped intervals shorter than this (noise)")
CLI = ap.parse_args()

import numpy as np
import soundfile as sf
import librosa
import torch

PROJECT = Path("/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore")
RVC_DIR = PROJECT / "Retrieval-based-Voice-Conversion-WebUI"
CHAR = PROJECT / "characters/otoya_sho_mix"
BASE = CHAR / "output" / CLI.song
SRC = BASE / "stage1_40k/source_40k.wav"
OUT = BASE / "stage3_cache_short"
OUT.mkdir(parents=True, exist_ok=True)
STATS = CHAR / "models/corpus_f0_stats.json"

PAD_S = 0.35
MIN_SEGMENT_S = 2.0

full, sr = librosa.load(str(SRC), sr=None, mono=True)

# --- re-derive the split, same params/quirk as _infer_single.py ------------
intervals = librosa.effects.split(full, top_db=35,
                                  frame_length=2048, hop_length=512)
merged = []
min_gap = int(3.0 * sr)
for i, (s, e) in enumerate(intervals):
    if i > 0 and s - intervals[i - 1][1] < min_gap:
        merged[-1] = (merged[-1][0], e)
    else:
        merged.append((s, e))

dropped = [(s, e) for s, e in merged
           if CLI.min_keep_s <= (e - s) / sr < MIN_SEGMENT_S]
print(f"{len(dropped)} dropped interval(s):")
for s, e in dropped:
    print(f"  {s/sr:7.2f}s - {e/sr:7.2f}s  ({(e-s)/sr:.2f}s)")
if not dropped:
    print("nothing to recover")
    sys.exit(0)

st = json.loads(STATS.read_text())


def pick_key(y: np.ndarray) -> int:
    f0, _, _ = librosa.pyin(y.astype(np.float32), fmin=65.0, fmax=1000.0,
                            sr=sr, frame_length=2048, hop_length=1024)
    v = f0[np.isfinite(f0)]
    v = v[(v > 66) & (v < 990)]
    if v.size < 10:
        return 0                                    # unvoiced — leave alone
    med = float(np.median(v))
    return min((0, 12, -12),
               key=lambda k: abs(np.log2(med * 2 ** (k / 12) / st["median"])))


# --- RVC setup -------------------------------------------------------------
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

_saved = sys.argv.copy()
sys.argv = ["_recover_short_segs.py"]
config = Config()
config.device = "cuda:0"
config.is_half = True
sys.argv = _saved
vc = VC(config)

wname = f"otoya_sho_{CLI.ckpt}.pth"
wdst = RVC_DIR / "assets/weights" / wname
if not wdst.exists():
    shutil.copy2(CHAR / f"models/{CLI.ckpt}_infer.pth", wdst)
vc.get_vc(wname)
index_path = CHAR / CLI.index

manifest = []
pad = int(PAD_S * sr)
for i, (s, e) in enumerate(dropped):
    ps, pe = max(0, s - pad), min(len(full), e + pad)
    snippet = full[ps:pe]
    key = pick_key(full[s:e])
    tmp = OUT / f"_in_{i:03d}.wav"
    sf.write(str(tmp), snippet, sr)
    try:
        _, (sro, ao) = vc.vc_single(
            sid=0, input_audio_path=str(tmp), f0_up_key=key, f0_file=None,
            f0_method="rmvpe", file_index=str(index_path),
            file_index2=str(index_path), index_rate=CLI.index_rate,
            filter_radius=7, resample_sr=0, rms_mix_rate=0.25, protect=0.33)
    except Exception as ex:
        print(f"  FAIL {i}: {ex}")
        continue
    y = ao.astype(np.float64)
    # trim the context pad back off (output sr may differ from source sr)
    a = int((s - ps) / sr * sro)
    b = a + int((e - s) / sr * sro)
    y = y[a:min(b, len(y))]
    outp = OUT / f"short_{i:03d}.wav"
    sf.write(str(outp), y.astype(np.float32), sro)
    manifest.append(dict(idx=i, start_sample=int(s), end_sample=int(e),
                         start_s=round(s / sr, 2), dur_s=round((e - s) / sr, 2),
                         key=key, ckpt=CLI.ckpt, sr=sro, path=outp.name))
    print(f"  short_{i:03d}: {s/sr:.2f}s +{(e-s)/sr:.2f}s  key{key:+d}  ok")
    tmp.unlink(missing_ok=True)

(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\n{len(manifest)} snippet(s) -> {OUT}")
