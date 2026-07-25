"""v4 preprocessing for the tokyo_summer male dry stem (inference source).

The current v3 inference source (stage1_40k/source_40k.wav) is a plain
40k-mono conversion of Desktop/AI翻唱/夏日.wav — it never went through the
v4 cleanup chain that the honoka source got. This applies the same chain,
starting from the original 44.1k file for maximum headroom:

    dereverb (anvuew less-aggressive Roformer)
      -> 3x Roformer vocals models, per-sample min-|x| intersection
      -> NO trim (unlike honoka's _prep_v4_nodemucs.py: this is a full-song
         stem and head-trimming would break BGM alignment at mix time)
      -> peak-safety only, then a 40k-mono copy for RVC

    python _prep_source_v4.py \
        --input "/mnt/c/Users/kevin/Desktop/AI翻唱/夏日.wav" \
        --outdir output/tokyo_summer_v4src
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

DEREVERB_MODEL = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"
DEREVERB_STEM = "noreverb"
VOCALS_MODELS = [
    ("mel_band_roformer_kim_ft_unwa.ckpt",         "vocals", "kim_ft"),
    ("melband_roformer_big_beta5e.ckpt",           "vocals", "big_beta5e"),
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt",  "vocals", "bs_viperx_1297"),
]
MODELS_CACHE = Path.home() / ".cache" / "audio-separator-models"


def separator_pass(model_file: str, stem_name: str,
                   input_wav: Path, out_wav: Path) -> Path:
    if out_wav.exists() and out_wav.stat().st_size > 0:
        print(f"[skip] {out_wav.name}", flush=True)
        return out_wav
    from audio_separator.separator import Separator
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=30, model_file_dir=str(MODELS_CACHE),
                    output_dir=str(out_wav.parent), output_format="wav",
                    output_single_stem=stem_name)
    print(f"loading {model_file} ...", flush=True)
    sep.load_model(model_filename=model_file)
    t0 = time.time()
    produced = sep.separate(str(input_wav),
                            custom_output_names={stem_name: out_wav.stem})
    if not produced:
        raise RuntimeError(f"{model_file} produced no output")
    raw = Path(produced[0])
    if not raw.is_absolute():
        raw = out_wav.parent / raw.name
    if raw.resolve() != out_wav.resolve() and raw.exists():
        raw.replace(out_wav)
    print(f"  -> {out_wav.name}   ({time.time()-t0:.1f}s)", flush=True)
    return out_wav


def min_intersection(files: list[Path], out_wav: Path) -> Path:
    """Per-sample sign-preserving min-|x| — the v4 conservative intersection.
    Lead frames survive (every model agrees they are vocals); anything that
    even one model rejects is pulled toward zero."""
    if out_wav.exists() and out_wav.stat().st_size > 0:
        print(f"[skip] {out_wav.name}", flush=True)
        return out_wav
    arrays, srs = [], []
    for p in files:
        a, sr = sf.read(str(p), always_2d=True)
        arrays.append(a.astype(np.float32))
        srs.append(sr)
    assert len(set(srs)) == 1, f"sr mismatch: {srs}"
    n = min(a.shape[0] for a in arrays)
    ch = max(a.shape[1] for a in arrays)
    fixed = []
    for a in arrays:
        a = a[:n]
        if a.shape[1] == 1 and ch > 1:
            a = np.tile(a, (1, ch))
        fixed.append(a)
    stack = np.stack(fixed, axis=0)
    idx = np.argmin(np.abs(stack), axis=0)
    row = np.arange(idx.shape[0])[:, None]
    col = np.arange(idx.shape[1])[None, :]
    winning = stack[idx, row, col]
    sf.write(str(out_wav), winning, srs[0], subtype="FLOAT")
    print(f"  -> {out_wav.name} (min-|x| across {len(files)} models)", flush=True)
    return out_wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    interm = outdir / "_intermediates"
    interm.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("\n=== Step 1: dereverb (anvuew less-aggressive) ===", flush=True)
    dereverb = separator_pass(DEREVERB_MODEL, DEREVERB_STEM,
                              args.input, interm / "vocals_dereverb.wav")

    print("\n=== Step 2: 3x Roformer vocals ===", flush=True)
    outs = [separator_pass(mf, stem, dereverb, interm / f"vocals_{tag}.wav")
            for mf, stem, tag in VOCALS_MODELS]

    print("\n=== Step 3: min-|x| intersection ===", flush=True)
    clean44 = min_intersection(outs, outdir / "source_clean_44k.wav")

    print("\n=== Step 4: 40k mono for RVC (no trim — alignment preserved) ===",
          flush=True)
    y, sr = librosa.load(str(clean44), sr=None, mono=True)
    if sr != 40000:
        y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=40000)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y / peak * 0.99
    out40 = outdir / "source_clean_40k.wav"
    sf.write(str(out40), y.astype(np.float32), 40000)

    print(f"\n=== DONE in {time.time()-t_all:.0f}s ===")
    print(f"44.1k clean : {clean44}")
    print(f"RVC input   : {out40}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
