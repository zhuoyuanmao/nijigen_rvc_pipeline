#!/usr/bin/env python3
"""V4-like preprocessing for pure vocal input (no demucs needed).
Chain: dereverb → 3×Roformer conservative min-|x| → trim + LUFS -20.
"""
import argparse, subprocess, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln

DEREVERB_MODEL = "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt"
DEREVERB_STEM = "noreverb"
VOCALS_MODELS = [
    ("mel_band_roformer_kim_ft_unwa.ckpt",         "vocals", "kim_ft"),
    ("melband_roformer_big_beta5e.ckpt",           "vocals", "big_beta5e"),
    ("model_bs_roformer_ep_317_sdr_12.9755.ckpt",  "vocals", "bs_viperx_1297"),
]
MODELS_CACHE = Path.home() / ".cache" / "audio-separator-models"


def run(cmd, **kw):
    print(f"$ {' '.join(str(x) for x in cmd)}", flush=True)
    subprocess.run([str(x) for x in cmd], check=True, **kw)


def separator_pass(model_file, stem_name, input_wav, out_wav):
    if out_wav.exists() and out_wav.stat().st_size > 0:
        print(f"[skip] {out_wav.name}")
        return out_wav
    from audio_separator.separator import Separator
    MODELS_CACHE.mkdir(parents=True, exist_ok=True)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sep = Separator(log_level=30, model_file_dir=str(MODELS_CACHE),
                    output_dir=str(out_wav.parent), output_format="wav",
                    output_single_stem=stem_name)
    print(f"loading {model_file} ...", flush=True)
    sep.load_model(model_filename=model_file)
    t0 = time.time()
    produced = sep.separate(str(input_wav), custom_output_names={stem_name: out_wav.stem})
    dt = time.time() - t0
    if not produced:
        raise RuntimeError(f"{model_file} produced no output")
    raw = Path(produced[0])
    if not raw.is_absolute():
        raw = out_wav.parent / raw.name
    if raw.resolve() != out_wav.resolve() and raw.exists():
        raw.replace(out_wav)
    print(f"  -> {out_wav.name}   ({dt:.1f}s)", flush=True)
    return out_wav


def min_intersection(files, out_wav):
    if out_wav.exists() and out_wav.stat().st_size > 0:
        print(f"[skip] {out_wav.name}")
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
    absv = np.abs(stack)
    idx = np.argmin(absv, axis=0)
    N, C = idx.shape
    row = np.arange(N)[:, None]
    col = np.arange(C)[None, :]
    winning = stack[idx, row, col]
    sf.write(str(out_wav), winning, srs[0], subtype="PCM_16")
    print(f"  -> {out_wav.name} (min-|x| across {len(files)} models)", flush=True)
    return out_wav


def trim_and_norm(src, dst, target_lufs=-20.0, silence_db=-40.0, keep_s=0.2):
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[skip] {dst.name}")
        return
    audio, sr = sf.read(str(src), always_2d=False)
    mono = audio.mean(axis=1).astype(np.float32) if audio.ndim > 1 else audio.astype(np.float32)
    trimmed, (start, end) = librosa.effects.trim(mono, top_db=-silence_db, frame_length=2048, hop_length=512)
    keep = int(keep_s * sr)
    s = max(0, start - keep)
    e = min(len(audio), end + keep)
    out = audio[s:e] if audio.ndim > 1 else audio[s:e]
    meter = pyln.Meter(sr)
    m = out.mean(axis=1) if out.ndim > 1 else out
    lufs_in = meter.integrated_loudness(m)
    if np.isfinite(lufs_in):
        out = pyln.normalize.loudness(out, lufs_in, target_lufs)
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    ceiling = 10 ** (-1 / 20)
    if peak > ceiling:
        out = out * (ceiling / peak)
    sf.write(str(dst), out, sr, subtype="PCM_16")
    print(f"  -> {dst.name} (LUFS {target_lufs})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    outdir = args.outdir.resolve()
    interm = outdir / "_intermediates"
    outdir.mkdir(parents=True, exist_ok=True)
    interm.mkdir(parents=True, exist_ok=True)

    t_all = time.time()

    # Step 1: dereverb
    print("\n=== Step 1: dereverb (anvuew less-aggressive) ===")
    dereverb = interm / "vocals_dereverb.wav"
    separator_pass(DEREVERB_MODEL, DEREVERB_STEM, args.input, dereverb)

    # Step 2: 3×Roformer → min-|x| intersection
    print("\n=== Step 2: multi-model vocals separation ===")
    model_outputs = []
    for model_file, stem_name, tag in VOCALS_MODELS:
        out = interm / f"vocals_{tag}.wav"
        separator_pass(model_file, stem_name, dereverb, out)
        model_outputs.append(out)

    print("\n=== Step 3: conservative intersection (min-|x|) ===")
    intersected = interm / "vocals_intersected.wav"
    min_intersection(model_outputs, intersected)

    print("\n=== Step 4: trim + LUFS -20 ===")
    final = outdir / "song_clean_lead.wav"
    trim_and_norm(intersected, final)

    print(f"\n=== ALL DONE in {time.time()-t_all:.0f}s ===")
    print(f"Output: {final}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
