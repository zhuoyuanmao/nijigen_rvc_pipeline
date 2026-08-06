"""Reference-guided pitch correction — surgical, not global.
Measured first: the covers are already in tune (median |dev| 20 ct, net bias +1 ct),
so a full-contour correction would pay ~1 dB of PSOLA damage on all 62 phrases to
fix a problem that mostly isn't there. Instead correct ONLY phrases with a stable,
clearly-audible offset, and leave everything else bit-identical.

Guards:
  * reference is a DUET mix -> if |raw dev| > 150 ct the reference holds the other
    voice/harmony there; skip (correcting would drag us onto the wrong note).
  * jitter (spread of the deviation) > JIT means the deviation wanders = expression
    (portamento/vibrato), not a tuning error; skip.
  * correction capped at CAP cents; PSOLA (Praat) chosen for the shift
    (~1 dB round-trip vs 3.2 dB for a phase vocoder).
-> corrected stems in 导出_tuned/ + a log of every change
"""
import glob, os, numpy as np, librosa, soundfile as sf
import parselmouth
from parselmouth.praat import call

SR, HOP = 44100, 512
EXP = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
OUT = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned"
REF = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals.wav"
VOICES = {"honoka":("honoka",100,700), "kotori":("kotori",100,700), "umi":("umi",100,700),
          "honoka-male":("otoya-tsuka",75,500), "kotori-male":("cecil-ai",75,500),
          "umi-male":("camus-toya",75,500)}
MIN_DEV, JIT, CAP, MAX_REF = 25.0, 200.0, 60.0, 150.0
os.makedirs(OUT, exist_ok=True)

def psola(x, cents, fmin, fmax):
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    man = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(man, "Extract pitch tier")
    call(pt, "Multiply frequencies", snd.xmin, snd.xmax, 2.0**(cents/1200.0))
    call([pt, man], "Replace pitch tier")
    return np.asarray(call(man, "Get resynthesis (overlap-add)").values).ravel()

print("reference F0 ...", flush=True)
ref, _ = librosa.load(REF, sr=SR, mono=True)
rf0, _, _ = librosa.pyin(ref, fmin=90, fmax=1000, sr=SR, frame_length=2048, hop_length=HOP)

log = []
for name, (key, fmin, fmax) in VOICES.items():
    src = [f for f in glob.glob(EXP+"/*.wav") if key in f.lower()][0]
    y, sr = sf.read(src, always_2d=True)
    mono = y.mean(1)
    print(f"  {name} ...", flush=True)
    cf0, _, _ = librosa.pyin(mono.astype(np.float32), fmin=90, fmax=1000, sr=SR,
                             frame_length=2048, hop_length=HOP)
    iv = librosa.effects.split(mono, top_db=40, frame_length=2048, hop_length=256)
    merged = []
    for s, e in iv:
        if merged and s-merged[-1][1] < 0.5*SR: merged[-1][1] = e
        else: merged.append([s, e])
    out = y.copy()
    for s, e in merged:
        if e-s < 0.8*SR: continue
        i0, i1 = s//HOP, e//HOP
        a = cf0[i0:i1]; b = rf0[i0:min(i1, len(rf0))]
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30: continue
        dev = 1200*np.log2(a[m]/b[m]); med = float(np.median(dev))
        jit = float(np.percentile(np.abs(dev-med), 75))
        if abs(med) > MAX_REF or abs(med) < MIN_DEV or jit > JIT:
            continue
        corr = float(np.clip(-med, -CAP, CAP))
        pad = int(0.05*SR); a0 = max(0, s-pad); b0 = min(len(mono), e+pad)
        for c in range(y.shape[1]):
            seg = y[a0:b0, c]
            sh = psola(seg, corr, fmin, fmax)
            k = min(len(sh), len(seg)); sh = sh[:k]
            w = np.ones(k); xf = min(int(0.02*SR), k//2)
            w[:xf] = np.linspace(0, 1, xf); w[-xf:] = np.linspace(1, 0, xf)
            out[a0:a0+k, c] = seg[:k]*(1-w) + sh*w
        log.append((s/SR, e/SR, name, med, corr, jit))
    sf.write(os.path.join(OUT, os.path.basename(src)), out, sr,
             subtype=sf.info(src).subtype)

# untouched stems (BGM) copied through so the folder is a drop-in replacement
import shutil
for f in glob.glob(EXP+"/*.wav"):
    d = os.path.join(OUT, os.path.basename(f))
    if not os.path.exists(d): shutil.copy2(f, d)

import json
json.dump([{"t0": t0, "t1": t1, "voice": v, "dev_ct": med, "shift_ct": corr, "jitter": jit}
           for t0, t1, v, med, corr, jit in log],
          open("/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/pitch_corrections.json", "w"),
          ensure_ascii=False, indent=1)
print(f"\n=== corrected {len(log)} phrases (of 62); all others bit-identical ===")
print(f"{'time':>13s} {'voice':12s} {'was off':>9s} {'shift':>8s} {'jitter':>7s}")
for t0, t1, v, med, corr, jit in sorted(log):
    print(f"{int(t0//60)}:{t0%60:04.1f}-{int(t1//60)}:{t1%60:04.1f} {v:12s} "
          f"{med:+6.0f} ct {corr:+6.0f} ct {jit:6.0f}")
print(f"\n-> {OUT}")
