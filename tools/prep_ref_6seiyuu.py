"""Build the new pitch-correction reference from the 6-seiyuu video.

1. 3x Roformer vocal separation + the house gated blend (loud frames min-|x|,
   quiet frames median — METHODOLOGY §3) on the extracted audio.
2. Align to the PROJECT timeline by envelope xcorr against the old reference
   (tokyo_summer_vocals.wav, which the AU exports are aligned to).
3. Safety checks before it may be used as a pitch reference:
   - drift across 4 windows (a re-encoded upload may be speed-shifted; speed
     shifts pitch, which would poison every correction value)
   - key check: median F0 on two solo passages vs the old reference
-> Desktop/AI翻唱/tokyo_summer_vocals_6seiyuu.wav  (+ aligned full mix)
"""
import glob, numpy as np, soundfile as sf, librosa
from scipy.signal import correlate
from pathlib import Path

SR, HOP = 44100, 128
TMP = "/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/scratchpad/sixtmp"
OLD_REF = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals.wav"
OUT_V = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals_6seiyuu.wav"
OUT_M = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_6seiyuu_mix.wav"

print("=== 1. gated blend of 3 separations ===", flush=True)
paths = sorted(glob.glob(f"{TMP}/six_v*.wav")) or \
        sorted(glob.glob(f"{TMP}/*ocals*.wav"))          # separator default naming fallback
assert len(paths) == 3, f"expected 3 separated files, got {paths}"
print("  inputs:", [Path(p).name for p in paths], flush=True)
vs = [sf.read(p, always_2d=True)[0] for p in paths]
n = min(v.shape[0] for v in vs)
vs = np.stack([v[:n] for v in vs])                      # (3, n, ch)
med = np.median(vs, axis=0)
mn = vs[np.argmin(np.abs(vs), axis=0), np.arange(n)[:, None], np.arange(vs.shape[2])[None, :]]
frame = 512
rms = np.sqrt(np.mean(med.mean(1)[:n//frame*frame].reshape(-1, frame)**2, axis=1))
db = 20*np.log10(rms + 1e-9)
gate = 1/(1 + np.exp(-(db - (-35.0))/4.0))              # 1 = loud -> min, 0 = quiet -> median
g = np.repeat(gate, frame)
g = np.pad(g, (0, n - len(g)), mode="edge")[:, None]
voc = (g*mn + (1-g)*med).astype(np.float32)
print(f"  blended {n/SR:.2f}s, gate loud {float((gate>0.5).mean()):.0%}", flush=True)

print("=== 2. align to project timeline (transport judged on the BGM, not on singers) ===", flush=True)
old, _ = librosa.load(OLD_REF, sr=SR, mono=True)
orig, _ = librosa.load("/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer.wav", sr=SR, mono=True)
newm = voc.mean(1)
newmix = sf.read(f"{TMP}/six_mix.wav", always_2d=True)[0].mean(1).astype(np.float32)[:n]
def env(x): return np.abs(x[:len(x)//HOP*HOP].reshape(-1, HOP)).max(1)
ea, eb = env(newmix), env(orig)                         # full mixes: BGM dominates
c = correlate(ea - ea.mean(), eb - eb.mean(), mode="full", method="fft")
s = int(np.argmax(c) - (len(eb) - 1))                   # new[t+s] ~ old[t], frames
print(f"  global offset (mix vs original): new leads by {s*HOP/SR:+.3f}s", flush=True)

def local_lag(ea_, eb_, t0, t1):
    i0, i1 = int(t0*SR/HOP), int(t1*SR/HOP)
    b = eb_[i0:i1]
    a = ea_[max(0, i0+s-200):i1+s+200]
    if len(a) < len(b) + 50: return np.nan
    cc = correlate(a - a.mean(), b - b.mean(), mode="valid")
    return (int(np.argmax(cc)) - (i0+s - max(0, i0+s-200))) * HOP/SR * 1000
wins = [(0.5, 20), (60, 90), (91, 101), (120, 150), (200, 227)]
lags = [local_lag(ea, eb, *w) for w in wins]
print("  BGM drift (ms):", [f"{l:+.0f}" for l in lags], flush=True)
drift = float(np.nanmax(lags) - np.nanmin(lags))
span = (wins[-1][0]+wins[-1][1])/2 - (wins[0][0]+wins[0][1])/2
cents = 1200*np.log2(1 + (drift/1000)/span)
if drift > 15:
    print(f"  !! BGM DRIFT {drift:.0f}ms (~{cents:.1f} ct speed shift) — resample needed"); raise SystemExit(2)
# singers' own timing vs the old singers — info only, does not gate
ev_a, ev_b = env(newm), env(old)
vlags = [local_lag(ev_a, ev_b, *w) for w in [(20, 50), (80, 110), (140, 170), (195, 225)]]
print("  (info) singer-timing offset vs old version (ms):", [f"{l:+.0f}" for l in vlags], flush=True)

print("=== 3. key check (solo passages, old timeline) ===", flush=True)
def med_f0(x, t0, t1):
    f, _, _ = librosa.pyin(x[int(t0*SR):int(t1*SR)], fmin=90, fmax=1000, sr=SR,
                           frame_length=2048, hop_length=512)
    v = f[np.isfinite(f)]
    return float(np.median(v)) if len(v) > 30 else np.nan
off = s*HOP
newm_al = newm[max(0, off):]
for (t0, t1, lab) in [(66, 78, "男 solo 1:06"), (78, 90, "女 solo 1:18")]:
    fo, fn = med_f0(old, t0, t1), med_f0(newm_al, t0, t1)
    ct = 1200*np.log2(fn/fo)
    fold = ct - round(ct/1200)*1200                     # different singer may sit an octave off
    print(f"  {lab}: old {fo:.1f}Hz | new {fn:.1f}Hz | folded diff {fold:+.0f} ct", flush=True)
    if abs(fold) > 60:
        print("  !! key mismatch — investigate before using"); raise SystemExit(3)

print("=== 4. write aligned files ===", flush=True)
L = len(old)
def cut(x2):
    o = x2[max(0, off):]
    if off < 0: o = np.concatenate([np.zeros((-off, x2.shape[1]), np.float32), o])
    if len(o) < L: o = np.concatenate([o, np.zeros((L-len(o), x2.shape[1]), np.float32)])
    return o[:L]
sf.write(OUT_V, cut(voc), SR, subtype="PCM_24")
mix = sf.read(f"{TMP}/six_mix.wav", always_2d=True)[0].astype(np.float32)[:n]
sf.write(OUT_M, cut(mix), SR, subtype="PCM_24")
print(f"  -> {OUT_V}\n  -> {OUT_M}\n  ({L/SR:.2f}s, project timeline)  ALL CHECKS PASSED")
