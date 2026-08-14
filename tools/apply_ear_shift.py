"""Lock 0:15.0 鸟♂ 合 Uh____ at -60 ct (user A/B choice).

Why: he holds B2 (126.2 Hz) in the parallel 合 Ah at 0:09 — which sounds right —
but drifted to C3 (128.4 Hz) here, a semitone up, while 鸟♀ stayed on D#4. -60 ct
puts this Uh back on 124.0 Hz ≈ the same B2 as his own Ah. The pitch-vs-original
reference was unusable for this phrase (marked "n/a 低1八度"), so this is a pure
ear decision on the internal Ah/Uh relationship.

Rebuilt from the RAW export, single PSOLA pass, spliced into the v13 stem."""
import glob, os, numpy as np, soundfile as sf, librosa, json
import parselmouth
from parselmouth.praat import call

SR = 44100
RAW = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
TUN = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2"
SHIFT, FMIN, FMAX = -60, 75, 500
TARGET = (15.0, 17.3)

raw_p = [f for f in glob.glob(RAW+"/*.wav") if "cecil-ai" in f.lower()][0]
tun_p = [f for f in glob.glob(TUN+"/*.wav") if "cecil-ai" in f.lower()][0]
raw, sr = sf.read(raw_p, always_2d=True)
mono = raw.mean(1)
iv = librosa.effects.split(mono, top_db=40, frame_length=2048, hop_length=256)
merged = []
for s, e in iv:
    if merged and s-merged[-1][1] < 0.5*SR: merged[-1][1] = e
    else: merged.append([s, e])
segs = [(s, e) for s, e in merged if e-s >= 0.8*SR]
t0, t1 = int(TARGET[0]*SR), int(TARGET[1]*SR)
s_abs, e_abs = max(segs, key=lambda se: max(0, min(se[1], t1)-max(se[0], t0)))

def psola(x, cents):
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    man = call(snd, "To Manipulation", 0.01, FMIN, FMAX)
    pt = call(man, "Extract pitch tier")
    call(pt, "Multiply frequencies", snd.xmin, snd.xmax, 2.0**(cents/1200.0))
    call([pt, man], "Replace pitch tier")
    return np.asarray(call(man, "Get resynthesis (overlap-add)").values).ravel()

y = sf.read(tun_p, always_2d=True)[0].copy()
pad = int(0.05*SR); a0 = max(0, s_abs-pad); b0 = min(len(mono), e_abs+pad)
for c in range(y.shape[1]):
    src = raw[a0:b0, c]
    out = psola(src, SHIFT)
    k = min(len(out), len(src)); out = out[:k]
    w = np.ones(k); xf = min(int(0.02*SR), k//2)
    w[:xf] = np.linspace(0, 1, xf); w[-xf:] = np.linspace(1, 0, xf)
    y[a0:a0+k, c] = y[a0:a0+k, c]*(1-w) + out*w
sf.write(tun_p, y, sr, subtype=sf.info(tun_p).subtype)
print(f"phrase {s_abs/SR:.2f}-{e_abs/SR:.2f}s locked at {SHIFT:+d} ct")

# verify: does his Uh now match his own Ah?
def med_f0(path, a, b, from_arr=None):
    yy = from_arr if from_arr is not None else librosa.load(path, sr=SR, mono=True)[0]
    f, _, _ = librosa.pyin(yy[int(a*SR):int(b*SR)].astype(np.float32),
                           fmin=70, fmax=1000, sr=SR, frame_length=2048, hop_length=512)
    v = f[np.isfinite(f)]
    return float(np.median(v)) if len(v) else 0.0
ah = med_f0(tun_p, 9.3, 11.0)
uh = med_f0(None, 15.2, 17.2, from_arr=y.mean(1))
print(f"  鸟♂ Ah {ah:.1f}Hz | Uh {uh:.1f}Hz -> gap {1200*np.log2(uh/ah):+.0f} ct (was +30)")

P = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/pitch_data.json"
d = json.load(open(P, encoding="utf-8"))
for dd in range(-2, 3):
    k = f"kotori-male|{15+dd}"
    if k in d:
        d[k]["shift"] = SHIFT; d[k]["method"] = "ear"; break
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"pitch_data: {k} -> shift {SHIFT:+d} (method=ear)")
