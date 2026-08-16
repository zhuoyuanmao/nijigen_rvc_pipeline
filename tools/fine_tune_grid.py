"""Fine per-note tuning for 2:48.8-2:54.7 海♂「時を止め帰りたくないよね、今日は」.

The reference is unusable here (the 6-seiyuu version sings a different part at this
spot: raw deviation +1870 ct, only 11% of frames within 60 ct even after octave
folding), so the basis is the equal-tempered grid — classic autotune, no reference.

Two safeguards against the usual autotune failure modes:
- only notes within +-MAXOFF of a semitone are touched, so "nearest note" is never
  ambiguous (all 11 notes here sit within +-43 ct, so none is a coin flip);
- each note is SHIFTED BY A CONSTANT rather than flattened, and the shift ramps
  smoothly between notes — vibrato, portamento and attack shape survive intact.
STRENGTH < 1 deliberately leaves a few cents of human variation.
"""
import glob, json, numpy as np, soundfile as sf, librosa
import parselmouth
from parselmouth.praat import call

SR, HOP = 44100, 512
RAW = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
TUN = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2"
TARGET = (168.8, 174.7)
FMIN, FMAX = 75, 500
MINDUR, MINOFF, MAXOFF, STRENGTH, RAMP = 0.15, 20.0, 50.0, 0.85, 0.08

raw_p = [f for f in glob.glob(RAW+"/*.wav") if "camus-toya" in f.lower()][0]
tun_p = [f for f in glob.glob(TUN+"/*.wav") if "camus-toya" in f.lower()][0]
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
print(f"乐句 {s_abs/SR:.2f}-{e_abs/SR:.2f}s")

pad = int(0.05*SR); a0 = max(0, s_abs-pad); b0 = min(len(mono), e_abs+pad)
f0, _, _ = librosa.pyin(mono[a0:b0].astype(np.float32), fmin=FMIN, fmax=FMAX, sr=SR,
                        frame_length=2048, hop_length=HOP)
n = len(f0)
corr = np.zeros(n)                       # cents to add, per frame
nm = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
touched = []
i = 0
while i < n:
    if not np.isfinite(f0[i]): i += 1; continue
    j = i; anchor = f0[i]
    while j < n and np.isfinite(f0[j]) and abs(1200*np.log2(f0[j]/anchor)) < 60: j += 1
    dur = (j-i)*HOP/SR
    if dur >= MINDUR:
        med = float(np.nanmedian(f0[i:j]))
        k = 69 + 12*np.log2(med/440.0)
        off = (k - round(k))*100                       # how far from the semitone
        if MINOFF <= abs(off) <= MAXOFF:
            corr[i:j] = -off*STRENGTH
            touched.append(((a0/SR)+i*HOP/SR, dur, f"{nm[int(round(k))%12]}{int(round(k))//12-1}",
                            med, off, -off*STRENGTH))
    i = j
# ramp between notes so nothing steps; keeps transitions natural
w = int(RAMP*SR/HOP) | 1
corr = np.convolve(np.pad(corr, w//2, mode="edge"), np.hanning(w)/np.hanning(w).sum(), mode="valid")[:n]

print(f"\n{'时间':>10}{'时长':>7}{'音':>6}{'Hz':>7}{'原偏差':>9}{'施加':>8}")
print("-"*48)
for t, d, nn, hz, off, c in touched:
    print(f"{t:>9.2f}s{d:>7.2f}{nn:>6}{hz:>7.0f}{off:>+8.0f} ct{c:>+7.0f}")
print(f"\n修正 {len(touched)} 个音 (强度 {STRENGTH:.0%}, 留 {1-STRENGTH:.0%} 自然浮动)")

def psola_curve(x, ct):
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    man = call(snd, "To Manipulation", 0.01, FMIN, FMAX)
    pt = call(man, "Extract pitch tier")
    npts = int(call(pt, "Get number of points"))
    if npts == 0: return x.copy()
    ts = [call(pt, "Get time from index", k+1) for k in range(npts)]
    fs = [call(pt, "Get value at index", k+1) for k in range(npts)]
    grid = np.arange(len(ct))*HOP/SR
    newpt = call("Create PitchTier", "c", snd.xmin, snd.xmax)
    for t, f in zip(ts, fs):
        call(newpt, "Add point", t, f*2.0**(float(np.interp(t, grid, ct))/1200.0))
    call([newpt, man], "Replace pitch tier")
    return np.asarray(call(man, "Get resynthesis (overlap-add)").values).ravel()

y = sf.read(tun_p, always_2d=True)[0].copy()
for c in range(y.shape[1]):
    src = raw[a0:b0, c]
    out = psola_curve(src, corr)
    k = min(len(out), len(src)); out = out[:k]
    wxf = np.ones(k); xf = min(int(0.02*SR), k//2)
    wxf[:xf] = np.linspace(0, 1, xf); wxf[-xf:] = np.linspace(1, 0, xf)
    y[a0:a0+k, c] = y[a0:a0+k, c]*(1-wxf) + out*wxf
sf.write(tun_p, y, sr, subtype=sf.info(tun_p).subtype)

# verify on the result
f1, _, _ = librosa.pyin(y.mean(1)[a0:b0].astype(np.float32), fmin=FMIN, fmax=FMAX, sr=SR,
                        frame_length=2048, hop_length=HOP)
def grid_err(f):
    v = f[np.isfinite(f)]
    k = 69 + 12*np.log2(v/440.0)
    return np.abs((k - np.round(k))*100)
print(f"\n离半音格 |偏差| 中位: {np.median(grid_err(f0)):.0f} → {np.median(grid_err(f1)):.0f} ct")

P = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/pitch_data.json"
d = json.load(open(P, encoding="utf-8"))
for dd in range(-2, 3):
    k = f"umi-male|{168+dd}"
    if k in d:
        d[k]["shift"] = int(round(np.mean([t[5] for t in touched]))) if touched else 0
        d[k]["method"] = "grid"; d[k]["notes"] = len(touched)
        break
json.dump(d, open(P, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"pitch_data: {k} -> method=grid, {len(touched)} 音")
