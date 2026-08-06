"""Pitch correction v2 — time-varying curve instead of constant shift.

Audit of v1 found: (a) constant per-phrase shift leaves residuals when the error
drifts WITHIN a phrase (1:59.5 honoka: +70ct early vs +40 median -> +30 residual);
(b) the ±60ct cap truncated 3:03.4 (-75 -> -20 residual). Fix both:
per-frame deviation curve -> outlier rejection -> 0.6s zero-phase smoothing ->
applied as a CONTINUOUS multiplier on the Praat pitch tier inside ONE PSOLA
resynthesis (no internal boundaries), cap widened to ±80ct.
Phrase SELECTION guards are unchanged (same 17 phrases; duet-reference and
expression-jitter defenses stay).
-> stems in 导出_tuned2/ + before/after verification incl. worst 1.5s sub-window
"""
import glob, os, shutil, numpy as np, librosa, soundfile as sf
import parselmouth
from parselmouth.praat import call

SR, HOP = 44100, 512
EXP = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
OUT = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2"
REF = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals.wav"
VOICES = {"honoka":("honoka",100,700), "kotori":("kotori",100,700), "umi":("umi",100,700),
          "honoka-male":("otoya-tsuka",75,500), "kotori-male":("cecil-ai",75,500),
          "umi-male":("camus-toya",75,500)}
MIN_DEV, JIT, CAP, MAX_REF = 25.0, 200.0, 80.0, 150.0
SMOOTH_S = 0.6
os.makedirs(OUT, exist_ok=True)

def psola_curve(x, corr_t, corr_c, fmin, fmax):
    """One PSOLA resynthesis with a time-varying pitch multiplier.
    corr_t: times (s, in segment frame) ; corr_c: correction in cents at those times."""
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    man = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(man, "Extract pitch tier")
    n = int(call(pt, "Get number of points"))
    if n == 0:
        return x.copy()
    ts = [call(pt, "Get time from index", i+1) for i in range(n)]
    fs = [call(pt, "Get value at index", i+1) for i in range(n)]
    newpt = call("Create PitchTier", "corr", snd.xmin, snd.xmax)
    for t, f in zip(ts, fs):
        c = float(np.interp(t, corr_t, corr_c))
        call(newpt, "Add point", t, f * 2.0**(c/1200.0))
    call([newpt, man], "Replace pitch tier")
    return np.asarray(call(man, "Get resynthesis (overlap-add)").values).ravel()

print("reference F0 ...", flush=True)
ref, _ = librosa.load(REF, sr=SR, mono=True)
rf0, _, _ = librosa.pyin(ref, fmin=90, fmax=1000, sr=SR, frame_length=2048, hop_length=HOP)

def subwin_dev(cf0, s, e, worst=True):
    """median dev per 1.5s sub-window; return worst |median| (frames)."""
    i0, i1 = s//HOP, e//HOP
    W = int(1.5*SR/HOP); devs = []
    for j in range(i0, max(i0+1, i1-W//2), max(1, W//2)):
        a = cf0[j:j+W]; b = rf0[j:min(j+W, len(rf0))]
        n = min(len(a), len(b)); m = np.isfinite(a[:n]) & np.isfinite(b[:n])
        if m.sum() < 15: continue
        d = 1200*np.log2(a[:n][m]/b[:n][m])
        d = d[np.abs(d - np.median(d)) < 300]
        if len(d) > 10: devs.append(float(np.median(d)))
    if not devs: return np.nan
    return devs[int(np.argmax(np.abs(devs)))] if worst else float(np.median(devs))

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
            continue                                  # same selection as v1
        # per-frame deviation curve on the phrase's frame grid
        idx_local = np.arange(n)
        dev_full = np.full(n, np.nan)
        dev_full[m] = 1200*np.log2(a[m]/b[m])
        anchors = np.isfinite(dev_full) & (np.abs(dev_full-med) < 150)   # outlier reject
        if anchors.sum() < 20: continue
        curve = np.interp(idx_local, idx_local[anchors], dev_full[anchors])
        w = int(SMOOTH_S*SR/HOP) | 1
        k = np.hanning(w); k /= k.sum()
        curve = np.convolve(np.pad(curve, w//2, mode="edge"), k, mode="valid")
        corr = np.clip(-curve, -CAP, CAP)             # continuous correction, cents
        pre_worst = subwin_dev(cf0, s, e)
        pad = int(0.05*SR); a0 = max(0, s-pad); b0 = min(len(mono), e+pad)
        corr_t = ((i0 + idx_local)*HOP - a0)/SR       # times in segment frame
        for c in range(y.shape[1]):
            seg = y[a0:b0, c]
            sh = psola_curve(seg, corr_t, corr, fmin, fmax)
            kk = min(len(sh), len(seg)); sh = sh[:kk]
            wxf = np.ones(kk); xf = min(int(0.02*SR), kk//2)
            wxf[:xf] = np.linspace(0, 1, xf); wxf[-xf:] = np.linspace(1, 0, xf)
            out[a0:a0+kk, c] = seg[:kk]*(1-wxf) + sh*wxf
        log.append((s/SR, e/SR, name, med, float(np.mean(corr)), pre_worst, s, e))
    sf.write(os.path.join(OUT, os.path.basename(src)), out, sr, subtype=sf.info(src).subtype)

for f in glob.glob(EXP+"/*.wav"):
    d = os.path.join(OUT, os.path.basename(f))
    if not os.path.exists(d): shutil.copy2(f, d)

print(f"\n=== corrected {len(log)} phrases; verifying (whole-phrase + worst 1.5s window) ===",
      flush=True)
print(f"{'time':>13s} {'voice':12s} {'med был':>8s} {'worst-win был':>13s} {'-> med':>7s} {'worst-win':>10s}")
for t0, t1, name, med, mcorr, prew, s, e in sorted(log):
    key = VOICES[name][0]
    tuned = [f for f in glob.glob(OUT+"/*.wav") if key in f.lower()][0]
    yt, _ = sf.read(tuned, always_2d=True)
    tf0, _, _ = librosa.pyin(yt.mean(1)[max(0,s-HOP*4):e+HOP*4].astype(np.float32),
                             fmin=90, fmax=1000, sr=SR, frame_length=2048, hop_length=HOP)
    off = max(0, s-HOP*4)//HOP
    full = np.full(len(rf0), np.nan); full[off:off+len(tf0)] = tf0
    i0, i1 = s//HOP, e//HOP
    a = full[i0:i1]; b = rf0[i0:i1]
    n = min(len(a), len(b)); m = np.isfinite(a[:n]) & np.isfinite(b[:n])
    med_after = float(np.median(1200*np.log2(a[:n][m]/b[:n][m]))) if m.sum() > 20 else np.nan
    postw = subwin_dev(full, s, e)
    print(f"{int(t0//60)}:{t0%60:04.1f}-{int(t1//60)}:{t1%60:04.1f} {name:12s} "
          f"{med:+7.0f} {prew:+12.0f} {med_after:+6.0f} {postw:+9.0f}")
print(f"\n-> {OUT}")
