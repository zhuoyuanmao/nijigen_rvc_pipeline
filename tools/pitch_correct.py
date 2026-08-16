"""Pitch correction v3 = v2 (time-varying curve, same guards) with two changes:

1. REFERENCE is now the 6-seiyuu version (casting matches our cast per line, so
   far fewer phrases have "the other voice" in the reference).
2. PROTECTED list: phrases the user ear-tested / A/B'd are excluded from the
   automatic pass entirely — their locked treatments are re-applied afterwards
   by the dedicated scripts (apply_kirei_p60 / apply_uh_m60 / fix_dialogue_209).
   0:55 stays uncorrected (ear-vetoed).
"""
import glob, os, shutil, numpy as np, librosa, soundfile as sf
import parselmouth
from parselmouth.praat import call

SR, HOP = 44100, 512
EXP = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
OUT = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出_tuned2"
REF = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals_6seiyuu.wav"
VOICES = {"honoka":("honoka",100,700), "kotori":("kotori",100,700), "umi":("umi",100,700),
          "honoka-male":("otoya-tsuka",75,500), "kotori-male":("cecil-ai",75,500),
          "umi-male":("camus-toya",75,500)}
MIN_DEV, JIT, CAP, MAX_REF = 25.0, 200.0, 80.0, 150.0
SMOOTH_S = 0.6
# user-locked phrases (ear/AB): the auto pass must not touch these regions.
# CRUCIALLY this includes the *partner* line of each ear decision: the A/B judged an
# INTERVAL between two voices, so the untested half is frozen too — correcting it
# would silently re-break the relationship the user approved.
PROTECT = {"kotori-male": [(14.5, 17.9)],            # 合 Uh — ear -60 (applied after)
           "kotori":      [(13.5, 17.5)],            # 合 Uh 女声半边 — 耳测音程的另一半, 冻结
           "umi":         [(150.5, 154.5)],          # 「綺麗だね」女声半边 — 同上
           "umi-male":    [(55.0, 58.0),             # 「これ飲めば？」 — ear-vetoed, stays raw
                           (129.0, 134.0),           # 「手作りクッキー？」 — A/B-approved old curve
                           (150.3, 152.4),           # 「綺麗だね」 — ear +60 (applied after)
                           (168.3, 175.2)]}          # 時を止め… — 参照在此为异声部, 走等律逐音修正
os.makedirs(OUT, exist_ok=True)

def psola_curve(x, corr_t, corr_c, fmin, fmax):
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
        if any(s/SR < b and e/SR > a for a, b in PROTECT.get(name, [])):
            print(f"    protected, skipped: {s/SR:.1f}-{e/SR:.1f}s", flush=True)
            continue
        i0, i1 = s//HOP, e//HOP
        a = cf0[i0:i1]; b = rf0[i0:min(i1, len(rf0))]
        n = min(len(a), len(b)); a, b = a[:n], b[:n]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 30: continue
        dev = 1200*np.log2(a[m]/b[m]); med = float(np.median(dev))
        jit = float(np.percentile(np.abs(dev-med), 75))
        if abs(med) > MAX_REF or abs(med) < MIN_DEV or jit > JIT:
            continue
        idx_local = np.arange(n)
        dev_full = np.full(n, np.nan)
        dev_full[m] = 1200*np.log2(a[m]/b[m])
        anchors = np.isfinite(dev_full) & (np.abs(dev_full-med) < 150)
        if anchors.sum() < 20: continue
        curve = np.interp(idx_local, idx_local[anchors], dev_full[anchors])
        w = int(SMOOTH_S*SR/HOP) | 1
        k = np.hanning(w); k /= k.sum()
        curve = np.convolve(np.pad(curve, w//2, mode="edge"), k, mode="valid")
        corr = np.clip(-curve, -CAP, CAP)
        pre_worst = subwin_dev(cf0, s, e)
        pad = int(0.05*SR); a0 = max(0, s-pad); b0 = min(len(mono), e+pad)
        corr_t = ((i0 + idx_local)*HOP - a0)/SR
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
