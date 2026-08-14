"""Absolute alignment check on the NEW exports: for each phrase, measure the residual
lag between the stem and the original song's vocals. Doesn't need the old exports —
it asks "is this where it belongs?", not "did it move?".

Only meaningful where that voice is the one singing in the reference, so we restrict
to phrases the stem itself is active in and report the distribution, flagging any
phrase beyond ±30 ms (the fusion/echo threshold from METHODOLOGY §12.3)."""
import glob, os, numpy as np, soundfile as sf, librosa
from scipy.signal import correlate

SR, HOP = 44100, 128
EXP = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
REF = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_vocals.wav"
VOICES = ["honoka", "kotori", "umi", "otoya-tsuka", "cecil-ai", "camus-toya"]
LAB = {"honoka":"果♀","kotori":"鸟♀","umi":"海♀",
       "otoya-tsuka":"果♂","cecil-ai":"鸟♂","camus-toya":"海♂"}
find = lambda k: [f for f in glob.glob(EXP+"/*.wav") if k in os.path.basename(f).lower()][0]

ref = librosa.load(REF, sr=SR, mono=True)[0]
env_r = np.abs(ref[:len(ref)//HOP*HOP].reshape(-1, HOP)).max(1)

print(f"{'音色':<6}{'乐句数':>6}{'中位残差':>10}{'|残差| p90':>11}{'超 ±30ms':>10}   最差乐句")
print("-"*72)
for v in VOICES:
    y = sf.read(find(v), always_2d=True)[0].mean(1)
    env = np.abs(y[:len(y)//HOP*HOP].reshape(-1, HOP)).max(1)
    iv = librosa.effects.split(y, top_db=40, frame_length=2048, hop_length=256)
    merged = []
    for s, e in iv:
        if merged and s-merged[-1][1] < 0.5*SR: merged[-1][1] = e
        else: merged.append([s, e])
    lags = []
    for s, e in merged:
        if e-s < 1.0*SR: continue
        i0, i1 = s//HOP, e//HOP
        a = env[i0:i1]; b = env_r[i0:min(i1, len(env_r))]
        n = min(len(a), len(b))
        if n < 40: continue
        a, b = a[:n]-a[:n].mean(), b[:n]-b[:n].mean()
        if a.std() < 1e-6 or b.std() < 1e-6: continue
        if float(np.corrcoef(a, b)[0, 1]) < 0.35: continue      # ref busy with the other voice
        c = correlate(a, b, mode="same")
        lags.append(((np.argmax(c)-n//2)*HOP/SR*1000, s/SR))
    if not lags:
        print(f"{LAB[v]:<6}{'—':>6}{'(参照里基本是另一声部)':>22}"); continue
    L = np.array([l for l, _ in lags])
    worst = max(lags, key=lambda x: abs(x[0]))
    bad = int(np.sum(np.abs(L) > 30))
    print(f"{LAB[v]:<6}{len(L):>6}{np.median(L):>+9.0f}ms{np.percentile(np.abs(L),90):>10.0f}ms"
          f"{bad:>9}   {int(worst[1]//60)}:{worst[1]%60:04.1f} ({worst[0]:+.0f}ms)")
