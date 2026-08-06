"""Full-song cast mix v2 — upgraded chain.
Per-voice: HPF -> static gain to target -> per-phrase constant gain (ramps only in
silent gaps; cannot dip mid-phrase) -> gentle leveler (+/-4dB slow drift) ->
de-ess (AFTER leveling: consistent threshold, fixes male sibilance) -> glue comp
-> peak limit -1dBFS.
Sum: concurrency-gated constant-power panning (solos center; ensembles spread
L/C/R) + 1/sqrt(N) unison law (user-approved climax fullness).
Bus: warmth/presence EQ + glue + TRUE-STEREO decorrelated reverb.
BGM: gentle sidechain duck (-1.2dB when vocals active).
Master: LUFS -12 -> limiter -> 4x-oversampled true-peak <= -1.0 dBTP.
"""
import glob, numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt, fftconvolve, lfilter, resample_poly
from scipy.ndimage import maximum_filter1d
import pyloudnorm as pyln

SR = 44100
D = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
OUT = "/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session/tokyo-summer-session_cast_MIX_v2.wav"
VOICES = ["honoka", "kotori", "umi", "otoya-tsuka", "cecil-ai", "camus-toya"]
BASE_PAN = {"honoka": -0.22, "kotori": 0.0, "umi": +0.22,
            "otoya-tsuka": -0.22, "cecil-ai": 0.0, "camus-toya": +0.22}

files = glob.glob(D + "/*.wav") + glob.glob(D + "/*.WAV")
def find(key):
    m = [f for f in files if key in f.lower()]
    if not m: raise SystemExit(f"missing stem for '{key}'")
    return m[0]

def load(p):
    y, sr = sf.read(p, always_2d=True)
    if sr != SR:
        y = np.stack([librosa.resample(y[:, c].astype(float), orig_sr=sr, target_sr=SR) for c in range(y.shape[1])], 1)
    if y.shape[1] == 1: y = np.repeat(y, 2, 1)
    return y.astype(np.float32)

bgm = load(find("offvocal"))
stems = {v: load(find(v)) for v in VOICES}
N = max([len(bgm)] + [len(s) for s in stems.values()])
def pad(y):
    return np.pad(y, ((0, N - len(y)), (0, 0))) if len(y) < N else y[:N]
bgm = pad(bgm); stems = {k: pad(v) for k, v in stems.items()}
print(f"N={N/SR:.1f}s, padded all to BGM length")

# ---------------- DSP ----------------
def op(x, tc): a = np.exp(-1/(tc*SR)); return lfilter([1-a], [1, -a], x, axis=0).astype(np.float32)
def zsm(x, tc):  # zero-phase one-pole smoothing (forward+backward)
    a = np.exp(-1/(tc*SR)); x = lfilter([1-a], [1, -a], x)
    return lfilter([1-a], [1, -a], x[::-1])[::-1]
def hpf(y, fc): return sosfilt(butter(2, fc/(SR/2), btype="high", output="sos"), y, axis=0).astype(np.float32)
def shelf(y, fc, g):
    A=10**(g/40); w0=2*np.pi*fc/SR; cw,sw=np.cos(w0),np.sin(w0); al=sw/2*np.sqrt((A+1/A)+2)
    b0=A*((A+1)-(A-1)*cw+2*np.sqrt(A)*al); b1=2*A*((A-1)-(A+1)*cw); b2=A*((A+1)-(A-1)*cw-2*np.sqrt(A)*al)
    a0=(A+1)+(A-1)*cw+2*np.sqrt(A)*al; a1=-2*((A-1)+(A+1)*cw); a2=(A+1)+(A-1)*cw-2*np.sqrt(A)*al
    return lfilter(np.array([b0,b1,b2])/a0, np.array([1,a1/a0,a2/a0]), y, axis=0).astype(np.float32)
def peaking(y, fc, g, Q=0.8):
    w0=2*np.pi*fc/SR; al=np.sin(w0)/(2*Q); A=10**(g/40); cw=np.cos(w0)
    bb=np.array([1+al*A,-2*cw,1-al*A]); aa=np.array([1+al/A,-2*cw,1-al/A])
    return lfilter(bb/aa[0], aa/aa[0], y, axis=0).astype(np.float32)
def deess(y, fc=6500, thr=-28.0, r=3.0):
    hi=hpf(y,fc); lo=y-hi; e=op(np.abs(hi.mean(1)),0.008)
    ov=np.maximum(0,20*np.log10(e+1e-9)-thr); return (lo+hi*(10**((-ov*(1-1/r))/20))[:,None]).astype(np.float32)
def comp(y, thr, r=3.0, rel=0.12, mk=0.0):
    e=op(np.abs(y.mean(1)),rel); ov=np.maximum(0,20*np.log10(e+1e-9)-thr)
    return (y*(10**((-ov*(1-1/r)+mk)/20))[:,None]).astype(np.float32)
def lim(y, c=0.977, look=0.003, rel=0.05):
    n=max(1,int(look*SR)); env=maximum_filter1d(np.max(np.abs(y),1),size=2*n+1,mode="nearest")
    g=op(np.minimum(1.0,c/(env+1e-9)),rel); return np.clip(y*np.minimum(g,1.0)[:,None],-c,c).astype(np.float32)
def arms(m):
    s=m[np.abs(m)>np.abs(m).max()*0.02]
    return float(np.sqrt(np.mean(s**2))+1e-12) if len(s) else 1e-9

def phrase_gain(y, target, top_db=40, merge_s=0.5, min_s=0.25, max_boost=10.0, max_cut=10.0):
    """Per-phrase CONSTANT gain: phrases (gaps<merge_s merged, so breaths never
    split a line) each get one gain -> anchored at target. Gain ramps live ONLY
    in silent gaps; 30ms corner smoothing. Mid-phrase dips are impossible."""
    m = y.mean(1).astype(np.float64)
    iv = librosa.effects.split(m, top_db=top_db, frame_length=2048, hop_length=256)
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] < merge_s*SR: merged[-1][1] = e
        else: merged.append([s, e])
    segs = [(s, e) for s, e in merged if e - s >= min_s*SR]
    if not segs: return y
    lv = [20*np.log10(np.sqrt(np.mean(m[s:e]**2))+1e-12) for s, e in segs]
    med = np.median(lv)
    pts_t, pts_g = [], []
    for (s, e), l in zip(segs, lv):
        if l < med - 18: continue          # isolated breath/noise: no anchor (no boost)
        g = np.clip(20*np.log10(target) - l, -max_cut, max_boost)
        pts_t += [s, e]; pts_g += [g, g]
    if not pts_t: return y
    gdb = np.interp(np.arange(len(m)), pts_t, pts_g)   # const in phrase, ramp across gaps
    gdb = zsm(gdb, 0.03)
    return (y * (10**(gdb/20))[:, None]).astype(np.float32)

def leveler(y, target, win=0.8, max_boost=4.0, max_cut=4.0, floor_rel=-42.0):
    """Gentle continuous auto-gain for slow drift WITHIN long phrases."""
    m = y.mean(1).astype(np.float64)
    hop = int(0.02*SR); fl = int(0.12*SR)
    rms = librosa.feature.rms(y=m, frame_length=fl, hop_length=hop, center=True)[0]
    rdb = 20*np.log10(rms+1e-9)
    floor = 20*np.log10(np.max(np.abs(m))+1e-9) + floor_rel
    voiced = rdb > floor
    if voiced.sum() < 2: return y
    idx = np.arange(len(rdb))
    rdb_i = np.interp(idx, idx[voiced], rdb[voiced])
    a = np.exp(-hop/(win*SR))
    lvl = lfilter([1-a], [1, -a], rdb_i); lvl = lfilter([1-a], [1, -a], lvl[::-1])[::-1]
    gdb = np.clip(20*np.log10(target) - lvl, -max_cut, max_boost)
    g = np.interp(np.arange(len(m)), idx*hop, gdb)
    return (y * (10**(g/20))[:, None]).astype(np.float32)

def activity(y, floor_rel=-38.0, smooth=0.15):
    m = np.abs(y.mean(1)).astype(np.float64)
    env = op(m, 0.03)
    thr = np.max(env) * 10**(floor_rel/20)
    return np.clip(zsm((env > thr).astype(np.float64), smooth), 0.0, 1.0)

def humanize(y, static_ms, seed, gate, walk_ms=12.0, walk_tc=1.0, lvl_db=1.2):
    """Ensemble humanization for same-source unison voices.

    Three same-source voices with identical timing/F0 fuse into ONE perceived
    singer no matter how different their timbres are — the ear groups sounds with
    identical onsets. A choir instead has, per singer: (a) onset scatter and
    (b) independent pitch wander. Both come from a drifting micro-delay:
    static offset (three distinct values) + random walk. The walk's slope IS a
    pitch deviation (d(delay)/dt): 12ms over ~1.0s ~= 1.2% ~= 20 cents, the
    classic choir amount. Plus independent slow level drift (singers' own
    dynamics). Max total delay stays < 30ms, the fusion/precedence limit —
    beyond that it reads as a discrete echo instead of another singer.
    `gate` (0..1, slow) confines all of it to ensemble sections; solos stay put.
    """
    n = len(y)
    rng = np.random.RandomState(seed)
    def walk(tc, blk=2048):
        w = rng.randn(n//blk + 2)
        w = np.interp(np.arange(n), np.arange(len(w))*blk, w)
        w = zsm(w, tc)
        return w / (np.max(np.abs(w)) + 1e-9)
    d = (gate * (static_ms + walk_ms*walk(walk_tc))) * SR/1000.0     # delay, samples
    g = 10 ** ((gate * lvl_db * walk(3.0)) / 20.0)                   # own dynamics
    idx = np.arange(n) - d
    out = np.empty_like(y)
    for c in range(y.shape[1]):
        out[:, c] = np.interp(idx, np.arange(n), y[:, c]) * g
    return out.astype(np.float32)

def reverb_st(y, send=0.14, decay=0.9, predelay=0.015):
    """TRUE-stereo plate: decorrelated L/R IRs (indep. noise), energy-normalized,
    wet matched to dry active RMS then mixed at `send` on top of unscaled dry."""
    n = int(decay*SR); t = np.arange(n)
    def mk(seed):
        ir = np.random.RandomState(seed).randn(n)*np.exp(-t/(decay*SR/4))
        ir = sosfilt(butter(2, [350/(SR/2), 7000/(SR/2)], btype="band", output="sos"), ir)
        ir = np.concatenate([np.zeros(int(predelay*SR)), ir])
        return (ir/np.sqrt(np.sum(ir**2)+1e-12)).astype(np.float32)
    wl = fftconvolve(y[:, 0], mk(7))[:len(y)]
    wr = fftconvolve(y[:, 1], mk(1234))[:len(y)]
    w = np.stack([wl, wr], 1).astype(np.float32)
    w *= arms(y.mean(1)) / (arms(w.mean(1)) + 1e-12)
    return (y + send*w).astype(np.float32)

def vstd(y):   # std of per-phrase mean level (dB), breath-robust: drop segs >10dB below median
    m=y.mean(1).astype(float)
    iv=librosa.effects.split(m, top_db=30, frame_length=2048, hop_length=512)
    lv=np.array([20*np.log10(np.sqrt(np.mean(m[s:e]**2))+1e-9) for s,e in iv if (e-s)>int(0.3*SR)])
    if len(lv)<2: return 0.0
    lv=lv[lv> np.median(lv)-10]
    return float(np.std(lv)) if len(lv)>1 else 0.0

# ---------------- per-voice chain ----------------
bgm_r = arms(bgm.mean(1)); target = bgm_r * 10**(5/20)          # vocals +5 dB over BGM (approved)
target_db = 20*np.log10(target)
print(f"BGM active RMS {20*np.log10(bgm_r):.1f} dBFS -> vocal target {target_db:.1f} dBFS")
vp = {}
for v in VOICES:
    y = hpf(stems[v], 75)
    b = vstd(y)
    g = target / arms(y.mean(1)); y = (y*g).astype(np.float32)    # static: avg -> target
    y = phrase_gain(y, target)                                    # per-phrase anchor (no dips)
    y = leveler(y, target)                                        # +/-4dB slow drift
    y = deess(y, thr=target_db-16)                                # AFTER leveling (works for males now)
    y = comp(y, thr=target_db, r=2.5, rel=0.12)                   # gentle glue
    a2 = vstd(y)
    y = lim(y, c=0.89)                                            # per-voice peak ~-1dBFS
    print(f"  {v:12s} static {20*np.log10(g):+5.1f}dB | phrase-std {b:4.1f}->{a2:4.1f} dB")
    vp[v] = y

# ---------------- concurrency: unison law + gated panning ----------------
acts = {v: activity(y) for v, y in vp.items()}
nv = np.sum(list(acts.values()), axis=0)
# Unison law 1/N^P. P=0.85 was too tame ("不够气势"); 0.5 = equal-power baseline;
# 0.42 leans fuller so the 6-voice finale is the loudest point of the song. Applied
# to ALL voices — a musical choice, not a male-only fudge.
P = 0.42
scale = (1.0 / np.maximum(nv, 1.0)**P).astype(np.float32)
wpan = np.clip(zsm(nv - 1.0, 0.3), 0.0, 1.0)                      # 0 solo .. 1 ensemble

# male ensemble humanization: the 3 males share ONE source performance (identical
# timing/F0) -> unison reads as one person. Gate on >=2 males singing; cecil (C)
# stays as the dry anchor; otoya/camus get drifting micro-timing.
male_gate = np.clip(zsm(acts["otoya-tsuka"] + acts["cecil-ai"] + acts["camus-toya"] - 1.0, 1.5), 0.0, 1.0)
# all three get their OWN offset + walk (no "dry anchor" — an unmoved voice keeps
# anchoring the ear to a single source). Spread ~22ms, peak delay <30ms.
MALE_HUM = {"otoya-tsuka": (+10.0, 101), "cecil-ai": (-6.0, 303), "camus-toya": (+16.0, 202)}
for v, (ms, seed) in MALE_HUM.items():
    vp[v] = humanize(vp[v], ms, seed, male_gate)
# NOTE: no makeup gain. Measured: the 3 male stems are already waveform-incoherent
# (pairwise corr ~0.02 even raw — different models give different phase), so they
# sum by power, and humanizing costs no level (sum RMS -24.9 dB before AND after).
# The earlier +1.5/+2.0dB "compensation" was unjustified and made the 3-male section
# louder than the 6-voice climax.
print(f"  male humanize: gate active {100*np.mean(male_gate>0.5):.0f}% of time | "
      f"offsets {{{', '.join(f'{k.split(chr(45))[0]}{v[0]:+.0f}ms' for k,v in MALE_HUM.items())}}} "
      f"±12ms walk + ±1.2dB drift | no makeup (decorrelation is level-neutral)")
vbus = np.zeros((N, 2), np.float32)
for v, y in vp.items():
    p = BASE_PAN[v] * wpan                                        # gated pan position
    ang = (p + 1.0) * np.pi/4
    gL = (np.sqrt(2)*np.cos(ang)).astype(np.float32)              # unity at center
    gR = (np.sqrt(2)*np.sin(ang)).astype(np.float32)
    vbus[:, 0] += y[:, 0]*gL; vbus[:, 1] += y[:, 1]*gR
vbus *= scale[:, None]
print(f"  concurrency: max {nv.max():.1f} | multi(N>1.5) {100*np.mean(nv>1.5):.0f}% | "
      f"scale {scale.min():.2f}..1.00 | pan gate active {100*np.mean(wpan>0.5):.0f}% of time")

# ---------------- vocal bus ----------------
vbus = shelf(vbus, 180, 1.5); vbus = peaking(vbus, 3800, 1.5)
vbus = comp(vbus, thr=target_db+6, r=2.0, rel=0.15)
vbus = reverb_st(vbus, send=0.10)   # drier: pull the vocal forward (less "karaoke room")

# ---------------- BGM duck + master ----------------
act_any = np.clip(zsm(np.clip(nv, 0, 1), 0.25), 0, 1)
bgm_d = (bgm * (10**(-1.2*act_any/20))[:, None]).astype(np.float32)
print(f"  BGM duck: -1.2 dB x activity (engaged {100*np.mean(act_any>0.5):.0f}% of time)")
mix = bgm_d + vbus
meter = pyln.Meter(SR)
mix = (mix * 10**((-12.0 - meter.integrated_loudness(mix))/20)).astype(np.float32)
mix = lim(mix, c=0.891)                                           # sample peak ~-1 dBFS
tp = 20*np.log10(max(np.max(np.abs(resample_poly(mix[:, 0], 4, 1))),
                     np.max(np.abs(resample_poly(mix[:, 1], 4, 1)))) + 1e-12)
if tp > -1.0:
    mix = (mix * 10**((-1.0 - tp)/20)).astype(np.float32); tp = -1.0
lf = meter.integrated_loudness(mix)
sf.write(OUT, mix, SR, subtype="PCM_24")
print(f"\n-> {OUT}\n   {N/SR:.1f}s | {lf:.1f} LUFS | true-peak {tp:.2f} dBTP")
