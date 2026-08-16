"""Did the -9.0 dB target actually land v17 on the 6-seiyuu original's balance?

Same measurement that found the 4.1 dB error: vocal bus vs BGM bus for ours (exact,
they are kept separate until the sum), LS-rescaled complementary stems for the
original, both loudness-aligned so only the internal balance is compared."""
import glob, numpy as np, soundfile as sf, pyloudnorm as pyln

SR = 44100
SP = "/mnt/c/Users/kevin/AppData/Local/Temp/claude/c--Users-kevin-ai-sing-by-ai/5135e3f3-7fe4-43b1-b058-d5c229f65a8e/scratchpad"
WORK = SP + "/sixtmp"
REFMIX = "/mnt/c/Users/kevin/Desktop/AI翻唱/tokyo_summer_6seiyuu_mix.wav"
J = "/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/tokyo_summer_session/"
mono = lambda p: sf.read(p, always_2d=True)[0].mean(1).astype(np.float64)
rms = lambda x: float(np.sqrt(np.mean(x**2))+1e-12)

rm = mono(REFMIX)
rv = mono(WORK+"/ref_voc.wav")
ri = mono([f for f in glob.glob(WORK+"/*.wav") if "6seiyuu_mix" in f and "(other)" in f][0])
n = min(len(rm), len(rv), len(ri)); rm, rv, ri = rm[:n], rv[:n], ri[:n]
ab, *_ = np.linalg.lstsq(np.stack([rv, ri], 1), rm, rcond=None)
rv, ri = rv*ab[0], ri*ab[1]

W = SR
fr = lambda x: np.array([np.sqrt(np.mean(x[i:i+W]**2))+1e-12 for i in range(0, len(x)-W, W//2)])
meter = pyln.Meter(SR)
rows = []
for lab, mixp, vp, bp in [
        ("六声优版原曲", REFMIX, None, None),
        ("v16 (旧配平)", J+"tokyo-summer-session_lovelive-cover_v16.wav",
         SP+"/bus_vocal.wav", SP+"/bus_bgm.wav"),
        ("v17 (新配平)", J+"tokyo-summer-session_lovelive-cover_v17.wav",
         SP+"/bus17_vocal.wav", SP+"/bus17_bgm.wav")]:
    if vp is None:
        voc, ins, mix = rv, ri, rm
    else:
        mix = mono(mixp); voc, ins = mono(vp), mono(bp)
        k = min(len(mix), len(voc), len(ins)); mix, voc, ins = mix[:k], voc[:k], ins[:k]
        g = rms(mix)/rms(voc+ins)                    # master gain, common path
        voc, ins = voc*g, ins*g
    fv, fi = fr(voc), fr(ins)
    act = 20*np.log10(fv) > 20*np.log10(fv).max()-20
    lv, li = float(np.median(20*np.log10(fv[act]))), float(np.median(20*np.log10(fi[act])))
    lufs = meter.integrated_loudness(np.stack([mix, mix], 1))
    rows.append((lab, lv, li, lv-li, lufs))

print("对齐到 -12 LUFS 后 (只比内部平衡)")
print(f"{'':<16}{'人声':>9}{'伴奏':>9}{'差值':>9}")
print("-"*44)
for lab, lv, li, d, lufs in rows:
    a = -12.0 - lufs
    print(f"{lab:<16}{lv+a:>+8.1f}{li+a:>+9.1f}{d:>+9.1f}")
ref_d = rows[0][3]
for lab, lv, li, d, lufs in rows[1:]:
    gap = d - ref_d
    print(f"\n{lab} 对原曲: {gap:+.1f} dB "
          f"{'✔ 相符 (<=0.5dB)' if abs(gap) <= 0.5 else '← 仍需调整'}")
    if abs(gap) > 0.5 and "v17" in lab:
        print(f"   建议把人声目标从 -9.0 再改为 {-9.0 - gap:+.1f} dB")
