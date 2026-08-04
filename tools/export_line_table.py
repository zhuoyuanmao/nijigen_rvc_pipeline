"""Export a per-line (per-phrase) table of every mixing decision, for review.
Replicates the mix chain's analysis stages and reports, for each sung phrase:
start/end, singer(s), voices-in-unison, static gain, phrase gain, net gain,
pan position, unison scale, and whether male humanization was engaged.
-> scratchpad/line_table.md (+ .csv)"""
import glob, numpy as np, soundfile as sf, librosa
from scipy.signal import butter, sosfilt, lfilter

SR = 44100
D = "/mnt/c/Users/kevin/Desktop/tokyo_summer_session_au/导出"
VOICES = ["honoka", "kotori", "umi", "otoya-tsuka", "cecil-ai", "camus-toya"]
LABEL = {"honoka": "honoka", "kotori": "kotori", "umi": "umi",
         "otoya-tsuka": "honoka-male", "cecil-ai": "kotori-male", "camus-toya": "umi-male"}
GRP = {"honoka": "🧡", "honoka-male": "🧡", "kotori": "🤍", "kotori-male": "🤍",
       "umi": "💙", "umi-male": "💙"}
SNG = {"honoka": "果♀", "honoka-male": "果♂", "kotori": "鸟♀", "kotori-male": "鸟♂",
       "umi": "海♀", "umi-male": "海♂"}
# lyric mapping keyed by (voice label, int(start second)) — from the cast sheet
LY = {
 ("kotori",6):"「やぁ、こんにちは」", ("kotori-male",7):"「こんにちは」→ 合 Ah____",
 ("kotori",9):"合 Ah____", ("kotori-male",11):"「ねぇ、調子どう？」",
 ("kotori",13):"「普通かな…」→ 合 Uh____", ("kotori-male",15):"合 Uh____",
 ("honoka",30):"「花火大会が来週あるんだってね」", ("honoka-male",35):"「あーゆー人が多いの、俺は苦手なんだよな」",
 ("honoka",41):"「あーあ。それじゃ誰か他をあたってみっか」", ("honoka-male",47):"「やっぱ、楽しそうだな。結構行きたいかも」",
 ("umi",54):"「喉渇いたな」", ("umi-male",55):"「これ飲めば？」",
 ("umi",57):"「これっていわゆる間接KISS？」", ("umi-male",60):"「意識した？」",
 ("umi",61):"「意識した…」→ 合 喉は渇いたまんま", ("umi-male",63):"合 喉は渇いたまんま",
 ("kotori-male",66):"待ってる左手にほんの少し触れてみる／繋ぎたい 繋ぎたい だけどポケットに隠れた",
 ("kotori",78):"ほんとは気づいてる ほんの少しで届く距離", ("kotori",84):"繋ぎたい 繋ぎたい 本音 背中に隠すの",
 ("honoka",107):"「ヒント！何か今日は違う気がしませんか？」", ("honoka",119):"「殴るよ？！１５センチ切った髪に気づけ」",
 ("umi",128):"「これ食べて」", ("umi",133):"「夏なのに！」→ 合 喉が渇きますね",
 ("umi",138):"ほんとは気づいてる ほんの少しで届く距離", ("umi",144):"繋ぎたい 繋ぎたい 掴む 袖口ひいてみる",
 ("umi-male",150):"「綺麗だね」", ("umi",151):"「綺麗だね」→ 合 Ah____", ("umi-male",153):"合 Ah____",
 ("umi-male",156):"「綺麗だよ」", ("umi",158):"「綺麗だよ」→ 合 Uh____", ("umi-male",159):"合 Uh____",
 ("kotori",162):"遠くから見てただけの", ("kotori-male",162):"遠くから見てただけの",
 ("honoka",165):"花火が今目の前に", ("honoka-male",165):"花火が今目の前に",
 ("umi-male",168):"時を止め帰りたくないよね、今日は", ("umi",168):"時を止め帰りたくないよね、今日は",
 ("honoka-male",188):"待ってる左手に ほんの少し触れてみる", ("kotori-male",188):"待ってる左手に ほんの少し触れてみる",
 ("umi-male",188):"待ってる左手に ほんの少し触れてみる",
 ("honoka-male",194):"繋ぎたい 繋ぎたい 君を 黙って奪うよ", ("kotori-male",194):"繋ぎたい 繋ぎたい 君を 黙って奪うよ",
 ("umi-male",194):"繋ぎたい 繋ぎたい 君を 黙って奪うよ",
 ("honoka",200):"ほんとは気づいてる ほんの少しで届く距離", ("kotori",200):"ほんとは気づいてる ほんの少しで届く距離",
 ("umi",200):"ほんとは気づいてる ほんの少しで届く距離", ("honoka-male",200):"ほんとは気づいてる ほんの少しで届く距離",
 ("kotori-male",200):"ほんとは気づいてる ほんの少しで届く距離", ("umi-male",200):"ほんとは気づいてる ほんの少しで届く距離",
 ("honoka",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ", ("kotori",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ",
 ("umi",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ", ("honoka-male",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ",
 ("kotori-male",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ", ("umi-male",206):"繋ぎたい 繋ぎたい ぎゅっと 握り返すよ",
}
def lyric(lab, t0):
    for d in (0, -1, 1):
        k = (lab, int(t0)+d)
        if k in LY: return LY[k]
    return "—"
BASE_PAN = {"honoka": -0.22, "kotori": 0.0, "umi": +0.22,
            "otoya-tsuka": -0.22, "cecil-ai": 0.0, "camus-toya": +0.22}
files = glob.glob(D + "/*.wav")
def find(k):
    return [f for f in files if k in f.lower()][0]
def load(p):
    y, sr = sf.read(p, always_2d=True)
    if y.shape[1] == 1: y = np.repeat(y, 2, 1)
    return y.astype(np.float32)
def op(x, tc): a=np.exp(-1/(tc*SR)); return lfilter([1-a],[1,-a],x,axis=0).astype(np.float32)
def zsm(x, tc):
    a=np.exp(-1/(tc*SR)); x=lfilter([1-a],[1,-a],x); return lfilter([1-a],[1,-a],x[::-1])[::-1]
def hpf(y, fc): return sosfilt(butter(2, fc/(SR/2), btype="high", output="sos"), y, axis=0).astype(np.float32)
def arms(m):
    s=m[np.abs(m)>np.abs(m).max()*0.02]
    return float(np.sqrt(np.mean(s**2))+1e-12) if len(s) else 1e-9
def activity(y, floor_rel=-38.0, smooth=0.15):
    m=np.abs(y.mean(1)).astype(np.float64); env=op(m,0.03)
    return np.clip(zsm((env>np.max(env)*10**(floor_rel/20)).astype(np.float64), smooth),0,1)

bgm = load(find("offvocal"))
target = arms(bgm.mean(1))*10**(5/20); tdb = 20*np.log10(target)
stems = {v: hpf(load(find(v)), 75) for v in VOICES}

# static gains + phrase segmentation (same params as the mix)
sg, phr, acts = {}, {}, {}
for v in VOICES:
    y = stems[v]
    g = target/arms(y.mean(1)); sg[v] = 20*np.log10(g)
    y = (y*g).astype(np.float32); stems[v] = y
    acts[v] = activity(y)
    m = y.mean(1).astype(np.float64)
    iv = librosa.effects.split(m, top_db=40, frame_length=2048, hop_length=256)
    merged=[]
    for s,e in iv:
        if merged and s-merged[-1][1] < 0.5*SR: merged[-1][1]=e
        else: merged.append([s,e])
    segs=[(s,e) for s,e in merged if e-s>=0.25*SR]
    lv=[20*np.log10(np.sqrt(np.mean(m[s:e]**2))+1e-12) for s,e in segs]
    med=np.median(lv) if lv else 0
    phr[v]=[(s,e,l,float(np.clip(tdb-l,-10,10)) if l>=med-18 else 0.0) for (s,e),l in zip(segs,lv)]

nv = np.sum(list(acts.values()), axis=0)
male_gate = np.clip(zsm(acts["otoya-tsuka"]+acts["cecil-ai"]+acts["camus-toya"]-1.0, 1.5),0,1)
wpan = np.clip(zsm(nv-1.0, 0.3), 0, 1)

rows=[]
for v in VOICES:
    for s,e,l,pg in phr[v]:
        n_mid = float(np.mean(nv[s:e])); pan = BASE_PAN[v]*float(np.mean(wpan[s:e]))
        rows.append(dict(t0=s/SR, t1=e/SR, voice=LABEL[v], raw=l, sg=sg[v], pg=pg,
                         net=sg[v]+pg, n=n_mid, pan=pan,
                         scale=20*np.log10(1/max(n_mid,1.0)**0.5),
                         hum=float(np.mean(male_gate[s:e]))))
rows.sort(key=lambda r: r["t0"])

def fmt(t): return f"{int(t//60)}:{t%60:04.1f}"
HEAD = """# 逐句混音参数表 — 東京サマーセッション (6 音色 cast 版)

> 成品: `tokyo-summer-session_cast_MIX.wav` (227.9s · −12.2 LUFS · 真峰值 −1.0 dBTP)
> 混音方法论: [METHODOLOGY §12](METHODOLOGY.md) · 参考实现: [tools/mix_full_cast.py](tools/mix_full_cast.py)
> 机器可读版: [tools/line_table.csv](tools/line_table.csv)

## 角色与符号

| 组 | 女声 | 男声 | 音色实现 |
|---|---|---|---|
| 🧡 果组 | 果♀ `honoka` | 果♂ `honoka-male` | otoya:tsukasa 2:1 |
| 🤍 鸟组 | 鸟♀ `kotori` | 鸟♂ `kotori-male` | cecil:ai 2:1 |
| 💙 海组 | 海♀ `umi` | 海♂ `umi-male` | camus + 20% toya |

> 男声按**配对的女声**命名, cast 结构一目了然。**爱心颜色 = 该句属于哪一组**;
> 同一时间出现两行同色 = **该组男女合唱**(如 `合 Ah____`); 六行同时 = **全员齐唱**。

## 曲式结构

| 时间 | 段落 | 谁唱 |
|---|---|---|
| 0:06–0:17 | 开头对白 | 🤍 鸟组 (男女交替 + 合 Ah/Uh) |
| 0:30–0:53 | 第一段主歌 A | 🧡 果组 (男女交替 ×4) |
| 0:54–1:06 | 第一段主歌 B | 💙 海组 (男女交替 + 合) |
| 1:06–1:30 | 第一段副歌 | 🤍 鸟组 |
| 1:41–2:14 | 第二段主歌 | 🧡 果组 → 💙 海组 (**男声待补**) |
| 2:18–2:30 | 第二段副歌 | 💙 海♀ |
| 2:30–2:41 | 綺麗だね/だよ | 💙 海组 (男女交替 + 合) |
| 2:42–2:55 | 三对 CP 特写 | 🤍 → 🧡 → 💙 (各组男女合唱) |
| 3:08–3:20 | 待ってる左手に | **三男齐唱** |
| 3:20–3:34 | 最后的副歌 | **六人齐唱 (高潮)** |

## ⏳ 待补 (尚未录制/生成 cover, 明日补)

| 时间 | 演唱 | 歌词 |
|---|---|---|
| ~1:41 | 🧡 果♂ | 「何怒ってんの？気に障る事しましたっけ？」 |
| ~1:53 | 🧡 果♂ | 「分かった！気にしないでいいよ太ったこと」 |
| ~2:06 | 💙 海♂ | 「お腹空いたな」 |
| ~2:09 | 💙 海♂ | 「これっていわゆる手作りクッキー？」 |
| ~2:11 | 💙 海♂ | 「夏なのに？」 |
| ~3:00 | 🤍 鸟♀ | 「好きかもね」 |
| ~3:03 | 🤍 鸟♂ | 「好きかもね」 |

> 已核实: 这些时段男声 stem 为数字静音 (−99dBFS), 即**尚未演唱/生成**, 非混音丢失。
> 补齐后重跑 `tools/mix_full_cast.py` + 本表生成脚本即可。

## 读表说明

| 列 | 含义 |
|---|---|
| 原始电平 | 该句在对齐 stem 里的原始 RMS (dBFS) |
| 静态增益 | 整轨均值 → 目标的固定增益 (女 +8~9 / 男 +17~18dB — 男声源本就轻约 9dB) |
| 逐句增益 | 在静态之上, 该句为达到统一目标的额外增减 (逐句恒定, 只在句间过渡) |
| **净增益** | 静态 + 逐句 = 该句实际被抬升的总量 |
| 同唱人数 | 该句时间窗内的平均并发音色数 (1=独唱, 3=男声齐唱, 6=全员) |
| 声像 | C=正中 / L·R+数值 = 左右偏移百分比 (仅齐唱段展开, 独唱保持正中) |
| 齐唱缩放 | 1/√N 齐唱定律施加的衰减 (防多人叠加过响) |
| 合唱去相关 | 男声人性化微时移是否生效 (3 男同源, 齐唱时需去相关才像多人) |

> 目标电平: 人声 = BGM 有声段 RMS **+5dB** (本曲 = −11.9 dBFS)。

## 逐句表 (55 乐句)

"""
md = ["| # | 时间 | 时长 | 组 | 演唱 | 歌词 | 音色 | 原始 | 静态 | 逐句 | 净增益 | 同唱 | 声像 | 齐唱缩放 | 去相关 |",
      "|---:|---|---:|:-:|:-:|---|---|---:|---:|---:|---:|---:|:-:|---:|:-:|"]
csv = ["idx,start_s,end_s,dur_s,group,singer,lyric,voice,raw_dbfs,static_gain_db,phrase_gain_db,"
       "net_gain_db,concurrent_voices,pan,unison_scale_db,humanize"]
for i,r in enumerate(rows,1):
    pan = "C" if abs(r["pan"])<0.02 else (f"L{abs(r['pan'])*100:.0f}" if r["pan"]<0 else f"R{r['pan']*100:.0f}")
    ly = lyric(r["voice"], r["t0"]); g = GRP[r["voice"]]; s = SNG[r["voice"]]
    md.append(f"| {i} | {fmt(r['t0'])}–{fmt(r['t1'])} | {r['t1']-r['t0']:.1f}s | {g} | {s} | {ly} | `{r['voice']}` | "
              f"{r['raw']:.1f} | {r['sg']:+.1f} | {r['pg']:+.1f} | **{r['net']:+.1f}** | {r['n']:.1f} | {pan} | "
              f"{r['scale']:+.1f} | {'是' if r['hum']>0.5 else '—'} |")
    csv.append(f"{i},{r['t0']:.2f},{r['t1']:.2f},{r['t1']-r['t0']:.2f},{g},{s},\"{ly}\",{r['voice']},{r['raw']:.1f},"
               f"{r['sg']:.1f},{r['pg']:.1f},{r['net']:.1f},{r['n']:.2f},{r['pan']:.2f},{r['scale']:.1f},{r['hum']:.2f}")
R="/mnt/c/Users/kevin/ai_sing_by_ai/nijigen_rvc_pipeline/"
open(R+"MIX_TABLE.md","w",encoding="utf-8").write(HEAD + "\n".join(md) + "\n")
open(R+"tools/line_table.csv","w",encoding="utf-8").write("\n".join(csv) + "\n")
print(f"{len(rows)} phrases -> MIX_TABLE.md / tools/line_table.csv")
print(f"target {tdb:.1f} dBFS | net gain range {min(r['net'] for r in rows):+.1f}..{max(r['net'] for r in rows):+.1f} dB")
print("\n".join(md[:12]))
