# umi — 亮嗓女高音 RVC v2 (第二女声)

> 📖 **跨音色统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。**
> 本文档只记 umi 特有增量。配方与 honoka / kotori 同族 (亮嗓女高音 → TITAN + 中值谱)。
>
> 角色: umi | 模型: RVC v2, 40kHz, RMVPE, ContentVec 768 | GPU: RTX 3090

---

## 一、语料 (v4.5)
- 源: `raw/umi_raw` (57 曲) → v4.5 语料重制 (§3) → **56 个干净人声文件, 237.9min**。
- 语料超目标量, 走打分择优 (`_score_slices.py`, §5): **2943 片入选 (~180min)** →
  `filelist.txt` 2945 行 (含 2 条 mute) → **185 steps/epoch**。

## 二、训练 (TITAN, §6)
- 底模 **TITAN** (亮嗓女高音, 见 METHODOLOGY §6.1)。`-se20 -te200 -bs16 -sr40k -f0 1`。
- 启动确认 `loaded pretrained ... All keys matched successfully` ✅ (G+D)。
- 完整训到 e200, 部署 10 个 `*_infer.pth`; 重建用 **e100–200 窗口** (6 ckpt:
  e101/121/142/162/182/202)。

## 三、⚠️ 底模 A/B: f0G40k 并不能让音色变"沉" (负结果)

用户听感"umi 偏亮", 提出**换官方 f0G40k 底模是否会更沉**。为此训了完整的第二臂
(`models_v2_f0g40k`, 仅 `-pg/-pd` 不同, 同语料同配方), 同源渲染后同类对比:

| | 谱质心 | air >12k | 中位 F0 |
|---|---:|---:|---:|
| **TITAN** (定稿) | 4022 Hz | −31.7 dB | 507.0 Hz |
| f0G40k (实验臂) | 4026 Hz | **−29.5 dB** | 508.6 Hz |
| 差 | +4 Hz | **+2.2 dB** | +1.6 Hz |

**结论: 换底模不解决"偏亮"。** 中位 F0 几乎不动 (+1.6Hz / +0.3%) → **不会更沉**;
谱质心持平; air 频段 f0G40k 反而**更亮 +2.2dB**。
→ **亮度是语料固有属性 (data-intrinsic), 底模改不了它**;想要更沉只能换音域更低的源歌手
(同 §11 音色天花板的道理)。**定稿保留 TITAN**, f0G40k 臂仅作对照留存。

## 四、推理 + 成品 (§7)
- `_infer_median.py`: e100–200 各 ckpt 推理 → **中值谱 ensemble** → `stage5_rebuilt/umi_MEDIAN.wav`。
- `_remove_breaths.py --duck-db -99`: 呼吸全静音 (用户口味, 同 honoka / kotori)。
- **配方定稿** = TITAN + 中值谱 + 呼吸静音, 与另两个女声一致。

## 五、状态
- ✅ **定稿 6 音色之一**; 配方经 25s 小样本耳测锁定。
- ✅ **整曲已渲染** (2026-08-05): 女声全曲干声源 (又海) 录制完成后照配方重跑 →
  `output/final/FINAL_umi_v2.wav` (207.9s), 同时归集到
  `tokyo_summer_session/female_full_covers/umi_full.wav`。占位文件已移除。
- 成品混音中的用法见 [MIX_TABLE.md](../../MIX_TABLE.md) (💙 海组 · 海♀)。
