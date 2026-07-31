# cecil_ai_mix — 混合音色 RVC v2 (cecil : ai 2:1)

> 📖 **统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。** 本文档只记特有内容。
>
> 日期: 2026-07-30 | RVC v2, 40kHz | **cecil ×2 + ai ×1** | 底模: 官方 f0G40k (男声)
>
> ⭐ **定稿男声之一** (6 音色)。

---

## 一、语料 (cecil:ai ≈ 2:1)
| 声部 | 曲数 | 复制 |
|---|---|---|
| cecil (愛島セシル) | 10 | ×2 (`_b` 文件级复制) |
| ai (美風藍) | 10 | ×1 |

- v4.5 分离 (30 单位 = cecil 10×2 + ai 10) → 1554 切片 / 92.8min (<180min 全量保留)。
- **F0 median 249.7Hz** (cecil 222 + ai 287; ai 偏高亮但 2:1 加权后落男声区) → §6 规则判**男声 → f0G40k**。

## 二、训练
- **f0G40k** warm-start; `-te200 -se20 -bs16`; e100-200 窗口中值谱 ensemble。
- 启动已确认 `All keys matched successfully`。

## 三、成品
- 男声源 `夏日.wav` (male_sources) → 中值谱逐段组装 → `FINAL_cecil_ai_v2.wav` (159.9s)。
- 混音: **`cecil-ai-kotori_MIX.wav`** (24.5s 对齐样本; 男声 +15.5dB / kotori 女声 +5.7dB, −12 LUFS)。
