# camus — 纯单音色 RVC v2

> 📖 **统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。** 本文档只记 camus 特有内容。
>
> 日期: 2026-07-31 | RVC v2, 40kHz | **纯单音色 (无混合)** | 底模: 官方 f0G40k (男声)
>
> 用途: 既是一个候选男声音色, 也是 [METHODOLOGY §11 音色区分度天花板](../../METHODOLOGY.md) 的关键实证 ——
> **证明相近歌手想出区别, 纯单音色 > 任何混合/index_rate 调节。**

---

## 一、语料
- camus 12 首独唱, 40.8min。**复用 toya_camus_mix 已分离的 camus stem** (§11.3 复用技巧, 免重分离)。
- 纯单音色: 无 `_b` 复制, 无第二声部。
- prep: `v2_prep_all` → 560 slices (语料 <180min, 全量保留, 跳过打分择优)。

## 二、训练
- **f0G40k/f0D40k** warm-start (男声: camus 语料 F0 median 222Hz → §6 亮嗓规则判"男声/暗嗓")。
- `-te200 -se20 -bs16`; steps/epoch ~35; e100-200 窗口 6 个 ckpt 中值谱 ensemble。
- 启动已确认 `loaded pretrained ... All keys matched successfully`。

## 三、成品与关键发现
- 男声源 = `夏日.wav` (male_sources, 真人干声, §11.4) → 中值谱逐段组装 → `FINAL_camus_v2.wav` (159.9s)。
- 混音: **`camus-kotori_MIX.wav`** (camus + kotori 女声 + off-vocal BGM, 24.5s 对齐样本, −12 LUFS)。
- **音色区分度实证** (log-mel 逐格 dB, 详见 [METHODOLOGY §11](../../METHODOLOGY.md)):
  - 亮度 2022Hz (比 otoya 2167 更暗 —— camus 本人音色偏暗)。
  - **距 toya (toya_camus 2:1) = 2.90dB, 命中源歌手天花板** (toya vs camus 源 LTAS 差 2.81dB)。
  - 对照混合版仅 1.76dB (2:1) / 1.90dB (c2) —— 听不出。
  - → **结论: toya 与 camus 两人音色本就相近 (源差 <3dB); 想要听得出区别的 camus 音色,
    只有纯单音色能恢复源歌手的全部区分度, 混合与 index_rate 都撞天花板。**
