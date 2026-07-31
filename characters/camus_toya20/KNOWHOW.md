# camus_toya20 — camus + 20% toya 微调档 (定稿男声)

> 📖 **统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。** 本文档只记特有内容。
>
> 日期: 2026-07-31 | RVC v2, 40kHz | camus (12曲) + **20.4% toya** (时长加权) | 底模 f0G40k
>
> ⭐ **定稿男声之一** (6 音色, 用户耳测选定)。是 [§11 音色旋钮](../../METHODOLOGY.md) 的低-toya 档。

---

## 一、语料
- camus 12 曲 (40.8min) + toya 6 曲 (toya_07/18/16/15/17/11, 10.5min) = **20.4% toya (按时长)**。
- 两者均**复用 toya_camus_mix 已分离 stem** (§11.3), 全 ×1 无复制。
- toya 子集按"最短的几首凑到目标 20%"挑选 (脚本 `select_toya_subsets.py`)。
- prep: `v2_prep_all` (18 文件)。

## 二、训练
- **f0G40k** warm-start (男声); `-te200 -se20 -bs16`; e100-200 窗口中值谱 ensemble。
- 启动已确认 `All keys matched successfully`。

## 三、成品
- 男声源 `夏日.wav` (male_sources) → `FINAL_camus_toya20_v2.wav` (159.9s) → `camus-toya20-kotori_MIX.wav`。

## 四、⚠️ §11 客观测量的诚实注记 (定稿依据 = 耳测)
低-toya 档 (camus_toya10=10.4% / camus_toya20=20.4%) 想做"温和可控的音色旋钮", **客观 log-mel 距离未验证**:
- 距 toya: 纯 camus 2.90 → toya20 **2.72** → toya10 **3.22** (非单调! toya10 比纯 camus 还远)。
- 两点同时远离 toya 和纯 camus (不在连线上) → **不是干净插值, 各自跑偏**。
- **原因**: 语料太小 (15/18 文件, ~45-50min) + 200ep, 小数据训练的**模型间方差 (~2-3dB) 淹没了 10-20% toya 的微弱信号**。对比 c2 (43 文件) 就能干净插值 (距 toya 1.76)。
- **定稿依据是用户耳测** (方法论一贯: 客观是参考, 耳朵是最终裁判)。
- 若要真正可控的音色旋钮: **先扩充 camus 基础语料** (多下 camus 曲把基座做大做稳), 再在大语料上加低比例 toya, 信号才不会被训练噪声淹没。
