# otoya_tsukasa_mix — 混合音色 RVC v2 (otoya : tsukasa 2:1)

> 📖 **统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。** 本文档只记特有内容。
>
> 日期: 2026-07-31 | RVC v2, 40kHz | **otoya ×2 + tsukasa ×1** | 底模: 官方 f0G40k (男声)
>
> 动机: 用 **tsukasa 替代 otoya_sho 里的 sho**, 试一个更不同的男声候选。
> (与 toya/camus 相近不同, tsukasa 与 otoya 差异够大, 2:1 混音**明显改变**了音色。)

---

## 一、语料 (加权 otoya:tsukasa ≈ 2:1, 与 otoya_sho 基线一致)
| 声部 | 曲数 | 时长 | 混合 |
|---|---|---|---|
| otoya (一十木音也) | 13 | 99min | ×2 (**复用 otoya_sho_mix 已分离 otoya**, §11.3) |
| tsukasa (天马司 / Project SEKAI) | 25 | 53.6min 人声 | ×1 (本轮新分离) |

- 加权占比 otoya 65% / tsukasa 35% ≈ 2:1 (otoya_sho 基线为 68/32)。
- tsukasa 25 曲含 6 首"短版/完整版或不同 take"的第二版本 —— 内容核实为**不同录音, 非冗余**, 全保留。
- prep: `v2_prep_all` (51 文件)。

## 二、训练
- **f0G40k/f0D40k** warm-start; `-te200 -se20 -bs16`; e100-200 窗口中值谱 ensemble。
- 启动已确认 `All keys matched successfully`。

## 三、成品与发现
- 男声源 `夏日.wav` (male_sources) → 中值谱逐段组装 → `FINAL_otoya_tsukasa_v2.wav` (159.9s)。
- 混音: **`otoya-tsukasa-kotori_MIX.wav`** (24.5s 对齐样本, −12 LUFS)。
- **距 otoya 基线 = 3.16dB log-mel (>3dB 可闻阈值)** → tsukasa 成功替代 sho 并**明显改变**了音色。
  亮度 2184Hz (略亮于 otoya 2167)。
- 对照 [METHODOLOGY §11](../../METHODOLOGY.md): 与 toya/camus 那种"相近歌手混不出区别"相反 ——
  **换一个真正远离目标的歌手 (源差够大), 2:1 混音才买得到可闻的音色变化。**
