# LEGACY: 纯 Liyuu 版本 (Keke 混合训练之前)

> 日期: 2026-07-11 ~ 2026-07-12 | 模型: RVC v2 Liyuu-only
> **此文件为冻结版本，不再更新。当前版本见 KNOWHOW.md**

---

## 回退方法

```bash
# 1. 使用 legacy ckpt (不含 keke)
cd output/stage3_rvc_model/
# legacy ckpt: liyuu_G3624 ~ liyuu_G36240 (10个)

# 2. 使用 legacy index
cp liyuu_character_Flat.index character.index

# 3. 使用 legacy V3 (10 epoch)
# infer_smart_chorus_v3.py 中 epochs 改回:
# epochs = [24, 48, 72, 96, 120, 144, 168, 192, 216, 240]

# 4. 使用 legacy mix_multitrack.py (无 de-esser, -10dB Polish EQ)
# 从 git 恢复或手动改回:
#   VOCAL_GAIN_DB = 10.0
#   无 BGM_GAIN_DB
#   Polish EQ: -10dB @5-9kHz
#   无 de-esser
#   有 normalize

# 5. 使用 legacy 推理方式 (逐段加载模型)
# python infer_smart_chorus_v3.py
```

## Legacy 关键参数

| 参数 | Legacy 值 | 当前值 (KNOWHOW.md) |
|------|-----------|---------------------|
| 训练数据 | Liyuu 2.5h | Liyuu 2.5h + Keke 36.6min |
| Epochs | 240 | 300 |
| ckpt 数 | 10 (e24-e240) | 16 (e24-e300, 含5个keke-mix) |
| Index | `liyuu_character_Flat.index` | `liyuu_keke_character.index` |
| V3 epochs | 10 | 15 |
| Polish EQ | -10dB @5-9kHz | -3dB @5-9kHz |
| De-esser | 无 | -24dB thr, 6-10kHz |
| 滤波方式 | lfilter (非线性相位) | filtfilt (零相位) |
| 人声增益 | +10dB | +14dB |
| BGM 衰减 | 无 (0dB) | -6dB |
| 推理速度 | ~60s/段 (每次加载模型) | ~1s/段 (batch per model) |
| Normalize 步骤 | 有 (冗余) | 无 |

## Legacy 输出

```
output/stage5_final_liyuu_chorus/
  下等马_AI-Liyuu.wav              # V2 版本
  下等马_AI-Liyuu_post-processed.wav # V3 版本 (legacy 最终)
  final_cover_t+0.wav              # V3 basic merge
```

## Legacy 核心脚本

```
infer_smart_chorus_v3.py   # 10 epoch version
mix_multitrack.py          # pre-deesser, -10dB Polish EQ
train_liyuu_ext4.py        # EPOCHS=240
fix_new_ckpts.py           # fix liyuu_G* only
```
