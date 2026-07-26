# Honoka RVC 翻唱流水线 — 实战指南

> 📖 **跨音色统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。**
> 本文档只记 honoka 特有的内容 (亮嗓女高音参数、呼吸修复、v2 训练+成品定案 §十一)。
>
> 日期: 2026-07-19 | 模型: RVC v2 | 角色: Honoka Kousaka (穂乃果) | GPU: RTX 3090 24GB
> 参考: [Liyuu KNOWHOW](../liyuu/KNOWHOW.md) — 大量方法论来自 Liyuu 项目实战
>
> ✅ **2026-07-25 推理侧优化完成** — 见 §九。
> ✅ **2026-07-26 v2 训练侧重做 + TITAN A/B + 成品定案完成** — 见 §十一。
> 成品: `stage5_rebuilt/FINAL_honoka_v2.wav` (TITAN 中值谱 ensemble + 呼吸静音, §11.4)。
> 方法论背景见 [otoya_sho_mix/KNOWHOW.md](../otoya_sho_mix/KNOWHOW.md) §6-7、§10。

---

## 一、项目概况

- **角色**: 高坂穂乃果 (Honoka Kousaka) — Love Live! μ's
- **训练数据**: 58首 Honoka solo 曲目, 原始 4.64h → 预处理后 ~4.38h
- **预处理**: demucs htdemucs → UVR-MDX-NET-Voc_FT → Reverb_HQ_By_FoxJoy → silence trim → LUFS -20
- **模型**: RVC v2, 40kHz, RMVPE F0, ContentVec 768-dim, FAISS Flat
- **训练配置**: EPOCHS=260 (160+100 continue), BATCH_SIZE=16, SAVE_EVERY=10 (cold start from f0G40k/f0D40k)

---

## 二、训练

### 训练参数 (实际执行)
```yaml
epochs: 260 (160 cold + 100 continue)
save_every: 10      # snapshots at e10/20/.../260
batch_size: 16
cold_start: true    # from f0G40k/f0D40k
```
> 第一轮 160 epochs, 2026-07-18, ~4h52m。最终 loss: disc=4.06, gen=3.54, mel=25.28, kl=1.06
> 第二轮 160→260 epochs, 2026-07-19, ~3h。最终 loss: disc=3.68, gen=4.07, mel=26.92, kl=1.06
> 3693 slices, 每 epoch ~234 steps。

### 可用 ckpt (共 20 对)
| Epoch | Step | 部署 |
|------|------|------|
| 70 | 16380 | ✅ |
| 80 | 18720 | ✅ |
| 90 | 21060 | ✅ |
| 100 | 23400 | ✅ |
| 110 | 25740 | ✅ |
| 120 | 28080 | ✅ |
| 130 | 30420 | ✅ |
| 140 | 32760 | ✅ |
| 150 | 35100 | ✅ |
| 160 | 37440 | ✅ |
| **170** | **39780** | ✅ **new** |
| **180** | **42120** | ✅ **new** |
| **190** | **44460** | ✅ **new** |
| **200** | **46800** | ✅ **new** |
| **210** | **49140** | ✅ **new** |
| **220** | **51480** | ✅ **new** |
| **230** | **53820** | ✅ **new** |
| **240** | **56160** | ✅ **new** |
| **250** | **58500** | ✅ **new** |
| **260** | **60840** | ✅ **new** |
> 全部 20 对 G_infer + index 已部署到 `models/`。每个 G_*.pth ~418MB, infer ~140MB, D_*.pth ~818MB。

### 数据规模演进
| 版本 | 歌曲数 | 时长 | 备注 |
|------|--------|------|------|
| v1 | 31 | 2.29h | 初始 |
| v2 | 45 | 3.51h | +14首 |
| **v3** | **58** | **~4.38h** | +12首 Honoka Mix, +1段说话(11.8min) |

### 预期时间 (3090)
| 阶段 | 预计 |
|------|------|
| preprocess (slice) | ~5 min |
| F0 extract (RMVPE) | ~7 min |
| HuBERT features | ~6 min |
| train (160ep) | ~5h |
| train continue (+100ep) | ~3h |
| build index | ~3 min |
| **总计 (160ep)** | **~90-115 min** |
| **总计 (260ep)** | **~5h cold + ~3h continue** |

---

## 三、推理流水线 — 复用 Liyuu 全流程

### 3.1 源素材预处理 (v4 最佳)
```bash
# 对推理用原曲做 V4 级别人声净化
python codes/vocal_prep_v4.py --input <song> --outdir output/v4_<name>/
```
> **V4 > V3**: dereverb (anvuew less-aggressive) + 3×Roformer min-|x| 保守交集。
> 和声几乎消失，主唱完整保留。比训练数据的 V3 预处理更强。

### 3.2 分段 → RVC 推理 → V3 Ensemble
```
原曲人声 → silence split → 逐段 RVC (全部 ckpt) → V3 打分 → Top-3 softmax 加权
```

**V3 参数** (Liyuu 调优):
```python
ENSEMBLE_TOP_K = 3
ENSEMBLE_TEMP = 0.1
P2_LAMBDA = 1.0          # 邻段一致性惩罚
P3_POOL_SIZE = 5          # per-track top-5 ckpt 收窄
P5_MIN_HIT_RATE = 0.05    # 剔除 hit<5% 的弱 ckpt
```

### 3.3 RVC 推理参数
```python
vc.vc_single(
    sid=0, f0_up_key=0, f0_file=None,
    f0_method="rmvpe",
    index_rate=0.50,      # 检索混合比
    filter_radius=7,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33,         # 保护呼吸声
)
```

### 3.4 混音 (M1-M7)
| 轨 | 角色 | 电平 | 声像 | HP |
|----|------|------|------|-----|
| T1 | 主唱 | 0dB | 中央 | 60Hz |
| T2 | 主唱 | -3dB | 中央 | 60Hz |
| T3 | 低和声 | -8dB | ← 40% | 80Hz |
| T4 | 中和声 | -9dB | → 40% | 100Hz |
| T5 | 高和声 | -8dB | 中央 | 200Hz |

### 3.5 后处理链路 (P 链 — 复用 Liyuu v2 优化版)
```
Match EQ 30%  →  Polish EQ     →  De-esser           →  Micro-chorus  →  混 BGM
 中频 -2.6dB    5-9k -3dB       6-10k, -24dB thr       2ms/0.8ms/7%    vocal +14dB
                zero-phase FIR   4:1 ratio                              BGM -6dB
```
- **Match EQ**: 匹配原曲人声频谱 (强度 30%), 减少 AI 味。不要超 40% (引入金属感)
- **Polish EQ**: 5-9kHz -3dB 零相位 FIR (511 tap), 压制高频毛刺但保留穿透力。不要超 -6dB
- **De-esser**: 动态齿音控制 (6-10kHz 检测, -24dB 阈值, 4:1 压缩比, 2ms attack / 20ms release)
- **Micro-chorus**: 打破频谱过于均匀的问题, 减少沙感/电音感。wet 不要超 10%
- **BGM**: 纯净 off-vocal, 非 Demucs 分离版
- **人声/BGM 平衡**: vocal +14dB, BGM -6dB → 净 20dB 人声突出

### 3.6 混音链 (M 链)
对于单轨翻唱 (非 Liyuu 5轨和声):
```
HP 100Hz → Pan 中央 → Gain staging (vocal +14dB, BGM -6dB)
```
多轨时参考 §3.4 的 M1-M7 分轨配置。

---

## 四、Liyuu 项目核心经验（直接复用）

### 4.1 训练
- **ext4 > 9p**: 训练数据放 `~/rvc_training/` (ext4), 60-90s/epoch; `/mnt/c/` (9p) 5-6min/epoch
- **cold start**: 从 f0G40k/f0D40k 开始，不要 warm-start 旧 ckpt
- **ckpt 命名**: 用 `global_step` 命名，禁用 `if_latest` 分支 (会覆盖所有 ckpt 为 G_2333333)
- **⚠️ ckpt 格式转换**: 训练保存的格式 `{model, iteration, optimizer, learning_rate}` 与推理所需格式 `{config, weight, f0, version}` 不同！必须转换才能用于 infer_cli.py。转换方法见 §4.6

### 4.2 分段推理
- **静音分割**: `min_silence_s=0.4, min_segment_s=2.0, thresh_db=-35`
- **静音段跳过**: RMS < -40dB
- **打分**: `HNR(30%) + Flat(40%) + RMSv(30%)`
- **批量推理**: 按模型批量 (每个模型加载1次→跑全部段), 比逐段加载快 ~40x

### 4.3 Ensemble
- **TEMP=0.1**: 权重接近均匀 (~33%), 平滑单 ckpt 偶发瑕疵
- **不需要重新推理**: 所有 ckpt 输出缓存在 `_tmp_segments/`

### 4.4 Offset 对齐
- **同源同歌**: PAD=0 (v4 人声和 BGM 天然对齐)
- **异源**: 能量包络互相关找精确 offset
- **人耳优先**: 算法结果必须经人耳确认

### 4.5 环境
- **torch**: 2.5.1+cu121 可行, 需 numpy<2 (faiss 兼容)
- **WSL**: 用 `C:\Windows\System32\wsl.exe` 完整路径
- **Python**: 写 .py 文件再运行, 避免 bash -c 嵌套引号

### 4.6 ⚠️ ckpt 格式转换 (训练→推理)
训练保存格式 (`train.py`):
```python
{"model": state_dict, "iteration": int, "optimizer": ..., "learning_rate": float}
```
推理需要格式 (`infer_cli.py`):
```python
{"config": [...18 elems...], "weight": state_dict, "f0": 1, "version": "v2"}
```
config 列表 = `[n_mel_channels, segment_size, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, spk_embed_dim, gin_channels, sampling_rate]`

转换脚本: `python _convert_ckpt.py` (读 config.json + G_*.pth → 写 G_*_infer.pth)

---

## 五、质量调优备忘

| 问题 | 解法 |
|------|------|
| 人声被埋 | 提 vocal bus gain / 降 BGM |
| 声音虚 | Polish EQ 别超 -6dB |
| 齿音刺耳 | De-esser -24dB thr |
| 电音感 | Micro-chorus 2ms/0.8ms/7% |
| ckpt 选择单一 | 降低 TEMP 让权重更均匀 |
| 段间跳跃感 | P2 consistency penalty |

---

## 六、当前状态

> 全流程已完成并定案；详情见 §九 (推理侧) 与 §十一 (v2 训练侧 + 成品)。
> 下面 v1 勾选表为历史记录，v2 收尾以 §11.4 为准。

- [x] v1 训练 (2026-07-18, 160 epochs, ~4h52m)
- [x] v2 训练侧重做 + TITAN A/B (2026-07-26, 200ep/se20, §十一)
- [x] flat 全量索引 (537,174 向量, 仅入选切片)
- [x] 推理: 东京夏日相会 (tokyo_summer_v3)
- [x] 中值谱 ensemble (6×e100-200 ckpt, §11.3)
- [x] P 链后处理 + 呼吸静音 (§11.4)
- [x] 成品定案: `FINAL_honoka_v2.wav` (TITAN 中值 + 呼吸静音, 用户 2026-07-26 定)

---

## 七、文件结构
```
characters/honoka/
├── config/           ← 40k.json + training.yaml
├── data/             ← 训练数据 + 推理输入
│   ├── v3_corpus/    ← Honoka 训练语料 (58首, 4.38h)
│   └── inference/    ← 推理项目 (每首歌一个子目录)
├── models/           ← 训练产出 ckpt + index
├── output/           ← 推理成品
└── logs/             ← 训练日志
```

---

## 八、推理工作流

### 8.1 推理源文件位置（重要！不要乱找）
```
characters/honoka/data/inference/<song_name>/
├── vocals.wav           ← v4 干声 (44.1kHz stereo s16)
└── accompaniment.wav    ← v4 伴奏 (44.1kHz stereo s16)
```
> 来源: `v3_corpus/stage1/song_*.wav` → 改名拷贝到 `inference/<song_name>/`
> v4 处理: demucs htdemucs_ft 分离 → UVR-MDX-NET-Voc_FT → Reverb_HQ_By_FoxJoy

### 8.2 RVC 推理前准备
```bash
# 干声必须转成 40kHz mono, 与训练格式一致
ffmpeg -i vocals.wav -ar 40000 -ac 1 -c:a pcm_s16le vocals_40k_mono.wav
```
> 训练格式: 40kHz, mono, s16le
> 不要直接用 44.1kHz stereo 扔进 RVC

### 8.3 完整推理流水线 (分段→Ensemble→P+M)
```
vocals_40k_mono.wav
  │
  ├─ 1. 静音分割 (min_silence=0.4s, min_seg=2.0s, thresh=-35dB)
  │      → N 个片段
  │
  ├─ 2. 转换全部 ckpt 为推理格式 (§4.6)
  │
  ├─ 3. 批量推理: 每 ckpt 加载1次→跑全部 N 段 (按模型外循环, 快 ~40x)
  │      → N×10 个缓存文件 in `_tmp_segments/`
  │
  ├─ 4. V3 打分: HNR(30%) + Flat(40%) + RMSv(30%)
  │      → 每段每个 ckpt 的分数
  │
  ├─ 5. Top-3 Ensemble: softmax(score/TEMP) 加权, TEMP=0.1
  │      → N 个 ensemble 片段
  │
  ├─ 6. 拼接: crossfade 10ms + RMS 归一化
  │      → 完整人声
  │
  ├─ 7. M 链: HP 100Hz → Pan 中央 → Gain staging
  │
  ├─ 8. P 链: Match EQ 30% → Polish EQ -3dB → De-esser → Micro-chorus 7%
  │
  └─ 9. 混 BGM: vocal +14dB, BGM -6dB → 最终成品
```

### 8.4 RVC 推理参数 (单次调用)
```python
vc.vc_single(
    sid=0, f0_up_key=0, f0_method="rmvpe",
    index_rate=0.50, filter_radius=7,
    resample_sr=0, rms_mix_rate=0.25, protect=0.33,
)
```
> 模型: `models/G_*_infer.pth` (需先转换格式, 见 §4.6)
> 索引: `models/added_IVF256_Flat_mi_baseline_src_feat.index`

### 8.5 输出目录约定
```
characters/honoka/output/<song_name>/
├── vocals_40k_mono.wav    ← 转换后输入
├── segments/              ← 静音分割结果
├── _tmp_segments/         ← 每段×每ckpt 缓存 (可清理)
├── ensemble/              ← ensemble 拼接后完整人声
├── mixed/                 ← 混 BGM 后 (P+M 前)
└── final/                 ← 最终成品
```
全部在 `characters/honoka/` 内, 不污染全局 `output/`。

---

## 九、推理侧优化 (2026-07-25) — honoka 专属参数, 勿盲搬 otoya

> 全流程与 otoya_sho_mix §6-7 相同 (伪影测量→flat 索引→参数扫描→
> 单 ckpt 选优→修复链→验收), 但**修复参数按她的测量重新定**。

### 9.1 她与 otoya 的关键差异 (全部实测)

| 维度 | otoya_sho_mix | **honoka** |
|---|---|---|
| 源特性 | 男声, rolloff95 ~9k | 女高音, **真实空气感到 15.5k** (rolloff 14.5k) |
| vocoder 梳齿位置 | >10k | **>16k** (17k 处 +20.9dB) |
| 毛刺带 | 2-4k | **2-5k** (+7.6~+9.6dB) |
| 缺失带 (勿削!) | 5-6k | **6-8k** (−4~−11dB) 与 12-15.5k |
| LPF 截止 | 12k | **15.8k** (12k 会砍掉 3kHz 真实内容) |
| tame | 3.8k −4dB + demud −2.5dB | **3.2k −5dB, 无 demud** (低频干净) |
| 索引敏感度 | 高 (timbre 差 3+) | **低 (差 0.8, 4.38h 语料训得足)** → flat + ir=0.5 |
| F0 verify | 主歌 OOD → +12 | **EDGE** (508Hz vs p75=452) — 无法 transpose, 训练侧补高音语料 |
| 短段丢失 | 1 处 (已恢复) | 无 |

她的 artifact_score 额外加了 **air_loss 罚项** (12-15.5k 缺失), 因为选 ckpt
时必须保住她源里的真实高频。

### 9.2 中值谱 ensemble (新方法, 两全)

旧 top-3 波形平均: 音色平滑✓ 但梳齿取并集✗。新方法 `_median_ensemble.py`:
**top-5 (按 artifact score) 的 STFT 幅度谱逐 bin 取中值** — 每根梳齿只存在
于一个 ckpt, 中值天然剔除 — 相位取最佳单 ckpt, 再过修复链。
实测 3-5k 残余毛刺从 +3.4dB (纯 EQ) 降到 **+0.1dB**, 其余持平。

### 9.3 验收 (tokyo_summer_v3, seg 全长 24.6s)

| 指标 | 源 | 旧 ensemble | 单ckpt flat 重建 | **中值ens (最终)** |
|---|---:|---:|---:|---:|
| >15.5k 窄带峰 | 0 | **57** | 0 | **1** |
| 16-20k 垃圾 | −47.0 | −45.2 | −87.8 | **−87.9** |
| 3-5k 毛刺 (源 −17.1) | — | −10.6 | −13.7 | **−17.0** |
| 动态范围 (源 4.24) | — | 6.44 | 4.07 | **4.14** |
| 音色距离(→语料) | 84.8 | **36.1** | 41.1 | 41.3 |

诚实注记: **音色 proxy 一项旧 ensemble 仍最优** (36.1) — 波形平均对 MFCC
均值有平滑红利; 但它带 57 根可闻啸叫和 +6.5dB 毛刺。以耳测裁决;
若耳朵也偏好旧版音色, 可对旧 ensemble 单独跑修复链折中。
12-15.5k 缺失 (−30 vs 源 −13) 是模型能力上限, 推理侧无法修复,
留给训练侧 (48k 模型或补亮嗓语料)。

### 9.5 呼吸啸叫修复 (2026-07-25, 用户耳测反馈驱动)

**症状**: 呼吸时有啸叫, 音色 OK。
**根因**: RMVPE 在呼吸段幻觉出 F0 → vocoder 沿假音高合成移动的哨音。
**指标盲区教训**: 此前所有谱指标用响帧平均, 呼吸帧伪影完全不可见。
新指标: 逐帧测 0.8-10k 最强 bin 相对该帧中值的凸起 (>18dB = 啸叫帧)。

**修法 (结构性, 非 EQ)**: `_fix_breaths.py` — 呼吸近乎与歌手无关的噪声,
直接把源的真实呼吸电平匹配后替换回去 (源侧检测: 低能量 1.5%-25% × pyin
无声; ±30ms pad; 20ms crossfade)。演唱帧按构造零改动。

| | 啸叫帧 (7 处呼吸区, 172 帧) |
|---|---:|
| 源 (基线) | 70 |
| 旧 ensemble | 133 |
| 中值 ens 未修 | 107 |
| **+breathfix** | **74 (超源仅 +4)** |

注意: 2.65-3.08s 与 18.43-18.71s 被 pyin 判为**有声**尾音, 未替换 —
若耳测这两处仍有啸叫, 属"安静有声尾音上的伪谐波", 需另行处理。

### 9.4 产物与脚本

```
output/tokyo_summer_v3/stage5_rebuilt/
├── vocals_rebuilt_flat_medens_breathfix.wav  ⭐ 最终推荐
├── vocals_rebuilt_flat_medens.wav            中值ens (呼吸未修)
├── vocals_rebuilt_flat.wav                   单 ckpt (e130) 版
├── vocals_rebuilt_ivf.wav                    旧索引重建 (对照)
└── rebuild_report.json

_build_flat_index.py   # 652,863 向量全量 Flat (2.01GB, 12s)
_param_sweep.py        # index/ir 扫描 (结论: 她不敏感, flat+ir0.5)
_infer_flat.py         # 20 ckpt 批量重推 (flat, ~40s/ckpt)
_rebuild_v4.py         # honoka 参数版重建 (LPF 15.8k / tame 3.2k / air_loss 罚项)
_median_ensemble.py    # 中值谱 ensemble (top-5)
_fix_breaths.py        # 呼吸段源替换 (啸叫修复, 任意 build 通用)
models/corpus_f0_stats.json  # F0 门禁统计 (tools/verify_source_f0.py)
```

---

## 十、honoka v2 训练侧重做 — 方案 (2026-07-25, 待执行)

> 沿用 otoya_sho_mix v2 的修正配方 (加载 pretrain + v4.5 干净语料 +
> `-te 200 -se 20` + e100-200 选 ckpt, 详见 [otoya KNOWHOW §十](../otoya_sho_mix/KNOWHOW.md#十))。
> honoka 独有的问题: **数据量近 2 倍, 需先质量打分择优削到与 otoya 相当。**

### 10.1 数据现状与目标

| | honoka | otoya_sho_mix (参照) |
|---|---|---|
| raw 曲目 | **58 首, 278min** (整曲, 需分离) | 26 轨, v45 后 96.5min unique |
| v2 训练量目标 | **~180min (保留 ~65%, 弃最差 ~35%)** | 146min (含 otoya 2× 加权) |

> **用户定 (2026-07-25): 适量留更多 → 目标 ~180min**。比 otoya 的 146min 多
> 约 25%, 保留 honoka 更宽的音域/音色覆盖, 同时靠质量打分弃掉最差 ~35% 的糙片
> (纯静音/噪声/F0 失败)。warm-start 下多留干净数据无害, 择优只去糙不伤覆盖。

### 10.2 新增环节: 切片质量打分与择优 (`_score_slices.py`)

v1 喂**单个 concat 文件** → 所有切片 `0_*`, 丢曲目身份 (隐藏缺陷)。
v2 喂 **58 个独立文件** → 切片名 `{曲号}_{片号}` (preprocess.py:114 按文件枚举),
曲目身份可恢复, 才能做"按曲多样性"。

**流程位置**: v4.5 prep → RVC preprocess 切片 (0_gt_wavs) → **打分择优** →
F0/特征 → filelist 只含选中片 → 训练。

**每片质量分** (越高越留):
```
quality = 0.40·HNR_norm        # 谐波清晰度 (harmonic/percussive 能量比)
        + 0.25·voiced_ratio    # 有音高帧占比 (可用发声内容)
        + 0.20·(1 − flatness)  # 谱平坦度低 = 像人声不像噪声
        + 0.15·(1 − silence_frac)  # 非静音占比
```
**硬剔除** (明确的糙片): voiced_ratio<0.15 (基本静音/纯呼吸/噪声) ·
HNR 低于地板 · F0 不稳 (>30% 帧跳八度 = 提取失败)。

**择优到 ~146min, 两个多样性护栏**:
1. **按曲配额**: 单曲贡献 ≤ 其比例份额的 1.4× (防某首干净曲独大) — 靠切片名曲号实现。
2. **音高覆盖**: 按中位 F0 分箱, 跨箱按比例选 → 保住 honoka 女高音全音域
   (她 F0 median 508Hz、范围宽, 不能只留好唱的中音区)。

产物: `selected_slices.txt` (选中片 basename) + `slice_score_report.json`
(打分分布、各曲入选数、音高覆盖)。

### 10.3 执行步骤 (脚本待建)

```
1. _prep_corpus_v45.py  (改指向 honoka_raw, 单音色无 2:1)
      → data/honoka_v45_corpus/  (58 干净单曲)         ~55-65min (GPU-bound)¹
2. RVC preprocess (58 文件切片)  → 0_gt_wavs/{曲}_{片}   ~10min
3. _score_slices.py --target-min 146                     ~5min
      → selected_slices.txt
4. F0 (RMVPE) + ContentVec 特征 (全片)                   ~25min
5. v2_step3b_filelist.py --keep-list selected_slices.txt
6. v2_step4_train.sh (改 exp=honoka_v2, -te 200 -se 20)  ~3-4h
7. 部署 + flat 索引 + 单中段 ckpt 重建 (同 otoya)         ~15min
```

### 10.4 配置 (与 otoya v2 完全一致)

- pretrain: `f0G40k + f0D40k` (启动确认 `All keys matched`)
- `-te 200 -se 20 -bs 16 -sr 40k -f0 1 -v v2`, 选 e100-200 最佳
- flat 全量索引; 推理侧单中段 ckpt, 不需 ensemble/中值谱/修复链
  (§九那套是 v1 欠训模型的补救, v2 训对后应不再需要)

### 10.5 用户已确认 (2026-07-25)

- [x] 目标训练量: **~180min** (适量留更多)
- [x] onnxruntime-gpu 已修 (正确性修复, 非提速; 分离本就 GPU-bound, 详见
  [../../METHODOLOGY.md](../../METHODOLOGY.md) §8); 磁盘已清 43G
- 待建: `_score_slices.py` (--target-min 180) + honoka 版 v2 step 脚本, 按 §10.3 跑。
- 统一流程见 [../../METHODOLOGY.md](../../METHODOLOGY.md) (跨音色方法论)。

---

## 十一、honoka v2 执行 + TITAN 底模 A/B (2026-07-26)

### 11.1 执行概况

按 §十 方案跑完, 外加一个 **TITAN 社区底模 A/B 实验臂** (用户批准, 探索
"官方 pretrain 是否是配方里唯一还停在默认的组件")。

- **语料**: 58 raw 轨 → v4.5 重制 (脏度 −52.9dB, 呼吸密度 15-22/min, 82.8min CPU)
- **切片择优** (`_score_slices.py`): 3666 切片 → 硬剔 45 → 择优 **2945 片 = 180min**,
  **58/58 曲全覆盖** (按曲配额 1.4× + F0 六分箱都生效)
- **双臂训练** (仅 `-pg/-pd` 不同, 特征/filelist/超参全同 → 归因干净):
  - baseline: 官方 f0G40k/f0D40k
  - titan: `assets/pretrained_v2_titan/{G,D}-f040k-TITAN.pth` (blaise-tk/TITAN,
    11.15h Expresso 继续预训练; 与 RVC v2 40k 架构兼容, `All keys matched`)
  - 各 200ep/se20/~4.7h, mel 收敛 ~20.5 (亮嗓宽音域, 地板天然高于 otoya 的 18.6)

### 11.2 A/B 客观结果 (`_ab_eval_v2.py`, seg_000)

核心问题: **两种底模各修回多少 honoka 的 12-15.5k 空气感缺陷** (v1 未解)。

| 版本 | peaks | **air12** | rolloff | 音色 | breath(raw) |
|---|---:|---:|---:|---:|---:|
| 源 | 0 | −13.1 | 14559 | 82.2 | 71/190 |
| v1 冠军 | 1 | −30.2 | 10579 | 45.7 | 93/190 |
| baseline 最优 e202 | 0 | −28.6 | 10957 | 46.1 | 174 |
| **TITAN 最优 e182** | 0 | **−26.5** | **11999** | 46.0 | 182 |
| TITAN e121 (音色最优) | 0 | −28.1 | 11703 | **43.2** | 185 |

**结论 (客观层面)**:
1. **TITAN 一致地更亮**——air12 −26.5 vs baseline −28.6 vs v1 −30.2; rolloff
   12k vs 11k vs 10.6k。正好补 honoka 最弱的空气感, 是社区底模在**这个音色**上的
   真实(虽不大)增益。
2. 但缺陷**只部分修复**——仍比源 −13.1 低 ~13dB, 是模型能力上限, TITAN 也没填平。
3. 两臂都干净 (peaks 0)。
4. **breath 那列两臂都 ~180, 不是差异项**——裸推理未加 breath-fix (v1 的 93 是
   加了 fix 的); 选定后照跑 `_fix_breaths` 压下来。

**倾向**: TITAN 臂略胜 (空气感 + rolloff), 但差距小, **最终耳测定夺**。
TITAN 内部有取舍: e182 最亮 (air −26.5, 音色 46), e121 音色最优 (43.2, air −28.1)。

### 11.3 耳测发现: 单 ckpt 的"和声噪音", 中值谱 ensemble 解决 (最终定案)

**用户耳测反馈**: v2 单 ckpt 相比 v1 "背后像有个和声噪音", v1 反而"更实"。
**客观查实 (`chorus_diag.py`)**——用户耳朵是对的:

| 版本 | 谐波间噪声(300-4k平坦度) | 梳齿 peaks | 体量(300-1k) |
|---|---:|---:|---:|
| 源 | 0.0123 | 0 | −2.0 |
| v1 波形 ensemble | 0.0135 | **25** | −0.8 |
| **v2 单 ckpt** | **0.0214** (和声噪音) | 0 | −2.8 |
| v2 baseline 中值 | 0.0159 | 0 | −2.6 |
| **v2 titan 中值** | **0.0149** | **0** | −2.5 |

- **根因**: v1 是 3-ckpt **波形平均** → 谐波相位一致被增强, 谐波间非相干噪声互相
  抵消 → 干净且"实"; **单 ckpt 无此抵消 → 谐波间噪声裸露** = 和声噪音。
  (但 v1 的干净有代价: 25 个 >15.5k 电流梳齿。)
- **解**: **中值谱 ensemble** (top-k v2 ckpt 幅度谱逐 bin 中值 + 最佳相位) —— 谐波间
  噪声降到 0.0149 (近源), **且 peaks 仍 0** (中值剔除梳齿, 不像波形平均叠加)。两全。
- **可从 stage8 逐 ckpt 输出直接算, 不占 GPU** (`median_from_stage8.py`)。
- "v1 更实" = v1 比源多 +1.2dB 低-中暖色 (染色非保真); 需要的话加 <600Hz +2dB 低搁架。

> **方法论修正**: §十 "训对后单 ckpt = 逐段选优/ensemble" 对 otoya (混合男声) 成立,
> 但**亮嗓女高音是例外**——单 ckpt 会暴露谐波间噪声, **中值谱 ensemble 才是真最优**。
> 已写入 [METHODOLOGY §7](../../METHODOLOGY.md)。

### 11.4 最终成品 (用户 2026-07-26 定案)

**`stage5_rebuilt/FINAL_honoka_v2.wav`** = **TITAN 中值谱 ensemble + 呼吸静音**
(`_AB.../8m_titan_MEDIAN_breath-muted.wav`)。
- 底模: TITAN (亮嗓空气感略胜 baseline)
- 重建: 6 个 e100-200 ckpt 的中值谱 ensemble (无和声噪音, 无梳齿)
- 呼吸: 全静音 (用户口味; 裸干声上句尾呼吸偏突出, 混 BGM 后本会被盖)

试听全集 (供复核): `output/tokyo_summer_v3/_AB_v2_baseline_vs_titan/`
0=v1冠军 1=单ckpt(有和声噪音) 2/3=单ckpt亮/音色 4/5/6=呼吸-8/-18/静音
7(w)=baseline中值(暖) 8(w/m)=titan中值(暖/静音) 9=源。

### 11.5 待用户耳测 (原始 A/B 记录, 已被 §11.4 取代)

底模 A/B 结论仍成立 (TITAN 略亮), 但成品形态从"单 ckpt"改为"中值谱 ensemble"。

### 11.6 产物与脚本

```
models_v2/        baseline 10 ckpt infer + flat 索引 (537,174 向量, 仅入选切片)
models_v2_titan/  TITAN 10 ckpt infer
_prep_corpus_v45.py / _score_slices.py / v2_step{0,3b,4,6}*.sh /
_build_flat_index_v2.py / _ab_eval_v2.py
tools/download_titan_pretrain.sh   (TITAN 底模下载+校验)
```

> **对后续 4 个音色的启示**: TITAN 在亮嗓女高音上有小增益; 是否值得作默认底模,
> 取决于耳测结论。若确认 TITAN 胜, 后续音色可直接 `-pg/-pd` 指向 TITAN 起训,
> 零额外成本。见 [../../METHODOLOGY.md](../../METHODOLOGY.md) §6。
