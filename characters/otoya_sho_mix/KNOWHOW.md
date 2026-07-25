# otoya_sho_mix — 混合音色 RVC v2

> 📖 **跨音色统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。**
> 本文档只记 otoya_sho_mix 特有的内容 (2:1 混合、音域、成品、历史)。
>
> 日期: 2026-07-23 | 模型: RVC v2 | 混合: 音也(Otoya) + 翔(Sho) 2:1 | GPU: RTX 3090
>
> 🔥 **2026-07-25 训练侧重做 (otoya_sho_mix_v2), 电流声从物理根源根治 — 见 §十。**
> 最终成品候选在 `output/tokyo_summer_v3/_AB_v1_vs_v2/`, 待耳测定夺。
> 一句话结论: v1 模型漏加载 pretrain (随机初始化) 是所有 vocoder 伪影总根源;
> 加载 pretrain + 干净语料重训后, **零后处理即 0 窄带峰** (v1 需全套修复链才能
> 从 143 峰压到 0), 且主歌 OOD 消失, 不再需要 +12 升八度 hack。

---

## 一、项目概况

- **音也 (Otoya)**: 一十木音也 CV:寺島拓篤, 13首独唱, ~50.5min
- **翔 (Sho)**: 来栖翔 CV:下野紘, 13首独唱, ~46.5min
- **混合比例**: otoya ×2 + sho ×1 ≈ 2.17:1 (偏向音也)
- **训练语料**: 147min, 675MB (otoya 101min + sho 46.5min)
- **预处理**: Stage1 demucs → Stage2 dereverb/trim/LUFS-20 (与 Honoka 同管线)
- **模型**: RVC v2, 40kHz, RMVPE F0, ContentVec 768-dim

---

## 二、训练配置

```yaml
epochs: 200
save_every: 10      # snapshots at e10/20/.../200
batch_size: 16
cold_start: true    # from f0G40k/f0D40k
```
> 实际训练 2026-07-23, ~12h on RTX 3090。1988 slices, 每 epoch ~126 steps。
> 最终 loss: disc=3.54, gen=3.83, mel=27.09, kl=0.75

### 可用 ckpt (共 14 对, e70-200)
| Epoch | Step | 部署 |
|------|------|:---:|
| 70 | 8820 | ✅ |
| 80 | 10080 | ✅ |
| 90 | 11340 | ✅ |
| 100 | 12600 | ✅ |
| 110 | 13860 | ✅ |
| 120 | 15120 | ✅ |
| 130 | 16380 | ✅ |
| 140 | 17640 | ✅ |
| 150 | 18900 | ✅ |
| 160 | 20160 | ✅ |
| 170 | 21420 | ✅ |
| 180 | 22680 | ✅ |
| 190 | 23940 | ✅ |
| 200 | 25200 | ✅ |
> 全部 14 对 G_infer + index 已部署到 `models/`。

## 三、数据来源

| 角色 | 歌曲数 | 时长 | 混合占比 |
|------|--------|------|----------|
| 音也 (otoya) | 13 | 50.5min | ~68.5% (×2) |
| 翔 (sho) | 13 | 46.5min | ~31.5% (×1) |

## 四、训练步骤

```bash
# 一次性: setup
bash characters/otoya_sho_mix/step0_setup.sh

# 依次执行:
bash characters/otoya_sho_mix/step1_preprocess.sh
bash characters/otoya_sho_mix/step2_f0.sh
bash characters/otoya_sho_mix/step3_features.sh

# 训练 (后台):
setsid bash characters/otoya_sho_mix/step4_train.sh > characters/otoya_sho_mix/logs/train.log 2>&1 &

# 监控:
tail -f characters/otoya_sho_mix/logs/train.log

# 训练完成后建索引:
bash characters/otoya_sho_mix/step5_index.sh
```

## 五、预期时间 (RTX 3090)

| 阶段 | 预计 |
|------|------|
| preprocess (slice) | ~10 min |
| F0 extract (RMVPE) | ~15 min |
| HuBERT features | ~12 min |
| train (200ep) | ~12h |
| build index | ~6 min |
| **总计** | **~12.5h** |

---

## 六、"电流声"诊断与修复 (2026-07-25, tokyo_summer_v3)

> 全部结论来自客观测量 (buzz 指标 = 12-20kHz 能量 − 1-3kHz 人声能量, dB)。
> 分析/修复脚本: `_rebuild_v4.py`, `_transpose_probe.py`, `_infer_transposed.py`。

### 6.1 三个成因 (按贡献排序)

1. **>10kHz vocoder 谐波梳**。源人声 rolloff95 只有 ~9kHz, 10-20k 窄带峰 0 个;
   RVC 输出有 40-100 个 (~100Hz 间距的梳状啸叫)。12k 以上全是模型伪造,
   直接 LPF@12k 零损失移除。
2. **Top-3 softmax ensemble 帮倒忙**。各 ckpt 梳齿频率不同 → 平均=取并集
   (峰 98→197 个); 真人声相位不相干 → 平均=相消 (1-3k −2.2dB)。
   **8/8 段 ensemble 都比最佳单 ckpt 差 2.9-7.3dB**。honoka 同病 (−22.7 vs −27.2)。
   旧打分公式 (HNR/flat/RMSv) 无高频项, 选不出干净 ckpt。
3. **低音主歌掉出训练分布**。语料 F0: median 248Hz / p05 173Hz (otoya、sho 都是
   高音系, 唱歌音域远高于说话)。这首歌 0-48s 主歌 ~127Hz, 低了整整一个八度,
   MFCC 距离 86 (副歌段只有 31)。低音段最佳 buzz 比副歌段差 12dB。
   **+12 移调后 buzz 改善 8.7dB, 回到源信号水平**; 副歌段 key+0 本来就最优。

### 6.2 修复链 (_rebuild_v4.py, 无需 GPU, 全部吃 stage3_cache)

```
逐段: artifact 打分选单 ckpt (罚 excess_hf/峰数/低频糊)
  → tame EQ (3.8kHz −4dB bell 去刺 + <300Hz −2.5dB 去糊)
  → LPF 12kHz (零相位 FIR 2047 tap, filtfilt)
  → 逐段 RMS 对齐源段自身 (恢复动态, 旧代码错用全局 RMS)
  → 10ms equal-power fade → 按 librosa split 复算的精确位置摆放
```

⚠️ **match EQ 教训**: 把 >5k 包络以 0.85 强度拉向源频谱会过修
(rolloff95 9017→5595Hz, 明显变闷)。LPF 单独就能清零窄带峰, EQ 只留轻量 tame。

### 6.3 实测结果 (整曲)

| | 12-20k vs 人声核心 | 10-20k 窄带峰 | rolloff95 |
|---|---:|---:|---:|
| 源 | −34.0 | 0 | 9017 |
| 旧 ensemble | **−25.2** | **143** | 10486 |
| 重建 (t12verse) | **−75.7** | **0** | 8933 |

主歌区间 (0-43s) 8-12k: 旧 −16.4 / key0 重建 −14.7 / **+12 重建 −20.8** (源 −22.6)。

### 6.4 产物

```
output/tokyo_summer_v3/stage5_rebuilt/
├── vocals_rebuilt.wav           # 全段 key+0
├── vocals_rebuilt_t12verse.wav  # ⭐ seg 0-2 用 +12 (主歌高八度), 3+ 用 0
├── variant_A_raw_best.wav       # 只换单 ckpt, 无修复 (A/B 用)
├── variant_B_lpf_only.wav       # + LPF
├── variant_C_lpf_tame.wav       # + tame EQ (= 默认链)
├── variant_D_lpf_tame_match.wav # + match EQ (偏闷, 不推荐)
└── rebuild_report.json
output/tokyo_summer_v3/stage3_cache_key+12/   # seg 0-2 × 14 ckpt 移调缓存
output/tokyo_summer_v3/stage6_transpose_probe/ # transpose 扫描原始输出
```

### 6.5 后续注意

- **给 honoka 也跑 _rebuild_v4** (改路径即可): 她的 ensemble 同样比最佳单 ckpt 差 4.6dB。
- ~~index nprobe=1 + IVF256 检索面太窄~~ → 已修, 见 §7。
- 未来选歌: 源 F0 median 落在语料 p25-p75 (217-295Hz) 内的段落质量最好;
  低于 ~170Hz 的段考虑 +12 或换低音系模型。→ 已工具化, 见 §7.3。

---

## 七、第二轮推理侧优化 (2026-07-25, flat 索引 + 全链, 不动训练)

### 7.1 Flat 全量索引 (关键修复)

旧部署索引 (`added_IVF256_...`, step5_index.sh 产物) 的真实构成:
**354,701 个特征先被 k-means 压成 10,000 个聚类中心, 再 IVF256 + nprobe=1
检索** — 每帧只搜 ~39 个向量, 检索层形同关闭。此前所有 stage3_cache
推理都在这个状态下跑。

修复: `_build_flat_index.py` → `models/flat_full_src_feat.index`
(354,701 全量向量, 精确检索, 1.09GB, 构建 6 秒)。
代价: 每段检索 ~10s (旧 0.4s), 整曲 14 ckpt 全量重推 ~35 min, 可接受。

### 7.2 参数扫描结论 (`_param_sweep.py`, 32 组客观测量)

| 结论 | 数据 |
|---|---|
| buzz 对 index/ir/protect 不敏感 | 各组差 ~1dB, 反正会被 LPF@12k 清掉 |
| **音色贴合度对 ir 单调改善** (flat) | MFCC距离: ir0=22.3 → 0.5=21.1 → **0.75=19.8** → 0.9=19.0 (趋平) |
| flat > ivf (音色) | 同 ir0.75: 19.8 vs 23.0 |
| **定案: flat + ir=0.75 + protect=0.33** | ir=0.9 留作耳测候选; protect=0.5 指标略好但=关闭清音保护, 未采用 |

> 本曲日语歌 + 日语语料, 高 ir 无跨语种发音风险 (liyuu 的 ir 顾虑是唱中文)。

### 7.3 F0 音域门禁 (新工具, 全角色通用)

```bash
# 一次性: 生成语料统计 (已缓存 models/corpus_f0_stats.json)
python tools/verify_source_f0.py --corpus <语料.wav> --stats <stats.json>
# 每首歌: 检查推理源 (退出码非0 = 有 OOD 段)
python tools/verify_source_f0.py --stats <stats.json> --input <源wav或segments目录>
```
逐段判 OK (p25-p75) / EDGE (p05-p95) / OOD, 并推荐 transpose (±12/0)。
本曲实测: seg_000/002 OOD→+12, seg_001 无声 (气声段), 其余 OK — 与人工诊断一致。

### 7.4 丢失短段恢复

分段器 MIN_SEGMENT_S=2.0 静默丢掉了 125.62s 处 1.16s 人声 (coverage 测量
发现)。`_recover_short_segs.py`: 重推短段 (带 0.35s 上下文 pad, RVC 对
亚秒片段 F0 跟踪不稳), 按 corpus stats 自动选 key, 写 manifest;
`_rebuild_v4.py --shorts` 回填。

### 7.5 最终产物与验收 (vocals_rebuilt_flat_t12.wav)

```bash
python _rebuild_v4.py --cache-name stage3_cache_flat \
    --tcache-name stage3_cache_flat_key+12 --transposed-segs 0 1 2 \
    --shorts --suffix _flat_t12
```

| 指标 (整曲) | 源 | 旧 ensemble | ivf t12 | **flat t12** |
|---|---:|---:|---:|---:|
| 12-20k 垃圾 (rel core) | −34.0 | −25.2 | −75.7 | **−75.6** |
| 10-20k 窄带峰 | 0 | 143 | 0 | **0** |
| rolloff95 | 9017 | 10486 | 8933 | **8967** |
| 动态范围 | 5.73 | 5.16 | 5.29 | **5.42** |
| 音色距离(→语料) | 52.6 | 50.0 | 44.9 | **43.4** |
| coverage 缺失 | — | 3.0s | 4.0s | **3.0s** (125.6s已回填) |

已知小限制:
- 118.9-119.5s 有 ~0.6s 的 −40dB 衰减尾音三版都缺 (低于 −35dB 切分门限,
  不构成独立区间), 混 BGM 后不可闻, 未修。
- seg_001 是耳语级气声段 (−50.7dB, pyin 测不出 F0)。旧版"听得到"它是因为
  RMS bug 把它抬了 16dB; 新版保留真实动态。coverage 用相对电平 (>15dB
  低于源) 判定, 勿用全局门限 (会把安静段误判为丢失)。

### 7.6 前处理侧结论: 本源不需要 v4 (负结果, 有测量)

> ⚠️ **来源勘误 (2026-07-25)**: `夏日.wav` 是**真人录制的单声道干声**,
> 不是从原曲分离出的 stem (之前一度误记为 stem)。这解释了为何 v4 无收益 ——
> 干声录音本就没有伴奏残留/另一声部/强混响, 无可分离之物。也印证了
> 48s 后 240-270Hz 是**男声本人高音区** (录音里只有一个人, 不存在声部泄漏)。
> **RVC 输入优先级: 真人干声录音 > 分离得好的 stem ≈ 干声 > 脏/湿/带另一声部的 stem。**
> 决定成败的是"分离后残留污染量", 不是"是否分离"; 分得好的 stem = 干声。
> 唯一分离器根治不了的是**同时段同类多人声重叠** (对唱齐唱同一句)。

对原始男声干声 (the raw source vocal, 44.1k float32 mono, 160s **真人干声录音**,
即 source_40k 的母带) 跑了完整 v4 链 (`_prep_source_v4.py`: anvuew dereverb
→ 3×Roformer min-|x|, 80s on 3090)。逐段对比:

| 区域 | v4 后存活 |
|---|---:|
| 主歌/副歌演唱 | **±0.0dB (完全不变)** |
| 耳语段 16.8-23.1s | −3.6dB (损伤) |
| 换气点 62/86/124/157s | **−64dB (被抹成数字静音)** |

全局: dereverb 仅移除 −54dB (无混响可去), 交集仅移除 −35dB 且全部集中在
气声/换气。**该源已是录音室级干声, v4 只有害无益, 不采用。**
产物保留在 `output/tokyo_summer_v4src/` 供复核。

**通用门禁规则 (适用后续 4 个音色)**: 拿到新推理源先跑一遍 v4 链 (80s),
若 dereverb 移除 < −40dB 且交集移除集中在气声段 → 源已干净, 直接用原始文件;
只有输入是混合物/带混响 (如 honoka 的源) 时才采用 v4 产物。
40k-mono PCM16 转换损耗 (vs 44.1k float32) 在 HuBERT 16k 下采样后无意义,
不必纠结母带格式。

### 7.7 脚本清单 (本轮新增)

```
_build_flat_index.py       # 全量 Flat 索引 (6s)
_param_sweep.py            # index/ir/protect 客观扫描
_infer_transposed.py       # 批量重推 (已泛化: --index/--index-rate/--cache-name)
_recover_short_segs.py     # 丢失短段恢复
_rebuild_v4.py             # 重建全链 (选ckpt→tameEQ→LPF12k→RMS→fade→回填)
_prep_source_v4.py         # v4 前处理 (无 trim 版, 保 BGM 对齐) — 本源判定不采用
tools/verify_source_f0.py  # F0 音域门禁 (项目级, 全角色通用)
```

> **可选精修**: honoka 侧后来发明的中值谱 ensemble
> ([honoka/_median_ensemble.py](../honoka/_median_ensemble.py), 见其 KNOWHOW §9.2)
> 实测能把 EQ 修不掉的 2-5k 残余毛刺压到 ~0。如需对本角色再榨一点,
> 把该脚本移植过来 (top-5 幅度谱中值 + 最佳 ckpt 相位 + 修复链)。
> ⚠️ 移植时修复参数用本角色的 (LPF 12k / tame 3.8k), 勿带 honoka 的。

---

## 十、训练侧重做 (otoya_sho_mix_v2, 2026-07-25) — 根治电流声

> 前九章都是**推理侧**优化 (在一个有缺陷的模型上打补丁)。本章重训模型本身。
> 结论: 之前所有 vocoder 伪影 (梳齿啸叫、呼吸幻觉、主歌 OOD) 的**总根源**是
> 两个 v1 训练 bug。修复后零后处理即达到 v1 全套修复链的效果。

### 10.1 ⚠️ 两个 v1 训练 bug (勘误)

**Bug 1 — pretrain 从未加载 (致命)**。v1 的 `step4_train.sh` 调 train.py 时
**没传 `-pg`/`-pd`**。RVC 默认值是空字符串, train.py:225 是
`if hps.pretrainG != "": <load>` — 空值静默跳过, **不报错**。
v1 训练日志 `grep pretrain` 零命中, 而 `assets/pretrained_v2/f0{G,D}40k.pth`
一直躺在盘上。**§二写的 "cold_start from f0G40k/f0D40k" 是记录错误** —
不是"从预训练冷启动", 而是真·随机初始化。用 147min 数据从零训 VITS =
让模型自己重新发明"人声是什么", 这是所有伪影的根。
> 佐证: RVC 源码 `infer-web.py:549` 的参考命令注释本就带
> `-pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth`。

**Bug 2 — filelist.txt 不是 train.py 生成的**。它由 WebUI 的 `click_train`
(`infer-web.py:545`) 写。step 脚本绕过 WebUI 就没有它, 训练直接崩。
v1 当时靠某种手动方式补过但没进文档, 是隐藏坑。
修复: `v2_step3b_filelist.py` 复刻该逻辑 (四阶段目录按 basename 取交集 +
2 条 mute padding + shuffle)。

### 10.2 v2 训练配方

| 项 | v1 | **v2** |
|---|---|---|
| pretrain | ❌ 随机初始化 | ✅ f0G40k+f0D40k (`All keys matched`) |
| 语料 | 707MB 单文件 concat (跨歌切片污染) | 39 个独立文件 (v4.5 重制) |
| 语料残留脏度 | −18.7dB (sho 半) | **−40.8dB** |
| 语料呼吸密度 | 15.3/min | **30.0/min** |
| epochs/存档 | 200/every10 | **200/every20** (从 e100-200 窗口选最佳)¹ |
| 产物位置 | C: (19.8GB 浪费) | ext4, C: 仅收 _infer (2.1GB) |
| 训练时长 | ~12h | ~5h (数据在 ext4) |
| **final loss_mel** | **27.09** | **18.6 (↓31%)** |

流程脚本: `v2_prep_all.sh` (setup+slice+F0+feature) → `v2_step3b_filelist.py`
→ `v2_step4_train.sh` (带 pretrain + 启动前存在性检查, 防止再次静默跳过)。

> ¹ **配方定稿 (后 4 个音色沿用)**: `-te 200 -se 20`, **从 e100-200 窗口选最佳
> ckpt** (早到避免后期漂移, 晚到让音色充分settle)。本轮 otoya 实际训到 300 才
> 发现 e20 就收敛, 故把配方降到 200; e140 (落在窗口内) 被选中且并列/胜过全部。
> 脚本已改 (`v2_step0_setup.sh` epochs=200, `v2_step4_train.sh` -te 200)。

### 10.3 v4.5 语料重制 (`_prep_corpus_v45.py`)

从 26 个原始 raw 轨 (`raw/{otoya,sho}_raw`) 重做:
```
ffmpeg 44.1k → 3×Roformer 直接分离 (弃 demucs: 其 torchaudio 在 venv 里坏了)
  → 门控混合: 响帧 min-|x| (去伴奏) / 静帧 median (保呼吸) sigmoid gate
  → anvuew dereverb → trim + LUFS-20
```
otoya 复制两份 (`_b` 后缀) 实现 2:1, **文件级混合, 不再跨歌拼接**。
验收 `_verify_corpus_v45.py`: 脏度/呼吸/F0/比例四项 + PASS 判定。
> ⚠️ 验收教训: 新语料数字静音 7-8% (旧 0.2%) 一度误判 REVIEW。查实是**间奏段
> 被正确归零** (旧语料那里塞满器乐残留, 被模型学了)。按静音**游程分布**判定
> (长段=间奏无害, <0.5s=呼吸被杀), 别看静音百分比。

### 10.4 客观验收 (v1 vs v2, 全部推理侧脚本复用)

Phase 3: `v2_step6_deploy.sh` (15 ckpt 转 infer, 只拷 _infer 到 C:) →
`_build_flat_index_v2.py` (354,496 向量) → `_infer_v2.py` (全量重推) →
`_rebuild_v4.py --cache-name stage3_cache_v2`。

整曲关键指标:

| | peaks (电流声) | rolloff | 后处理需求 |
|---|---:|---:|---|
| OLD ens (v1 模型) | **143** | 10486 | 波形平均 |
| v1 champ (v1 救活) | 0 | 8967 | **需全套修复链** (LPF12k+tame) |
| **v2 raw (零后处理)** | **0** | **10147** | **无** |

主歌区 (OOD 段):

| | transpose | peaks | rolloff |
|---|---|---:|---:|
| v1 champ | **需 +12 升八度** | 1 | 8541 |
| **v2 key0** | **原调** | **0** | 7946 |

**两个突破**: (1) v1 需把高频砍到 8967 才能压掉 143 峰; v2 零处理就 0 峰且保住
10147 亮度 → 电流声在模型层面消失。(2) v1 主歌必须 +12 升八度 hack; v2 原调
就 0 峰 → OOD 消失, 不再变调。timbre (对新语料) v1/v2 均在 37-40 噪声区内,
无显著差异。

产物: `output/tokyo_summer_v3/_AB_v1_vs_v2/` 5 个候选按序耳测:
```
0_OLD_buzzy_v1ensemble.wav   # 用户抱怨的电流声版 (143 峰)
1_v1champion_needs+12.wav    # v1 推理侧极限 (需升八度)
2_v2_raw_noEQ.wav            # ⭐ v2 零后处理 (最亮/最自然动态)
3_v2_key0_repaired.wav       # v2 + 轻修复链 (可能已过度)
4_v2_+12verse.wav            # v2 主歌仍升八度 (对比用, 预期不需要)
```
> 待耳测: v2_raw 12-20k 有 +9dB 宽带高频 (非梳齿, peaks=0), 若听感偏亮可加
> 14k 以上软 shelf (别用 v1 的硬 LPF 12k — 会砍掉 v2 保住的真实亮度)。

### 10.4b 收敛后单 ckpt = 逐段选优 (方法论沉淀)

v2 用 warm-start + 干净语料训练, **epoch 20 就收敛** (mel loss 从头到尾在
14-17 抖, 无下降趋势; 300ep 是浪费, ~100 足够)。后果: 所有 ckpt 处处
peaks=0, 高度等价。实测把"逐段选优" vs "全曲单个中段 ckpt"并列对比:

| 候选 | peaks | rolloff | dynR | 音色距离(新语料) |
|---|---:|---:|---:|---:|
| 逐段选优 | 0 | 10147 | 5.65 | 40.4 |
| 单 e60 | 0 | 10257 | 5.63 | 40.3 |
| **单 e140 (最终选)** | 0 | 10431 | 5.35 | **40.2** |
| 单 e240 | 0 | 10525 | 5.4 | 41.3 |

**全部在噪声内, 逐段选优没买到任何东西。** 最终成品改用**单 e140 (G_17780)
全曲**: 客观持平/音色微胜, 且无段间 ckpt 切换 (消除边界质感微跳风险), 更简单可复现。
`FINAL_v2_vocals.wav` = 该版本。脚本 `_build_single_ckpt.py` (全 ckpt 打分排名 +
中段优选 + 整曲单模型组装)。

> **核心教训 (对后 4 个音色)**: 逐段选优 / ensemble / 中值谱这些推理侧技巧,
> 本质是在**补救欠训模型的 epoch 间差异**。训练做对了 (加载 pretrain +
> 干净语料), 模型收敛后各 epoch 等价, 这些技巧的边际收益趋近于零。
> **把力气花在训练侧, 推理侧就能极简。** (honoka 的中值谱 ensemble、v1 的
> transpose/修复链, 都是 v1 欠训模型时代的产物。)

### 10.5 后续 (音域增强实验臂, 用户已批准事后做)

主线 v2 不加音域增强。待 v2 耳测确认后, 可另训一个加 −5/−7 半音保共振峰
降调副本的实验臂做 A/B, 归因干净。但注意: v2 主歌 key0 已 peaks=0,
OOD 可能已被 pretrain 底模的低音能力解决, 增强臂的边际收益存疑。

### 10.6 v2 脚本清单
```
_prep_corpus_v45.py     # v4.5 语料重制 (门控 min/median 保呼吸)
_verify_corpus_v45.py   # 语料验收 (脏度/呼吸/F0/比例 + PASS 判定)
v2_step0_setup.sh       # ext4 拷贝 + symlink + config (300ep/se20)
v2_prep_all.sh          # setup+slice+F0+feature 一条龙
v2_step3b_filelist.py   # 生成 filelist.txt (WebUI 之外必需)
v2_step4_train.sh       # 训练 (带 -pg/-pd + 存在性检查) ⭐核心
v2_step6_deploy.sh      # 转 infer 格式 (只拷 _infer 到 C:)
_build_flat_index_v2.py # v2 全量 Flat 索引
_infer_v2.py            # v2 全量重推
tools/watch_log.sh      # 清洗版 tail (去 tqdm CR 乱码)
```
