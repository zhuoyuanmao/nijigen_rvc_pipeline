# METHODOLOGY — 跨音色统一方法论 (character-agnostic)

> 这是**训练/推理任意新音色的单一真相源**。所有音色共用的流程、配方、原则都在这里。
> 每个角色只在自己的 `characters/<name>/KNOWHOW.md` 里记**角色特有的增量**
> (语料构成、F0 音域、特有参数、成品)。
>
> 适用范围: RVC v2, 40kHz, RMVPE F0, ContentVec 768-dim, RTX 3090 (本机 WSL)。
> 建立日期: 2026-07-25, 从 otoya_sho_mix v2 / honoka 的实战中提炼。

---

## 0. 一句话总纲

**把力气花在训练侧, 推理侧就能极简。**
训练做对 (加载 pretrain + 干净语料), 模型收敛后各 epoch 高度等价 →
推理侧塌缩为"挑一个中段 ckpt, 完事"。ensemble / 中值谱 / transpose / 修复链
这些技巧本质是**欠训模型的补救**, 训对了就不需要。

---

## 1. 端到端流水线

```
raw 整曲 (raw/<voice>_raw/*.wav)
   │  [A] v4.5 语料重制 (_prep_corpus_v45.py)
   ▼
data/<voice>_v45_corpus/  (每曲一个干净人声文件)
   │  [B] RVC preprocess 切片 → 0_gt_wavs/{曲号}_{片号}.wav
   ▼
   │  [C] (大语料才需) 切片质量打分择优 (_score_slices.py)
   ▼
   │  [D] F0 (RMVPE) + ContentVec 特征
   │  [E] filelist.txt (v2_step3b_filelist.py, 可选 --keep-list)
   ▼
   │  [F] 训练 (v2_step4_train.sh, 带 pretrain)
   ▼
   │  [G] 部署转 infer 格式 + flat 索引
   ▼
   │  [H] 推理: 单中段 ckpt 重建 (_build_single_ckpt.py)
   ▼
成品人声 → (混 BGM = 最终翻唱)
```

---

## 2. ⚠️ 两个必须避开的致命坑 (v1 的教训)

1. **pretrain 必须显式传 `-pg`/`-pd`**。RVC 默认空字符串, `train.py:225`
   `if hps.pretrainG != "": <load>` — **空值静默跳过, 不报错**。漏传 = 随机
   初始化从零训 = 所有 vocoder 伪影 (梳齿啸叫/呼吸幻觉) 的总根源。
   **启动后必查日志出现 `loaded pretrained ... All keys matched successfully`。**
2. **filelist.txt 不是 train.py 生成的**, 是 WebUI 的 `click_train`
   (`infer-web.py:545`) 写。绕过 WebUI 就没有它, 训练直接崩。
   用 `v2_step3b_filelist.py` 生成 (四阶段目录按 basename 取交集 + 2 条 mute)。

> 这两条修复后, otoya_sho_mix loss_mel 从 27.09 → 18.6, 零后处理即 0 窄带峰。

---

## 3. [A] v4.5 语料重制 (`_prep_corpus_v45.py`)

每曲: `ffmpeg 44.1k → 3×Roformer 直接分离 → 门控混合 → dereverb → trim + LUFS-20`

- **弃 demucs** (其 torchaudio 在本 venv 里坏了; Roformer 直接吃整曲 SDR 更高)。
- **门控混合**: 响帧取 3 模型 min-|x| (保守去伴奏) / 静帧取中值 (保呼吸,
  ≥2/3 模型认可就留)。sigmoid gate on frame RMS。—— 解决 v4 纯 min-|x|
  "把呼吸抹成 −64dB 静音"的问题 (训练数据里呼吸是真实样本, 不能杀)。
- **弃 concat 大文件**: 每曲独立输出。多说话人混合 (如 otoya:sho 2:1) 靠
  **文件级复制** (otoya 每曲存两份 `_b`), 不用 ffmpeg 拼 → 杜绝跨歌切片污染,
  且切片名保留曲号 (见 §5)。
- 3 个 Roformer: kim_ft_unwa + big_beta5e + bs_roformer_1297 (不同架构, 错误
  模式独立); dereverb 用 anvuew less-aggressive。模型缓存
  `~/.cache/audio-separator-models/` (~3.7GB, 首次自动下载)。

验收 `_verify_corpus_v45.py`: 脏度(<−30dB)/呼吸密度/F0/比例 + PASS 判定。
> 验收教训: 别看"数字静音百分比"(会把正确归零的间奏段误判为呼吸被杀),
> 要看**静音游程分布** (长段=间奏无害, <0.5s=呼吸被杀) + 呼吸事件密度。

---

## 4. 推理源质量 (与训练语料不同的判断)

**RVC 输入优先级: 真人干声录音 > 分离得好的 stem ≈ 干声 > 脏/湿/带另一声部的 stem。**
决定成败的是"分离后残留污染量", 不是"是否分离"; 分得好的 stem = 干声。
唯一分离器根治不了的: **同时段同类多人声重叠** (对唱齐唱同一句)。

- **门禁**: 新源先跑一遍 v4 链**当测量**——去混响移除 <−40dB 且交集集中在
  呼吸段 → 源已干净, 直接用原始; 只有混合物/湿源才采用 v4 产物。
- **F0 音域门禁** `tools/verify_source_f0.py`: 逐段判 OK/EDGE/OOD + 推荐
  transpose。语料统计缓存 `models/corpus_f0_stats.json`。
  > 注: 训对的 v2 模型 (warm-start) 已具备 pretrain 底模的低音能力, 旧的
  > "主歌 OOD 就 +12 升八度"hack 基本不再需要 — 但门禁仍用于选歌/预警。
- HuBERT 抽特征时下采样到 16k, 所以**源的超高频保真/母带格式 (40k vs 44.1k
  float32) 不影响结果**, 不必纠结。

---

## 5. [B][C] 切片 + 质量打分择优 (大语料专用)

RVC 切片名 = `{曲号}_{片号}` (`preprocess.py:114` 按文件枚举 idx0)。
**必须喂多个独立文件** (v45_corpus 那样), 才有曲号; v1 喂 concat 大文件 →
全叫 `0_*`, 丢曲目身份 (隐藏缺陷)。

语料远超目标量时 (如 honoka 278min → 目标 180min), 用 `_score_slices.py`:

**每片质量分** (越高越留):
```
quality = 0.40·HNR_norm + 0.25·voiced_ratio + 0.20·(1−flatness) + 0.15·(1−silence_frac)
```
**硬剔除**: voiced_ratio<0.15 (纯静音/呼吸/噪声) · HNR 太低 · F0 不稳 (>30% 帧跳八度)。

**择优到目标量, 两个多样性护栏**:
1. **按曲配额**: 单曲 ≤ 其比例份额 1.4× (防某曲独大, 靠切片名曲号)。
2. **音高覆盖**: 按中位 F0 分箱, 跨箱按比例选 (保全音域, 不只留好唱的中音)。

产物 `selected_slices.txt` → filelist 只含选中片。
> 语料未超量的音色 (如 otoya) 跳过本步, 全量用。

---

## 6. [F] 训练配方 (所有音色统一)

```bash
python train.py -se 20 -te 200 -bs 16 -sr 40k -f0 1 -l 0 -sw 0 \
    -e <exp> -v v2 -c 0 \
    -pg assets/pretrained_v2/f0G40k.pth \
    -pd assets/pretrained_v2/f0D40k.pth
```

| 参数 | 值 | 依据 |
|---|---|---|
| pretrain | f0G40k + f0D40k | **核心**, 见 §2 |
| `-te` | **200** | warm-start 后 ~e20 就收敛; 300 是浪费 |
| `-se` | 20 | 快照 e20,40,…,200 |
| **选 ckpt** | **e100-200 窗口最佳** | 早到避免后期漂移, 晚到让音色 settle |
| batch | 16 | 3090 24GB |

> **底模选择 (2026-07-26 双音色 A/B 定论)**: **TITAN 是亮嗓工具, 男声用官方 f0G40k**。
> 社区底模 TITAN (blaise-tk/TITAN, 40k, 与 v2 架构兼容, `tools/download_titan_pretrain.sh`,
> 仅换 `-pg/-pd`) 实测:
> - **honoka 亮嗓女高音**: 空气感/亮度**小胜** (air12 −26.5 vs baseline −28.6)。
> - **otoya 男声**: 音色**反而略差** (39.5→41.0), 无空气感缺陷 → TITAN 用不上。
>
> **规则**: 亮嗓/女高音 → TITAN; 男声/暗嗓 → 官方 f0G40k。见 honoka §11 / otoya A/B。

- **detached 运行**: `setsid bash v2_step4_train.sh > logs/train_v2.log 2>&1 &`
  (训练不依赖终端/会话存活)。
- **存储策略**: 训练产物全留 ext4 (`~/rvc_data/<exp>/`, 800G+ 空闲);
  C: 盘 (紧张) 只在部署时收小的 `*_infer.pth`。v1 把 D+full-G 全拷 C:
  浪费 16.9GB/音色。
- 时长: 300ep ~5h (数据在 ext4)。200ep ~3-4h。
- 监控: `tools/watch_log.sh` (清洗 tqdm CR 乱码); loss 看 `loss_mel`
  (对应频谱重建质量, 是电流声的直接指标)。

---

## 7. [G][H] 部署 + 推理侧 (训对后极简)

1. **部署** (`v2_step6_deploy.sh`): 15 个 ckpt 转 infer 格式, **只拷 _infer 到 C:**。
2. **flat 全量索引** (`_build_flat_index_v2.py`): 精确检索全部特征向量。
   > 旧 IVF256 + nprobe=1 + 10k 聚类中心 = 检索形同关闭 (音色距离差 1.9-3.1)。
3. **重建**: 两条路, 依音色而定 ——
   a. **单中段 ckpt** (`_build_single_ckpt.py`): e100-200 打分选一个整曲用。简单;
      对 otoya (混合男声) 与逐段选优/ensemble 客观持平。
   b. **中值谱 ensemble** (`honoka/_median_ensemble.py` 或从 stage8 逐 ckpt 输出直接
      取 top-k 幅度谱**逐 bin 中值** + 最佳 ckpt 相位): 对**亮嗓/高频丰富**音色是**真最优**。

   > ⚠️ **单 ckpt 的例外 (honoka 实测, 2026-07-26)**: 单 ckpt 会**裸露该 ckpt 的
   > 谐波间噪声** (300-4k 谱平坦度 0.021 vs 源 0.012), 听感是"背后一层和声/沙噪"。
   > **中值谱 ensemble 同时解决**: 各 ckpt 谐波间噪声落在不同 bin → 中值剔除
   > (降到 0.015, 近源); 谐波一致 → 保留体量; 且**不叠加梳齿** (peaks 仍 0,
   > 不像 v1 波形平均那样把梳齿取并集)。otoya 因音色不同没暴露此点, honoka 亮嗓放大了它。
   > **规则**: 亮嗓/女高音默认用中值谱 ensemble; 单 ckpt 是"够用的简版"。
   > **实测边界 (2026-07-26)**: otoya 男声上中值谱**不比单 ckpt 好** (谐波间噪声
   > 0.057 vs 单 ckpt 0.054, 略高), 因为男声本无 honoka 那种谐波间噪声问题。
   > 所以: **亮嗓 → 中值; 男声/暗嗓 → 单 ckpt**。与"底模选择"同一条分界线。

**训对后不再需要的旧补救**: 波形平均 ensemble (叠加梳齿)、transpose 升八度、
LPF+tame 修复链、逐段选 ckpt。若 v2 仍出**梳齿/呼吸幻觉**, 先查训练 (§2); 若出
**谐波间和声噪音**, 用中值谱 ensemble。

**可选的成品口味层** (混音级, 非缺陷修复):
- **呼吸**: `_fix_breaths.py` (塞回源真呼吸, 修 RMVPE 呼吸段啸叫) / `_remove_breaths.py`
  (−8/−18dB/静音, 嫌呼吸吵时用)。**绝不在前处理去呼吸** (会让 RVC 幻觉啸叫)。
- **暖厚**: 低-中 (<600Hz) +2dB 低搁架, 追 v1 那种"实"的染色 (非保真)。

---

## 8. 通用运维 (WSL / 本机)

- **GPU**: RTX 3090 在本机 WSL, `.venv` torch 2.5.1+cu121, CUDA 可用。
- **PowerShell 调 wsl 的坑**: 命令里带 `|`/`()`/中文易被 PS 解析破坏 →
  **写 .sh 脚本再 `wsl bash script.sh`**, 别 inline。
- **后台长任务**: `run_in_background` / `setsid`; 完成有通知再接续。
- **分离是 GPU-bound 的, ~75s/轨无法绕过** (实测勘误 2026-07-25): v4.5 用
  Roformer (PyTorch, 一直在 GPU), 单模型在 4min 轨上 infer 12-28s 是真实
  GPU 算力。日志里 `CUDAExecutionProvider not available` 警告是关于
  **onnxruntime (ONNX 模型)** 的红鲱鱼——v4.5 不用 ONNX。
  - **环境**: 别同时装 `onnxruntime` + `onnxruntime-gpu` (CPU 版会遮蔽 gpu 版,
    CUDA provider 消失; 且卸载易互相弄残)。正确做法: 只装 `onnxruntime-gpu`
    (本机 1.20.2, 配 CUDA12/cuDNN9)。这是正确性修复, 不提分离速度。
  - 唯一可省: prep 改 **model-major** (每模型全程只加载一次, 而非每轨重载)
    → 58 轨省 ~10min。分离算力下限 (~55min/58轨) 仍不可绕。
  - onnxruntime CUDA 若跑 **ONNX 模型** (MDX/VR), 需把 torch 的 cudnn/cublas
    放 `LD_LIBRARY_PATH` (`.venv/.../nvidia/{cudnn,cublas}/lib`) 才能真正建
    CUDA session (provider "available" ≠ 可用)。
- **磁盘**: C: 紧张 (~9x% 满), ext4 800G+ 空闲 → 重产物一律 ext4。

---

## 9. 各音色进度 (character-specific 详见各 KNOWHOW)

本项目 = **6 音色 (3 男 + 3 女)**, 各带 solo + 男声齐唱 / 女声齐唱。

| 音色 | 性别 | 状态 | KNOWHOW |
|---|---|---|---|
| otoya_sho_mix | 男 (otoya:sho 2:1) | ✅ v2 完成, 成品 e140 单模型 | [link](characters/otoya_sho_mix/KNOWHOW.md) |
| honoka | 女 | v1 成品在; v2 计划中 (§10) | [link](characters/honoka/KNOWHOW.md) |
| (待建 ×4) | 2男 + 2女 | 用本文档配方从头训 | — |

> `liyuu` **不计入这 6 音色** — 是更早的独立项目 (中文歌翻唱, 含唐可可混合),
> 仅作 legacy 方法论源头保留 ([link](characters/liyuu/KNOWHOW.md))。

后 4 个音色: 直接用本文档配方从头训, 跳过 v1 的推理侧挣扎。

---

## 10. 文档体系

- **本文档 (METHODOLOGY.md)**: character-agnostic, 单一真相源。
- **characters/<name>/KNOWHOW.md**: character-specific 增量。
- **legacy (不用于当前 agent)**: `AGENT_LESSONS.md`、`README.md` (描述已死的 v1
  通用管线)、各 `data/*RUN_ME*.md` / `data/*PIPELINE*.md`、
  `latest_feedback/GPU_VOCAL_PREP_v4.md` (v4 原始方案, 已被 v4.5 取代) —
  均为早期 (DeepSeek agent 时代) 产物, 保留备查, 当前流程以本文档为准。
