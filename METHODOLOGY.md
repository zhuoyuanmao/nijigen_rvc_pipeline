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
推理侧塌缩为一条确定性链: **中段 ckpt 的中值谱 ensemble (去谐波间噪声) + 可选呼吸处理, 完事**。
真正的"欠训补救" (波形平均 ensemble / transpose 升八度 / LPF+修复链) 训对了就不需要;
但**中值谱 ensemble 不是补救**, 是零 GPU 成本的去噪 (从已有 ckpt 输出直接算), 男女声都用
(otoya 男声 2026-07-26 耳测确认, 见 §7)。

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
   │  [H] 推理: 中段 ckpt 中值谱 ensemble 重建 (+ 可选呼吸处理)
   ▼
成品人声 (归集到 characters/<name>/output/final/) → (混 BGM = 最终翻唱)
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
> 这是男女声流程的**唯一差异**; 其余 (语料、训练、中值谱重建、去噪、呼吸) 完全相同。

### 6.1 男声 vs 女声 方法论 (换音色只换底模)

| 环节 | 男声 / 暗嗓 | 女声 / 亮嗓 |
|---|---|---|
| 语料重制 (§3) | v4.5 | v4.5 (相同) |
| 训练配方 (§6) | `-se20 -te200 -bs16 -sr40k -f0 1` | 相同 |
| **底模 `-pg/-pd`** | **官方 f0G40k / f0D40k** | **TITAN G/D** |
| 重建 (§7) | 中值谱 ensemble | 中值谱 ensemble (相同) |
| 去噪音 (§7) | = 中值谱 + 源侧分离 | 相同 |
| 呼吸处理 (§7) | 标准步, 按源决定 | 标准步, 按源决定 (相同) |

> **换音色只改一处: `-pg/-pd` 指向 f0G40k (男) 还是 TITAN (女)。其余完全一致。**
> 成品实例: otoya (男) = f0G40k + 中值谱 (`FINAL_otoya_v2.wav`);
> honoka (女) = TITAN + 中值谱 + 呼吸静音 (`FINAL_honoka_v2.wav`)。

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
3. **重建 = 中值谱 ensemble (男女声统一默认)**: 从 stage8 逐 ckpt 输出直接取 e100-200
   窗口各 ckpt 幅度谱**逐 bin 中值** + 中间 ckpt 相位 (`honoka/_median_ensemble.py` /
   `_median_from_stage8.py`)。**零 GPU 成本** (从已有 ckpt 输出算)。
   - **为什么是它**: 各 ckpt 的谐波间噪声落在不同 bin → 中值剔除 (honoka 亮嗓上
     300-4k 谱平坦度 0.021→0.015, 近源); 谐波一致 → 保留体量; **不叠加梳齿**
     (peaks 仍 0, 不像 v1 波形平均取梳齿并集)。听感 = "背后那层和声/沙噪"消失。
   - **单中段 ckpt** (`_build_single_ckpt.py`) 是**够用的简版**: e100-200 打分选一个整曲用,
     省掉中值那步; 客观上常与中值谱在噪声内。作为备选/快速验证保留。

   > **男声也用中值谱 (otoya 耳测定论, 2026-07-26)**: 早前客观测量一度显示 otoya 男声上
   > 中值 vs 单 ckpt 在噪声内 (谐波间 0.057 中值 vs 0.054 单, 略高), 故曾定"男声用单 ckpt"。
   > 但用户 3 路 A/B 耳测 (`0` 单e140 / `1` baseline中值 / `2` TITAN中值) **选定
   > `1` baseline 中值谱** —— 中值那层平滑/去噪在听感上胜出, 客观差异属噪声级。
   > **结论: 中值谱 ensemble 是男女声统一默认; 单 ckpt 仅作简版备选。**
   > 男女声的分界线**只在底模** (§6: 男 f0G40k / 女 TITAN), **不在重建方式**。

**训对后不再需要的旧补救**: 波形平均 ensemble (叠加梳齿)、transpose 升八度、
LPF+tame 修复链、逐段选 ckpt。若 v2 仍出**梳齿/呼吸幻觉**, 先查训练 (§2)。

**成品处理 (男女声通用)**:
- **去噪音 = 没有独立降噪器**。谐波间"和声/沙噪"由**中值谱 ensemble** 去 (§7.3, 默认就做);
  背景/混响噪声在**源侧**解决 (§3 语料 v4.5 分离 + §4 推理源 Roformer 分离 + dereverb)。
  训对的模型 + 中值谱 = 成品已干净, **不要再挂降噪器** (它反而伤谐波)。男声同此。
- **呼吸 (标准步, 按源决定, 男女通用)**: `_fix_breaths.py` (塞回源真呼吸, 修 RMVPE 呼吸段
  啸叫) / `_remove_breaths.py` (−8/−18dB/静音, 嫌句尾呼吸吵时用, 如 honoka 成品)。
  **绝不在前处理去呼吸** (会让 RVC 幻觉啸叫)。otoya 定稿未静音呼吸 (源无碍); honoka 静音。
- **暖厚 (可选口味)**: 低-中 (<600Hz) +2dB 低搁架, 追 v1 那种"实"的染色 (非保真)。
- **成品归集 (约定)**: 各版本混音前的最终人声统一拷到 `characters/<name>/output/final/`,
  命名 `FINAL_<name>_v<N>.wav` (整曲)。stage5/stage8 等中间产物留原处;
  `output/final/` 只放"可交付/可混 BGM"的整曲成品, 一眼可取。
- **小样本锁配方 → 整曲渲染 (工作流)**: 配方 (ckpt 窗口 / 底模 / 呼吸 / 口味) 可在
  单句~数十秒**小样本**上耳测定稿, 再对**全曲源**照配方整曲渲染。若全曲干声源为**后录**
  (如女声全曲), 先把小样本放进 `output/final/` 命名 `*.PLACEHOLDER-<Ns>.wav` 占位,
  录源后照配方重跑, 用整曲 `FINAL_<name>_v<N>.wav` 替换占位。

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

> ⭐ **2026-07-31 定稿 6 音色 (用户拍板)**:
> **女**: honoka · kotori · umi   **男**: otoya_tsukasa_mix · cecil_ai_mix · camus_toya20
>
> 男声经多候选 A/B 耳测选定 (候选: otoya_sho 基线 / toya_camus±c2 / cecil_ai / camus 纯 /
> otoya_tsukasa / camus_toya10·20, 见 §11)。女声全曲干声源已录并出整曲翻唱
> (又果→honoka / 又海→umi / 又鸟→kotori, §11.4, `tokyo_summer_session/female_full_covers/`)。
>
> **2026-08-05: 三个女声的 25s 占位已按 §7 约定替换为整曲成品** —— 全曲干声源录制完成后
> 照原配方重跑, `output/final/FINAL_<name>_v2.wav` 现为 ~208s 整曲 (占位文件已移除)。

| 音色 | 性别 | 定稿 | 说明 | KNOWHOW |
|---|---|:---:|---|---|
| **honoka** | 女 | ⭐ | TITAN 中值谱 + 呼吸静音; **整曲已渲染** (`FINAL_honoka_v2.wav` 208.1s) | [link](characters/honoka/KNOWHOW.md) |
| **kotori** | 女 | ⭐ | TITAN 中值谱 + 呼吸静音 (e100-180); **整曲已渲染** (208.5s) | [link](characters/kotori/KNOWHOW.md) |
| **umi** | 女 | ⭐ | TITAN 中值谱 + 呼吸静音; 含 f0G40k 底模 A/B 负结果 (§底模改不了亮度) | [link](characters/umi/KNOWHOW.md) |
| **otoya_tsukasa_mix** | 男 (otoya:tsukasa 2:1) | ⭐ | tsukasa 替 sho; 距 otoya 3.16dB (明显不同, §11) | [link](characters/otoya_tsukasa_mix/KNOWHOW.md) |
| **cecil_ai_mix** | 男 (cecil:ai 2:1) | ⭐ | f0G40k 中值谱 | [link](characters/cecil_ai_mix/KNOWHOW.md) |
| **camus_toya20** | 男 (camus + 20% toya) | ⭐ | 纯 camus 加 20% toya 微调档, 用户耳测选定; f0G40k 中值谱 (§11) | [link](characters/camus_toya20/KNOWHOW.md) |
| otoya_sho_mix | 男 | 候选 | 基线 baseline 中值谱 (`FINAL_otoya_v2.wav`), 被 otoya_tsukasa 取代 | [link](characters/otoya_sho_mix/KNOWHOW.md) |
| camus / toya_camus±c2 / camus_toya10 | 男 | 实验 | §11 音色天花板实证 (toya≈camus 混不出区别; 纯 camus 命中天花板; 小语料微调档客观不稳) | [link](characters/camus/KNOWHOW.md) |

> ⭐ **2026-08-05 全曲成品定稿**: `tokyo-summer-session_cast_MIX.wav` (227.9s, −12.2 LUFS,
> 真峰值 −1.0 dBTP) —— 6 音色逐句选角 (cast) 的完整翻唱, 人工 AU 逐乐句对齐 + 自动混音链
> (**§12**), 逐句参数见 **§13**。早期各音色 24.5s 对齐样本一并保留在临时云盘。

> `liyuu` **不计入这 6 音色** — 是更早的独立项目 (中文歌翻唱, 含唐可可混合),
> 本仓库不含它, 仅作本方法论的部分经验源头被提及。

后 4 个音色: 直接用本文档配方从头训, 跳过 v1 的推理侧挣扎。

---

## 10. 文档体系

- **本文档 (METHODOLOGY.md)**: character-agnostic, 单一真相源。
- **characters/<name>/KNOWHOW.md**: character-specific 增量。

---

## 11. 音色区分度天花板 — 混合 vs 纯单音色 (何时听得出) [2026-07-31]

> 需求场景: 想让一个新音色"听起来不同" (如 toya_camus 太像 toya, 想更偏 camus)。
> **教训: 单靠调数据配比 / index_rate, 感知差异可能远小于配比变化。**

**先测源歌手本身的音色差距 —— 它是天花板, 模型再怎么调都超不过。**

- **度量**: 全曲 LTAS (长时平均谱) 或 **log-mel 逐格 dB 差** (对数梅尔更贴感知)。
- **刻度**: 同一歌手不同录音 ~2-3dB LTAS; 明显不同的两个歌手 ~5-8dB。
  log-mel 逐格: **<1dB 基本听不出, ~1-2dB 微妙, >3dB 明显** (混入 BGM 后再打对折)。

### 11.1 实证 (toya vs camus, 两人音色本就相近)

| 对象 | 距 toya (log-mel dB) | 可闻? |
|---|---:|---|
| 源歌手 toya vs camus (LTAS) | **2.81** | — (天花板, "同一歌手不同录音"档) |
| 2:1 toya:camus (混合) | 1.76 | 否 |
| 1:2 c2 (camus 加权) | 1.90 | 否 |
| index_rate 0.75→1.0 各版 (mix 男声段) | **全 <1.3** | 否 (用户耳测"都一样") |
| **纯 camus 单音色** (只 12 首 camus, 无 toya) | **2.90** | ✅ 命中天花板, 终于可闻 |
| 对照: otoya_tsukasa 2:1 (两人差异够大) | vs otoya **3.16** | ✅ 明显不同 |

> index_rate 0.75→1.0 实测: 音色仅位移 ~1dB, 且 **不引入 buzz** (flatness 不升反微降)。

### 11.2 决策规则

1. 想要**一个听得出区别的新音色**, 而候选歌手互相相近 (源差 <3dB):
   **混合/调 index_rate 都撞天花板 → 直接训纯单音色** (纯音色恢复源歌手的全部区分度)。
2. 混合 (2:1) 的用途是**借某音色的特质 / 做平均**, 不是"用相近歌手 A 稀释相近歌手 B"。
3. index_rate 对音色区分度影响有限 (~1dB), 不能替代语料选择。
4. 想要大幅不同的音色 → 换一个**真正远离**目标 (源差 ≥5dB) 的歌手 (如 otoya→tsukasa)。

> **拿到"想混一个新音色"的需求, 先花 1 分钟测源歌手 log-mel/LTAS 差, 再决定"混"还是"纯"。**

### 11.3 省算力: 2:1 混音复用技巧

改配比 (如 2:1→1:2, 或从 otoya_sho 换 sho→tsukasa) **不必重新分离**——分离产物
音色无关, 只是文件级复制份数变了。**复用已分离的 `data/v45_corpus`**, 只重做 `_b`
复制 (改 DUP_VOICE) 即可进 v2_prep_all。省 ~45min/角色的分离。
(otoya_tsukasa 就复用了 otoya_sho 的 otoya 分离; camus 复用了 toya_camus 的 camus 分离。)

### 11.4a 混音方法论索引

整曲成品混音 (对齐 → 分轨 → 自动混音链) 见 **§12**; 本曲的逐句混音参数表见 **§13**。

### 11.4 整曲推理源准备 (male_sources / female_sources)

整曲翻唱需**整曲干声源** (小样本只锁配方, §7.3)。源归集到
`<song>/{male,female}_sources/` (男声 1 份共享给所有男角色; 女声每角色 1 份)。

- 结构 = 男声推理源同构: `<src>_full.wav` (高保真原始) + `stage1_40k/source_40k.wav`
  + `stage2_segments/` (静音切段 top_db=35, merge<3s, min2s)。
- **轻前处理判定 (按源实测, 别默认全套)**:
  - 干声源 (底噪 ≤ −80dB、sub-120Hz <0.5% = 无伴奏残留) → **不分离、不降噪** (§4)。
  - 热母带 (LUFS −5~−7、峰 0dBFS、有削波, 如本轮女声源) → declip (cubic 插值 <1ms
    平顶) + 降电平到 ~−3dBFS 留 headroom + 裁首尾静音。夏日男声源本就 −25LUFS/无削波 → 零处理。
  - 源 SR/母带格式不影响结果 (HuBERT 16k 下采样, §4)。
- 女声整曲推理 `index_rate=0.5` (男声 0.75); 其余同 §7 中值谱**逐段推理 + 组装**
  (男声 `_infer_median_full` 的多段版)。

---

## 12. 整曲成品混音 (对齐 + 自动混音链) [2026-08-05 定稿]

> 从 6 条整曲翻唱 (`{male,female}_full_covers/`) + off-vocal BGM 到**可发布成品**。
> 分工: **人工负责创作决策** (逐句选角 casting + 逐乐句对齐), **脚本负责工程处理**
> (电平/动态/空间/母带)。本曲成品: `tokyo-summer-session_cast_MIX.wav` (227.9s)。

### 12.1 对齐: 人工 DAW, 不要自动 (负结果)

**人声 ↔ off-vocal 无法自动对齐** —— 两者没有共同音频内容, 互相关无峰 (实测失败)。
可行的自动方案只有"cover ↔ **原曲人声**"(ASA / VocAlign), 但需先分离原曲干声,
且对唱歌常不准 + 会时间拉伸。**结论: 人工对齐是金标准。**

- 工具: **Adobe Audition 多轨** (比 PR 更适合纯音频: 采样级缩放 + 方向键微移)。
- **粒度 = 乐句 (一口气唱的一句), 不是逐词**。切得比乐句碎会"听起来断了":
  句内字间的小静音含呼吸/辅音尾/自然衰减, 在那里下刀并挪动会破坏演唱呼吸。
- 规则: 只在 ≥0.3-0.5s 真停顿下刀 · **呼吸跟着后一句走** (吸气服务于下句, 别切开) ·
  每个切口 5-20ms 淡化 · 整段偏就整段挪 · ±20-30ms 内不必动 (耳朵是标准, 不是波形)。
- 删除**必须用普通删除留空**, 绝不用波纹删除 (会左移后续内容, 对齐全毁)。

**导出给混音脚本**: 每音色 solo → **导出多轨缩混 > 整个会话** → 全曲等长、位置正确、
其余静音的独立 stem (+ BGM 一条)。24-bit 或 32-float, **轨道增益归 0** (否则增益被烤进
文件且可能削波)。**必须分轨** —— 合并成一条就失去按音色分别处理的能力。

### 12.2 自动混音链 (`mix_full_cast.py`)

```
每音色: HPF75 → 静态增益 (整轨均值 → 目标) → 逐句恒定增益 → 缓慢 leveler (±4dB)
        → de-ess → 轻胶合压缩 → 逐声部前瞻限幅 (−1dBFS)
合唱处理: 男声人性化去相关 (仅合唱段) → 并发门控声像 → 1/√N 齐唱定律
总线:   低搁架(180,+1.5) + 临场(3.8k,+1.5) + 胶合压缩 → 真立体声混响 (send 0.10)
BGM:    人声活动联动闪避 (−1.2dB)
母带:   LUFS −12 → 限幅 → 4× 过采样真峰值 ≤ −1.0 dBTP
```

**目标电平**: 人声 = BGM 有声段 RMS **+5dB** (本曲 BGM −16.9 → 目标 −11.9 dBFS)。

### 12.3 五个关键设计 (都来自实测/耳测迭代)

1. **逐句恒定增益 (核心)**。整轨静态增益只对齐"平均", 主歌 vs 副歌的句间落差仍在
   → 听感"一会儿大一会儿小"。逐句检测 (gap<0.5s 合并, 呼吸不断句) 后**每句一个恒定增益**,
   **增益过渡只发生在句间静音处** + 30ms 平滑。
   > ⚠️ **踩过的坑**: 早期版本在句子内部按"检测到的静音"分段归一 → 一句中间的短气口被
   > 当成边界、不提升而两侧提升 → **句子中间突然变小再变大**。修法即"增益只在句间变化,
   > 句内恒定" —— 物理上不可能中间凹陷。
   > 效果: 逐句电平标准差 男声 3.8-4.9dB → **~1.0dB**, 全部音色 ≤1.3dB。
2. **de-ess 必须在调平之后**。男声原始 −31dBFS, 放在增益前根本触发不了阈值,
   抬 +17dB 后齿音全裸奔。移到调平后 → 六音色阈值一致生效。
3. **1/√N 齐唱定律**。每音色都调到"主唱电平", 多人同唱时**求和叠加** (3男 +9dB / 6人 +15dB)
   → 合唱段炸。按并发人数 N(t) 缩放 **1/N^P**: P=0.85 收得太狠 (合唱比独唱还低 1.8dB,
   "不够气势"); **P=0.5 (等功率) 定稿** → 独唱 −14.3 / 3男 −13.2 / 6人 −13.2 dB,
   高潮 +0.5~1dB, 丰满不炸。
4. **同源男声必须去相关**。3 个男声由**同一条源演唱**推理 (时值/F0/呼吸逐采样相同) →
   齐唱听起来"像一个人被复制三份"; 女声是 3 条独立录音 → 天然像合唱。
   修法: 缓慢漂移微时移 (静态 ±11-13ms + ±8ms 随机游走, 慢速调制顺带产生 ±10-15 音分
   音高漂移 = chorus 原理), **仅在 ≥2 男声同唱时门控生效** (独唱段一个采样不动),
   去相关损失的 ~1.2dB 相干叠加用门控联动 +1.5dB 补回。
5. **并发门控声像**。独唱**保持正中** (人工对齐的位置不动), 合唱时才展开 L/C/R
   (男女各成对分列)。宽度 ±0.35 实测"过于立体、像 KTV" → 定稿 **±0.22**;
   混响 send 0.14 → **0.10** (同因)。

### 12.4 母带与验收

- **LUFS −12** (用户选定) · **真峰值 ≤ −1.0 dBTP** (4× 过采样测量, 保证有损编码不过冲)。
- 验收: 逐句电平 std ≤1.3dB · 合唱 vs 独唱 delta ≈ +0.5~1dB · 零削波 ·
  人声段整体高于纯 BGM 段。

> **可调旋钮 (供后续口味微调)**: 人声/BGM 比 (+5dB) · 齐唱 P (0.5) · 混响 send (0.10) ·
> 声像宽度 (±0.22) · 去相关量 (±11-13ms) · LUFS (−12)。

---

## 13. 逐句混音参数表 → [MIX_TABLE.md](MIX_TABLE.md)

本曲 55 个乐句的**逐句混音决策**(演唱音色 + 各维度处理值)已独立成文档,便于分享/收集反馈:

- **[MIX_TABLE.md](MIX_TABLE.md)** — 逐句表 + 读表说明 + 曲式结构
- `tools/line_table.csv` — 机器可读版
- 生成方式: 由混音链的分析阶段直接导出(与实际施加的处理值一致,非人工誊写)

> **音色命名 (成品表内)**: 男声按配对的女声命名, 使 cast 结构一目了然 ——
> `honoka-male` = otoya_tsukasa_mix · `kotori-male` = cecil_ai_mix · `umi-male` = camus_toya20。
