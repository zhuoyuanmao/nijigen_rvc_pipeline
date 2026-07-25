# nijigen_rvc_pipeline

**用 RVC v2 把日语歌翻唱成二次元角色 (nijigen) 音色的方法论 + 参考代码。**
A methodology repository for RVC → nijigen-character singing covers. Powered by vibe coding.

这个仓库沉淀的是**一整套踩过坑、有客观测量支撑的流程**。代码里的路径都从仓库结构
推导 (`Path(__file__).parents[...]`) 或用环境变量配置 (`RVC_DIR` / `RVC_DATA_DIR` /
`RAW_DIR`)，不绑定某台机器 —— 但它需要外部的模型/数据/RVC 源码才能真正跑，
适合用 AI agent (vibe coding) 按自己环境二次开发。

---

## 从这里开始

### 📖 [METHODOLOGY.md](METHODOLOGY.md) — 核心，先读这个

跨角色·跨歌曲的统一方法论：端到端流水线、两个致命训练坑、v4.5 语料重制、
推理源质量判断、切片打分择优、训练配方、"训练做对→推理极简"原则、运维要点。
**训任意新角色都照它。**

> **一句话总纲**：把力气花在训练侧，推理侧就能极简。训练做对 (加载 pretrain +
> 干净语料)，模型收敛后各 epoch 高度等价 → 推理塌缩为"挑一个中段 ckpt，完事"。
> ensemble / 中值谱 / transpose / 修复链这些技巧，本质是欠训模型的补救。

---

## 项目 / Projects

### 🎵 東京サマーセッション (Tokyo Summer Session / 东京夏日相会)

男女对唱翻唱。目标 **6 音色 (3 男 + 3 女)**，每音色出 solo，另做**男声齐唱**
(3 男同唱) 与**女声齐唱** (3 女同唱)。

| # | 音色 | 性别 | 模型 | solo 成品 |
|:-:|---|:-:|---|:-:|
| 1 | **otoya_sho_mix** (音也:翔 2:1) | 男 | ✅ v2 · e140 单模型 | ✅ |
| 2 | *(待建)* | 男 | ⬜ | ⬜ |
| 3 | *(待建)* | 男 | ⬜ | ⬜ |
| 4 | **honoka** (穂乃果) | 女 | ✅ v1 · v2 择优计划中 | ✅ v1 |
| 5 | *(待建)* | 女 | ⬜ | ⬜ |
| 6 | *(待建)* | 女 | ⬜ | ⬜ |
| — | 男声齐唱 (1+2+3) | — | — | ⬜ |
| — | 女声齐唱 (4+5+6) | — | — | ⬜ |

各音色的实战细节见 `characters/<name>/KNOWHOW.md`。

### 🍂 未来季节曲 (planned)

同套方法论炮制的季节续作，各自一个项目章节：
**东京秋日相会 · 东京冬日相会 · 东京春日相会**。

---

## 结构

```
METHODOLOGY.md            ⭐ 跨角色·跨歌曲方法论 (character-agnostic)
requirements.txt          已验证的依赖版本 (torch 2.5.1+cu121 等)
tools/
├── verify_source_f0.py   源音域 OOD 门禁 (选歌/预警, 全角色通用)
├── watch_log.sh          清洗版 tail (去 tqdm CR 乱码)
└── download_rvc_pretrains.sh
characters/               各角色的参考实现 (character-specific)
├── otoya_sho_mix/        ⭐ 最完整的 v2 参考 (男声, 音也:翔 2:1 文件级混合)
│   ├── KNOWHOW.md        全链实战 (含 §10 训练侧根治电流声)
│   ├── _prep_corpus_v45.py / _verify_corpus_v45.py / _prep_source_v4.py
│   ├── v2_step0_setup / v2_prep_all / v2_step3b_filelist / v2_step4_train / v2_step6_deploy
│   ├── _build_flat_index_v2.py / _infer_v2.py / _infer_transposed.py
│   ├── _rebuild_v4.py / _build_single_ckpt.py
│   └── _param_sweep.py / _transpose_probe.py / _recover_short_segs.py
└── honoka/               女高音; character-specific 修复参数 + 两个通用技术
    ├── _median_ensemble.py   中值谱 ensemble (剔除各 ckpt 梳齿而不相消)
    ├── _fix_breaths.py       呼吸段源替换 (修 RMVPE 呼吸幻觉啸叫)
    └── _rebuild_v4.py        honoka 参数版 (对比 otoya, 见"勿盲搬"教训)
```

> 环境变量：`RVC_DIR` (RVC 源码位置)、`RVC_DATA_DIR` (训练特征/ckpt 的快盘目录)、
> `RAW_DIR` (原始整曲下载)、`PYTHON` (解释器)。都有从仓库结构推导的默认值。

---

## 方法论要点 (都有实测支撑)

1. **pretrain 必须显式传 `-pg`/`-pd`** — RVC 默认空串会**静默跳过**、随机初始化，
   是所有 vocoder 伪影 (电流声/呼吸幻觉) 的总根源。启动后必查日志
   `loaded pretrained ... All keys matched successfully`。
2. **filelist.txt 由 WebUI 生成**，绕过 WebUI 的脚本必须自己生成 (`v2_step3b_filelist.py`)。
3. **v4.5 语料重制**：3×Roformer 分离 + 门控混合 (响帧去伴奏 / 静帧保呼吸)。
   弃 demucs、弃 concat 大文件 (切片保留曲目身份)。
4. **flat 全量索引**：旧 IVF256+nprobe=1 检索形同关闭。
5. **推理侧极简**：训对后单个中段 ckpt = 逐段选优/ensemble (客观持平)。出伪影先查训练。

案例演进见各 `characters/*/KNOWHOW.md`；otoya_sho_mix §10 记录了从"电流声很重"
到"零后处理即 0 窄带峰"的完整训练侧根治过程。

---

## 环境

RVC v2 · 40kHz · RMVPE F0 · ContentVec 768-dim · RTX 3090 (WSL, torch 2.5.1+cu121)。
依赖安装的三角约束 (RVC / demucs / audio-separator) 见 `requirements.txt` 顶部注释。

---

*License: MIT. 代码为参考实现，需外部模型/数据 + 按环境适配。*
