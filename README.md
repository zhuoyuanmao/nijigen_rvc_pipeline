# nijigen_rvc_pipeline

**用 RVC v2 把日语歌翻唱成二次元角色 (nijigen) 音色的方法论 + 参考代码。**
A methodology repository for RVC → nijigen-character singing covers. Powered by vibe coding.

这个仓库沉淀的是**一整套踩过坑、有客观测量支撑的流程**，不是一键运行的工具。
代码里的路径是原机器 (`/mnt/c/Users/<you>/...`, WSL + RTX 3090) 特有的，
拿去用需要按自己环境适配 —— 价值在方法论与代码逻辑，适合用 AI agent (vibe coding) 二次开发。

---

## 从这里开始

### 📖 [METHODOLOGY.md](METHODOLOGY.md) — 核心，先读这个

跨角色统一方法论：端到端流水线、两个致命训练坑、v4.5 语料重制、推理源质量判断、
切片打分择优、训练配方、"训练做对→推理极简"原则、运维要点。**训任意新角色都照它。**

### 一句话总纲

> 把力气花在训练侧，推理侧就能极简。训练做对 (加载 pretrain + 干净语料)，
> 模型收敛后各 epoch 高度等价 → 推理塌缩为"挑一个中段 ckpt，完事"。
> ensemble / 中值谱 / transpose / 修复链这些技巧，本质是欠训模型的补救。

---

## 结构

```
METHODOLOGY.md            ⭐ 跨角色方法论 (character-agnostic)
requirements.txt          Python 依赖 (RVC / demucs / audio-separator 三方约束)
tools/
├── verify_source_f0.py   源音域 OOD 门禁 (选歌/预警, 全角色通用)
├── watch_log.sh          清洗版 tail (去 tqdm CR 乱码)
└── download_rvc_pretrains.sh
characters/               各角色的实战案例 (character-specific) + 参考脚本
├── otoya_sho_mix/        ⭐ 最完整的 v2 参考实现 (男声, 音也:翔 2:1 混合)
│   ├── KNOWHOW.md        全链实战记录 (含 §10 训练侧根治电流声)
│   ├── _prep_corpus_v45.py / _verify_corpus_v45.py / _prep_source_v4.py
│   ├── v2_step0_setup / v2_prep_all / v2_step3b_filelist / v2_step4_train / v2_step6_deploy
│   ├── _build_flat_index_v2.py / _infer_v2.py / _infer_transposed.py
│   ├── _rebuild_v4.py / _build_single_ckpt.py
│   └── _param_sweep.py / _transpose_probe.py / _recover_short_segs.py
├── honoka/               女高音; character-specific 修复参数 + 两个通用技术
│   ├── _median_ensemble.py   中值谱 ensemble (剔除各 ckpt 梳齿而不相消)
│   ├── _fix_breaths.py       呼吸段源替换 (修 RMVPE 呼吸幻觉啸叫)
│   └── _rebuild_v4.py        honoka 参数版 (对比 otoya, 见"勿盲搬"教训)
└── liyuu/                更早的方法论源头 (含失败尝试表, legacy)
```

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

*License: MIT. 代码为参考实现，路径/环境需自行适配。*
