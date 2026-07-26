# nijigen_rvc_pipeline — 动漫角色 RVC 翻唱生产线

用 RVC v2 声音转换，把一首日语歌翻唱成**动漫角色音色**的成品。
当前项目：《東京サマーセッション》（男女对唱）的 **6 音色翻唱**
（3 男 + 3 女，每个音色出 solo，另有男声齐唱 / 女声齐唱）。

本机 RTX 3090 (WSL Ubuntu, `.venv` torch 2.5.1+cu121)，全流程本地跑。

---

## 从这里开始

| 你要做什么 | 看哪个 |
|---|---|
| **训练/推理任意音色的完整流程与配方** | **[METHODOLOGY.md](METHODOLOGY.md)** ← 单一真相源 |
| 某个音色的特有细节（语料、音域、成品） | `characters/<name>/KNOWHOW.md` |
| 新歌/新音色开工前检查源音域是否 OOD | `tools/verify_source_f0.py` |
| 看长任务日志（分离/训练，去 tqdm 乱码） | `tools/watch_log.sh` |

**核心原则**（详见 METHODOLOGY §0）：
> 把力气花在训练侧，推理侧就能极简。训练做对（加载 pretrain + 干净语料），
> 模型收敛后各 epoch 等价 → 推理塌缩为"挑一个中段 ckpt，完事"。

---

## 目录结构

```
nijigen_rvc_pipeline/
├── METHODOLOGY.md          ⭐ 跨音色统一方法论 (character-agnostic)
├── characters/<name>/      每个音色一个自包含目录
│   ├── KNOWHOW.md          该音色特有的实战记录 (character-specific)
│   ├── data/               语料 (v45_corpus/) + 推理源
│   ├── models_v2/          训练产出 (只放 *_infer.pth + flat 索引)
│   ├── output/<song>/      推理中间产物 + 成品 (stage5_rebuilt/)
│   ├── v2_step*.sh         训练流程脚本 (setup/prep/train/deploy)
│   └── _*.py               语料重制/打分/推理/重建脚本
├── tools/                  跨音色通用工具 (verify_source_f0, watch_log)
├── models_cache/           分离器 ONNX 权重
└── Retrieval-based-Voice-Conversion-WebUI/   RVC 源码 (第三方)
```

重产物（完整 G/D checkpoint、训练特征）一律放 WSL ext4
`~/rvc_data/<exp>/`（800G+ 空闲）；C: 盘紧张，只收小的 `*_infer.pth`。

---

## 音色进度（6 音色 = 3 男 + 3 女）

| 音色 | 性别 | 状态 |
|---|---|---|
| **otoya_sho_mix** | 男 (音也:翔 2:1) | ✅ v2 完成，成品 e140 单模型 |
| **honoka** | 女 | ✅ v2 完成，成品 TITAN 中值谱 ensemble + 呼吸静音 (`FINAL_honoka_v2.wav`)，详见 KNOWHOW §11.4 |
| **kotori** | 女 | 🔄 v2 TITAN 训练中 (200ep)，完成后走与 honoka 相同的收尾链 |
| (待建 ×3) | 2 男 + 1 女 | 用 METHODOLOGY 配方从头训 |

> `liyuu` **不属于**这 6 音色 —— 是更早的独立项目（中文歌翻唱，含唐可可混合），
> 保留为方法论源头（见其 KNOWHOW 的失败尝试表），不计入本项目进度。

---

## 关键经验（都有实测支撑，详见 METHODOLOGY）

1. **pretrain 必须显式传 `-pg`/`-pd`**——漏传会静默跳过、随机初始化，是所有
   vocoder 伪影（电流声/呼吸幻觉）的总根源。启动后必查日志
   `loaded pretrained ... All keys matched successfully`。
2. **v4.5 语料重制**：3×Roformer 分离 + 门控混合（响帧去伴奏 / 静帧保呼吸）
   + 去混响。弃 demucs、弃 concat 大文件（切片保留曲目身份）。
3. **flat 全量索引**：旧 IVF256+nprobe=1 检索形同关闭。
4. **推理侧极简**：训对后不需要 ensemble/中值谱/transpose/修复链——那些是
   欠训模型的补救。出伪影先怀疑训练没跑对。

---

## 历史文档 (legacy, 备查)

以下为早期 (DeepSeek agent 时代) 产物，当前流程**不以它们为准**：
`AGENT_LESSONS.md`、`steps/` + `codes/pipeline.py`（已冻结的 v1 通用管线）、
各 `characters/*/data/*RUN_ME*.md`、`latest_feedback/GPU_VOCAL_PREP_v4.md`
（v4 原始方案，已被 v4.5 取代）。

> 注：本仓库当前不是 git 仓库；机器间传递靠打包 zip（见 `tools/zip_for_gpu.sh`），
> 不是 README 早期版本描述的双机 git 工作流。
