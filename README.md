# nijigen_rvc_pipeline — 动漫角色 RVC 翻唱生产线

用 RVC v2 声音转换，把一首日语歌翻唱成**动漫角色音色**的成品。
当前项目：《東京サマーセッション》（男女对唱）的 **6 音色翻唱**（3 男 + 3 女）。
每个音色有整曲 solo（`{male,female}_full_covers/`），成品是**逐句选角（cast）版**——
每句歌词派给一个音色，副歌段三男齐唱 / 六人齐唱。

本机 RTX 3090 (WSL Ubuntu, `.venv` torch 2.5.1+cu121)，全流程本地跑。

---

## 从这里开始

| 你要做什么 | 看哪个 |
|---|---|
| **训练/推理任意音色的完整流程与配方** | **[METHODOLOGY.md](METHODOLOGY.md)** ← 单一真相源 |
| **看成品每句谁唱、混音怎么处理的** | **[MIX_TABLE.md](MIX_TABLE.md)** ← 逐句参数表 |
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

> ⭐ **2026-07-31 定稿 6 音色**：女 **honoka · kotori · umi**，男 **otoya_tsukasa_mix · cecil_ai_mix · camus_toya20**。

| 音色 | 性别 | 状态 |
|---|---|---|
| **honoka / kotori / umi** | 女 ×3 | ⭐ 定稿（TITAN 中值谱 + 呼吸静音）；女声全曲干声源已录，**整曲翻唱已出**（`female_full_covers/`） |
| **otoya_tsukasa_mix** | 男 (音也:司 2:1) | ⭐ 定稿，tsukasa 替翔，距 otoya 3.16dB 明显不同 |
| **cecil_ai_mix** | 男 (cecil:ai 2:1) | ⭐ 定稿，f0G40k 中值谱 |
| **camus_toya20** | 男 (camus + 20% toya) | ⭐ 定稿（纯 camus 加 20% toya 微调，用户耳测选定） |
| otoya_sho / toya_camus±c2 / camus 纯 / camus_toya10 | 男 | 候选/实验 → 见 [METHODOLOGY §11](METHODOLOGY.md) 音色天花板实证 |

> 男声经多候选 A/B 耳测选定上述 3 个。成品混音交付到临时云盘。

### 🎵 全曲成品 (2026-08-07 定稿)

**`tokyo-summer-session_lovelive-cover_v16.wav`** — 227.9s / 62 乐句，6 音色**逐句选角 (cast)** 的
完整翻唱（−12.2 LUFS / 真峰值 −1.0 dBTP）。

工作流：人工在 Audition 里**逐乐句对齐 + casting** → 脚本做**自动混音链**
（逐句调平 → 去齿音 → **男声齐唱去相关** → 并发声像 → 齐唱定律 → 真立体声混响
→ BGM 避让 → 真峰值母带）→ **外科式修音**（21/62 句：19 句按原唱参照自动修，
2 句 A/B 耳测定值 —— 其中 1 句参照根本不可用，靠**曲内平行段自洽**查出）。

> 修音参照自 v16 起换成**六声优版**原唱 —— 它的逐句选角与本 cast 对应，
> 可比对的句子 42→47、自动修音 16→18 句。**参照与 cast 的声部匹配度决定修音覆盖率。**

- **逐句参数表（每句谁唱 + 各维度处理值 + 音准/修音）：[MIX_TABLE.md](MIX_TABLE.md)** ← 想给人看/收 feedback 看这个
- 混音方法论：[METHODOLOGY §12](METHODOLOGY.md)　·　修音：[§12.5](METHODOLOGY.md)
- 参考实现：[tools/mix_full_cast.py](tools/mix_full_cast.py)（混音）·
  [tools/pitch_correct.py](tools/pitch_correct.py)（修音）·
  [tools/export_line_table.py](tools/export_line_table.py)（生成本表）

> 成品表内男声按**配对的女声**命名，cast 结构一目了然：
> `honoka-male` = otoya_tsukasa_mix　·　`kotori-male` = cecil_ai_mix　·　`umi-male` = camus_toya20

> `liyuu` **不属于**这 6 音色 —— 是更早的独立中文翻唱项目（含唐可可混合），
> 本仓库不含它，只是本方法论的部分经验源头，故各处会提及 Liyuu 项目。

---

## 关键经验（都有实测支撑，详见 METHODOLOGY）

1. **pretrain 必须显式传 `-pg`/`-pd`**——漏传会静默跳过、随机初始化，是所有
   vocoder 伪影（电流声/呼吸幻觉）的总根源。启动后必查日志
   `loaded pretrained ... All keys matched successfully`。
2. **v4.5 语料重制**：3×Roformer 分离 + 门控混合（响帧去伴奏 / 静帧保呼吸）
   + 去混响。弃 demucs、弃 concat 大文件（切片保留曲目身份）。
3. **flat 全量索引**：旧 IVF256+nprobe=1 检索形同关闭。
4. **推理侧极简**：训对后不需要波形平均 ensemble/transpose/修复链——那些是
   欠训模型的补救。**唯一保留的推理技巧是中值谱 ensemble**（零成本去谐波间噪声，
   男女声统一默认，otoya 2026-07-26 耳测确认），单 ckpt 为简版备选。
5. **男女声唯一差异 = 底模**：男/暗嗓用官方 f0G40k，亮嗓/女高音用 TITAN；
   其余（语料、训练、中值谱重建、去噪、呼吸）完全相同。
6. **音色区分度有天花板**（§11）：想混一个"听得出不同"的新音色前，先测**源歌手本身**的
   log-mel/LTAS 差。歌手相近（<3dB）时，调配比/index_rate 都撞天花板、听不出——
   要用**纯单音色**才能恢复全部区分度（camus 纯版 2.90dB vs 混合 1.76dB 实证）。
7. **同源人声齐唱会融合成一个人**（§12.3）：三个男声由同一条源演唱推理 → 时值/F0 逐采样
   相同，耳朵按共同起音归并成单一声源。时值错开有 **~30ms 硬上限**（超过变回声）；
   真正管用的是**共振峰微移**（不同体型）+ **独立颤音**。
8. **修音要先测量**（§12.5）：RVC 忠实保留源演唱的 F0，音准问题只来自演唱本身。本曲实测
   中位偏差仅 20 音分 → 只修 21/62 句。对唱歌曲尤其危险：参照里可能是**另一个声部**，
   八度折叠还会把"男女差 16 个半音"伪装成"中等走音"，照修会把声部拽错。

---

## 参考与致谢 (Acknowledgments)

本仓库是在开源 **RVC (Retrieval-based Voice Conversion)** 之上的**方法论与生产实践**；
声音转换引擎本身来自上游开源项目，特此致谢：

- **Retrieval-based-Voice-Conversion-WebUI** (RVC-Project, MIT) —— 核心训练 / 推理引擎。
  <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>
  目录树中的 `Retrieval-based-Voice-Conversion-WebUI/` 即其源码（第三方，**不随本仓库分发**）。

底模与关键子模型（同样不随本仓库分发，请按各自许可自行获取）：

- **TITAN** 社区底模 (blaise-tk) —— 亮嗓女高音起训底模，见 METHODOLOGY §6/§7 与 `tools/download_titan_pretrain.sh`。
  <https://huggingface.co/blaise-tk/TITAN>（镜像 <https://huggingface.co/Politrees/RVC_resources>）
- **ContentVec**（768-dim 内容特征）、**RMVPE**（F0 提取）—— RVC v2 默认组件。
- 人声分离 / 去混响（v4.5 语料重制）：**BS-RoFormer / Mel-RoFormer** 系列 + **anvuew** dereverb。

> 本仓库仅含**方法论文档 + 参考脚本**（MIT，见 [LICENSE](LICENSE)），
> 不包含也不分发上述任何模型权重或第三方源码。
