# kotori — 亮嗓女高音 RVC v2 (第三女声)

> 📖 **跨音色统一流程/配方见 [../../METHODOLOGY.md](../../METHODOLOGY.md)。**
> 本文档只记 kotori 特有增量。配方与 honoka 同族 (亮嗓女高音 → TITAN + 中值谱)。
>
> 角色: kotori | 模型: RVC v2, 40kHz, RMVPE, ContentVec 768 | GPU: RTX 3090

---

## 一、语料 (v4.5)
- 源: `raw/kotori_raw` → v4.5 语料重制 (§3): 3×Roformer 分离 + 门控混合 + dereverb + LUFS-20。
- 打分择优 (`_score_slices.py`, §5): **2962 片入选 / ~180min** (63 曲), flat 索引 537,125 向量。

## 二、训练 (TITAN, §6)
- 底模 **TITAN** (亮嗓女高音, 见 METHODOLOGY §6.1)。`-se20 -te200 -bs16 -sr40k -f0 1 -pg/-pd TITAN`。
- 启动确认 `loaded pretrained ... All keys matched successfully` ✅ (G+D)。
- mel loss ~20.4 (亮嗓健康区)。
- ⚠️ **训练止于 e182** (中途被系统 OOM 杀掉, 非配置问题)。warm-start 下 ~e20 即收敛,
  重建用 **e100–180 窗口** (5 ckpt: e101/121/142/162/182) 做中值谱, 完全够, 未重训。

## 三、推理 + 成品 (§7)
- 部署 `v2_step6_deploy.sh` → `models_v2/` 9 个 `*_infer.pth` + `_build_flat_index_v2.py` flat 索引。
- `_infer_median.py`: e100–180 各 ckpt 在女声源段推理 → **中值谱 ensemble** → `stage5_rebuilt/kotori_MEDIAN.wav`。
- `_remove_breaths.py --duck-db -99`: 呼吸全静音 (用户口味, 同 honoka)。
- **配方定稿** = TITAN 中值谱 + 呼吸静音 (素/无 warm), 与 honoka 一致。

## 四、状态
- ✅ 配方定稿, 25s 小样本验收。
- ⏳ `output/final/FINAL_kotori_v2.PLACEHOLDER-25s.wav` 仅占位; **整曲渲染待录制女声全曲干声源**后
  照配方重跑 (工作流见 METHODOLOGY §7 "小样本锁配方 → 整曲渲染"; 同 honoka)。
