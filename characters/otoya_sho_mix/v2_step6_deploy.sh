#!/bin/bash
# otoya_sho_mix_v2 Step 6: deploy ckpts to infer format.
#
# Storage policy fix vs v1: v1's step6_deploy.sh copied every G_*.pth AND
# every D_*.pth to C: (16.9GB of dead weight — D is only for resuming, full
# G only for conversion). Here the heavy originals stay on ext4; only the
# small *_infer.pth (~140MB each) land in models_v2/ on C:.
set -e
CHAR=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/characters/otoya_sho_mix
E=/home/kevin/rvc_data/otoya_sho_mix_v2/logs
CFG=$E/config.json
DST=$CHAR/models_v2
PYTHON=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/.venv/bin/python
MIN_STEP=2540   # epoch 20

mkdir -p "$DST"
echo "=== Converting G_*.pth (epoch>=20) to infer format ==="
$PYTHON - <<PYEOF
import json, torch
from pathlib import Path
cfg = json.loads(Path("$CFG").read_text())
m, d, t = cfg["model"], cfg["data"], cfg["train"]
config_list = [
    d["n_mel_channels"], t["segment_size"],
    m["inter_channels"], m["hidden_channels"], m["filter_channels"],
    m["n_heads"], m["n_layers"], m["kernel_size"], m["p_dropout"],
    m["resblock"], m["resblock_kernel_sizes"], m["resblock_dilation_sizes"],
    m["upsample_rates"], m["upsample_initial_channel"], m["upsample_kernel_sizes"],
    m["spk_embed_dim"], m["gin_channels"], d["sampling_rate"],
]
src_dir = Path("$E")
dst_dir = Path("$DST")
n = 0
for src in sorted(src_dir.glob("G_*.pth"), key=lambda p:int(p.stem[2:])):
    step = int(src.stem[2:])
    if step < $MIN_STEP: continue
    dst = dst_dir / f"{src.stem}_infer.pth"
    if dst.exists():
        print(f"  SKIP {src.name}"); continue
    cpt = torch.load(str(src), map_location="cpu")
    w = cpt["model"] if "model" in cpt else cpt["weight"]
    torch.save({"config": config_list, "weight": w, "f0": 1,
                "version": "v2", "info": f"otoya_sho_mix_v2 {src.stem}"}, str(dst))
    print(f"  OK  {src.name} -> {dst.name}")
    n += 1
print(f"converted {n} checkpoints")
PYEOF
echo ""
echo "models_v2 size: $(du -sh "$DST" | cut -f1)"
ls "$DST"/*_infer.pth | wc -l
