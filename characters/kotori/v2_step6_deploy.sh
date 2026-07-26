#!/bin/bash
# kotori v2 deploy — convert kotori_v2 G snapshots to infer format -> models_v2.
# Heavy originals stay on ext4; only *_infer.pth land on C:.
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAR=$PROJECT/characters/kotori
PYTHON=$PROJECT/.venv/bin/python
EXP="${1:-kotori_v2}"
E=${RVC_DATA_DIR:-$HOME/rvc_data}/$EXP/logs
CFG=$E/config.json
DST=$CHAR/models_v2
mkdir -p "$DST"

echo "=== convert $EXP G_*.pth -> infer format -> $DST ==="
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
n = 0
for src in sorted(Path("$E").glob("G_*.pth"), key=lambda p: int(p.stem[2:])):
    dst = Path("$DST") / f"{src.stem}_infer.pth"
    if dst.exists():
        print(f"  SKIP {src.name}"); continue
    cpt = torch.load(str(src), map_location="cpu")
    w = cpt["model"] if "model" in cpt else cpt["weight"]
    torch.save({"config": config_list, "weight": w, "f0": 1,
                "version": "v2", "info": f"$EXP {src.stem}"}, str(dst))
    print(f"  OK  {src.name}")
    n += 1
print(f"converted {n}")
PYEOF
ls "$DST"/*_infer.pth | wc -l
