#!/bin/bash
# honoka v2 deploy — convert an arm's G snapshots to infer format.
#   bash v2_step6_deploy.sh honoka_v2         -> models_v2/
#   bash v2_step6_deploy.sh honoka_v2_titan   -> models_v2_titan/
# Heavy originals stay on ext4; only *_infer.pth land on C:.
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAR=$PROJECT/characters/honoka
PYTHON=$PROJECT/.venv/bin/python
EXP="${1:?usage: v2_step6_deploy.sh honoka_v2|honoka_v2_titan}"
E=${RVC_DATA_DIR:-$HOME/rvc_data}/$EXP/logs
case "$EXP" in
  honoka_v2)       DST=$CHAR/models_v2 ;;
  honoka_v2_titan) DST=$CHAR/models_v2_titan ;;
  *) echo "unknown exp" >&2; exit 1 ;;
esac
CFG=$E/config.json
mkdir -p "$DST"

echo "=== Converting $EXP G_*.pth to infer format -> $DST ==="
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
