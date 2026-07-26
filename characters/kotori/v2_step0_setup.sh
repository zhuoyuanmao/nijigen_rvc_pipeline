#!/bin/bash
# kotori_v2 Step 0: v4.5 corpus to ext4, symlink, config (200ep/se20/bs16).
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVC=$PROJECT/Retrieval-based-Voice-Conversion-WebUI
PYTHON=$PROJECT/.venv/bin/python
CHAR=$PROJECT/characters/kotori
EXP=kotori_v2
EXTHOME=${RVC_DATA_DIR:-$HOME/rvc_data}/$EXP

echo "=== $EXP Step 0: Setup ==="
mkdir -p "$EXTHOME/dataset_raw" "$EXTHOME/logs"

echo "Copying v4.5 corpus to ext4 ..."
rsync -a --delete "$CHAR/data/v45_corpus/" "$EXTHOME/dataset_raw/"
echo "  $(ls "$EXTHOME/dataset_raw" | wc -l) files, $(du -sh "$EXTHOME/dataset_raw" | cut -f1)"

rm -rf "$RVC/dataset_raw/$EXP" "$RVC/logs/$EXP"
ln -sfn "$EXTHOME/dataset_raw" "$RVC/dataset_raw/$EXP"
ln -sfn "$EXTHOME/logs" "$RVC/logs/$EXP"

cp "$RVC/configs/v1/40k.json" "$EXTHOME/logs/config.json"
$PYTHON - <<PYEOF
import json
p = "$EXTHOME/logs/config.json"
cfg = json.load(open(p))
cfg["train"].update({"epochs": 200, "save_every_epoch": 20, "batch_size": 16})
json.dump(cfg, open(p, "w"), indent=2)
print("config.json: epochs=200, save_every=20, batch_size=16")
PYEOF
echo "Setup done."
