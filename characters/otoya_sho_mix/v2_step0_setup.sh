#!/bin/bash
# otoya_sho_mix_v2 Step 0: setup — v4.5 corpus to ext4, symlink, config.
#
# Changes vs the v1 round:
#   * corpus is now 39 PER-TRACK files (v45_corpus/), not one 707MB concat,
#     so no slice ever straddles two songs / two singers.
#   * everything heavy stays on ext4 (826G free); C: is at 91% and only
#     receives the small *_infer.pth files at deploy time.
set -e
RVC=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/Retrieval-based-Voice-Conversion-WebUI
PYTHON=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/.venv/bin/python
CHAR=/mnt/c/Users/kevin/ai_sing_by_ai/ja_tts_explore/characters/otoya_sho_mix
EXP=otoya_sho_mix_v2
EXTHOME=/home/kevin/rvc_data/$EXP

echo "=== $EXP Step 0: Setup ==="

mkdir -p "$EXTHOME/dataset_raw" "$EXTHOME/logs"

echo "Copying v4.5 corpus to ext4 ..."
rsync -a --delete "$CHAR/data/v45_corpus/" "$EXTHOME/dataset_raw/"
echo "  $(ls "$EXTHOME/dataset_raw" | wc -l) files, $(du -sh "$EXTHOME/dataset_raw" | cut -f1)"

rm -rf "$RVC/dataset_raw/$EXP" "$RVC/logs/$EXP"
ln -sfn "$EXTHOME/dataset_raw" "$RVC/dataset_raw/$EXP"
ln -sfn "$EXTHOME/logs" "$RVC/logs/$EXP"

cp "$RVC/configs/v1/40k.json" "$EXTHOME/logs/config.json"
cd "$RVC"
$PYTHON - <<PYEOF
import json
p = "$EXTHOME/logs/config.json"
cfg = json.load(open(p))
cfg["train"].update({"epochs": 200, "save_every_epoch": 20, "batch_size": 16})
json.dump(cfg, open(p, "w"), indent=2)
print("config.json: epochs=200, save_every=20, batch_size=16")
PYEOF

echo "Setup done. dataset_raw -> $EXTHOME/dataset_raw"
