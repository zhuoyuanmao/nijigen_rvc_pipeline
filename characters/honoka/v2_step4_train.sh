#!/bin/bash
# honoka v2 TRAIN — one script for both A/B arms.
#
#   bash v2_step4_train.sh baseline   # arm A: official f0G40k/f0D40k
#   bash v2_step4_train.sh titan      # arm B: TITAN community pretrain
#
# Arm B reuses arm A's features: logs/honoka_v2_titan gets a copy of
# config.json + filelist.txt whose absolute paths point into honoka_v2's
# stage dirs — no re-extraction. Only -pg/-pd differ (clean attribution).
#
# ALWAYS verify the log prints:  loaded pretrained ... All keys matched
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVC=$PROJECT/Retrieval-based-Voice-Conversion-WebUI
PYTHON=$PROJECT/.venv/bin/python
ARM="${1:?usage: v2_step4_train.sh baseline|titan}"

case "$ARM" in
  baseline)
    EXP=honoka_v2
    PG="$RVC/assets/pretrained_v2/f0G40k.pth"
    PD="$RVC/assets/pretrained_v2/f0D40k.pth"
    ;;
  titan)
    EXP=honoka_v2_titan
    PG="$RVC/assets/pretrained_v2_titan/G-f040k-TITAN.pth"
    PD="$RVC/assets/pretrained_v2_titan/D-f040k-TITAN.pth"
    # arm dir on ext4, sharing arm A's features via filelist paths
    EXTHOME=${RVC_DATA_DIR:-$HOME/rvc_data}/$EXP
    mkdir -p "$EXTHOME/logs"
    rm -rf "$RVC/logs/$EXP"
    ln -sfn "$EXTHOME/logs" "$RVC/logs/$EXP"
    cp "$RVC/logs/honoka_v2/config.json" "$RVC/logs/$EXP/config.json"
    cp "$RVC/logs/honoka_v2/filelist.txt" "$RVC/logs/$EXP/filelist.txt"
    ;;
  *) echo "unknown arm: $ARM" >&2; exit 1 ;;
esac

for f in "$PG" "$PD"; do
    [ -f "$f" ] || { echo "FATAL: missing pretrain $f" >&2; exit 1; }
done

cd "$RVC"
echo "=== honoka v2 TRAIN arm=$ARM exp=$EXP ==="
echo "EPOCHS=200 SAVE_EVERY=20 BATCH=16  (select best from e100-200)"
echo "pretrainG=$PG"
echo "pretrainD=$PD"
echo ""

$PYTHON -u infer/modules/train/train.py \
    -se 20 -te 200 -bs 16 -sr 40k -f0 1 -l 0 -sw 0 \
    -e "$EXP" -v v2 -c 0 \
    -pg "$PG" -pd "$PD"

echo "Training complete ($ARM)!"
