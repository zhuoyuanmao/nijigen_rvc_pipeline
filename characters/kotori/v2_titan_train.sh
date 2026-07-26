#!/bin/bash
# kotori v2 TRAIN — TITAN pretrain, 200ep/se20 (Kevin adopted TITAN after honoka).
# Single run (no baseline A/B). Config/filelist come from v2_prep_all.
# ALWAYS confirm 'loaded pretrained ... All keys matched'. Run detached.
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVC=$PROJECT/Retrieval-based-Voice-Conversion-WebUI
PYTHON=$PROJECT/.venv/bin/python
EXP=kotori_v2
PG="$RVC/assets/pretrained_v2_titan/G-f040k-TITAN.pth"
PD="$RVC/assets/pretrained_v2_titan/D-f040k-TITAN.pth"

for f in "$PG" "$PD" "$RVC/logs/$EXP/config.json" "$RVC/logs/$EXP/filelist.txt"; do
    [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

cd "$RVC"
echo "=== $EXP TRAIN (TITAN) — 200ep/se20, select best e100-200 ==="
echo "pretrainG=$PG"; echo "pretrainD=$PD"; echo ""

$PYTHON -u infer/modules/train/train.py \
    -se 20 -te 200 -bs 16 -sr 40k -f0 1 -l 0 -sw 0 \
    -e "$EXP" -v v2 -c 0 \
    -pg "$PG" -pd "$PD"

echo "Training complete (kotori TITAN)!"
