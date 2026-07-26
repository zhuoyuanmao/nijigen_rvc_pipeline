#!/bin/bash
# otoya_sho_mix TITAN A/B arm — shares otoya_sho_mix_v2's features/filelist,
# only -pg/-pd differ (community TITAN pretrain). Config copied verbatim from
# the baseline so epochs/se/bs match exactly (300ep/se20) -> clean attribution.
# Run detached; ALWAYS confirm 'loaded pretrained ... All keys matched'.
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVC=$PROJECT/Retrieval-based-Voice-Conversion-WebUI
PYTHON=$PROJECT/.venv/bin/python
BASE=otoya_sho_mix_v2
EXP=otoya_sho_mix_v2_titan
EXTHOME=${RVC_DATA_DIR:-$HOME/rvc_data}/$EXP
PG="$RVC/assets/pretrained_v2_titan/G-f040k-TITAN.pth"
PD="$RVC/assets/pretrained_v2_titan/D-f040k-TITAN.pth"

for f in "$PG" "$PD" "$RVC/logs/$BASE/config.json" "$RVC/logs/$BASE/filelist.txt"; do
    [ -e "$f" ] || { echo "FATAL: missing $f" >&2; exit 1; }
done

# arm dir on ext4, sharing baseline's stage dirs via the copied filelist paths
mkdir -p "$EXTHOME/logs"
rm -rf "$RVC/logs/$EXP"
ln -sfn "$EXTHOME/logs" "$RVC/logs/$EXP"
cp "$RVC/logs/$BASE/config.json" "$RVC/logs/$EXP/config.json"
cp "$RVC/logs/$BASE/filelist.txt" "$RVC/logs/$EXP/filelist.txt"

# Force 200ep for this arm (user choice; A/B selects from the e100-200 window
# anyway, so 200 suffices and aligns with the current recipe). Override the
# copied config so LR decay matches -te.
$PYTHON -c "import json;p='$RVC/logs/$EXP/config.json';c=json.load(open(p));c['train']['epochs']=200;json.dump(c,open(p,'w'),indent=2)"
EP=200
SE=$($PYTHON -c "import json;print(json.load(open('$RVC/logs/$EXP/config.json'))['train']['save_every_epoch'])")

cd "$RVC"
echo "=== $EXP TRAIN (TITAN arm) — EPOCHS=$EP SE=$SE (matched to $BASE) ==="
echo "pretrainG=$PG"; echo "pretrainD=$PD"; echo ""

$PYTHON -u infer/modules/train/train.py \
    -se "$SE" -te "$EP" -bs 16 -sr 40k -f0 1 -l 0 -sw 0 \
    -e "$EXP" -v v2 -c 0 \
    -pg "$PG" -pd "$PD"

echo "Training complete (otoya TITAN arm)!"
