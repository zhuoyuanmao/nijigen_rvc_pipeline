#!/bin/bash
# otoya_sho_mix_v2 Step 4: TRAIN.
#
# ####################################################################
# THE headline fix of this round: -pg / -pd are now passed.
#
# The v1 round never loaded a pretrained model. step4_train.sh omitted
# -pg/-pd, RVC defaults them to "" and train.py:225 does
#     if hps.pretrainG != "": <load>
# so both nets were RANDOMLY INITIALISED. The v1 KNOWHOW's "cold start
# from f0G40k/f0D40k" was a documentation error: nothing was loaded.
# Training a VITS from scratch on 147 min is almost certainly the root
# cause of the vocoder artifacts (comb whine, breath hallucination).
# ####################################################################
#
# Run detached so it survives this shell:
#   setsid bash v2_step4_train.sh > logs/train_v2.log 2>&1 &
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RVC="${RVC_DIR:-$PROJECT/Retrieval-based-Voice-Conversion-WebUI}"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
EXP=otoya_sho_mix_v2
cd "$RVC"

PG="$RVC/assets/pretrained_v2/f0G40k.pth"
PD="$RVC/assets/pretrained_v2/f0D40k.pth"
for f in "$PG" "$PD"; do
    [ -f "$f" ] || { echo "FATAL: missing pretrain $f" >&2; exit 1; }
done

# -te 200: the warm-started model converges by ~epoch 20 (mel loss flat 14-17
# the whole run), so 300 was overkill. 200 with save_every 20 gives snapshots
# at e20,40,...,200; SELECT THE BEST CKPT FROM THE e100-200 WINDOW (early
# enough to avoid any late drift, late enough for timbre to fully settle).
# otoya_sho_mix_v2 validated this: e140 was chosen and tied/beat everything.
echo "=== $EXP Step 4: Train ==="
echo "EPOCHS=200 SAVE_EVERY=20 BATCH=16  (select best ckpt from e100-200)"
echo "pretrainG=$PG"
echo "pretrainD=$PD"
echo ""

$PYTHON -u infer/modules/train/train.py \
    -se 20 \
    -te 200 \
    -bs 16 \
    -sr 40k \
    -f0 1 \
    -l 0 \
    -sw 0 \
    -e "$EXP" \
    -v v2 \
    -c 0 \
    -pg "$PG" \
    -pd "$PD"

echo "Training complete!"
