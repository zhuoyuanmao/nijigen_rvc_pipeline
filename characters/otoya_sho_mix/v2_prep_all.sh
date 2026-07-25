#!/bin/bash
# otoya_sho_mix_v2: setup + slice + F0 + ContentVec features, in one go.
# Training is launched separately so the pretrain load can be confirmed
# before committing 20h of GPU.
set -e
CHAR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$CHAR/../.." && pwd)"
RVC="${RVC_DIR:-$PROJECT/Retrieval-based-Voice-Conversion-WebUI}"
PYTHON="${PYTHON:-$PROJECT/.venv/bin/python}"
EXP=otoya_sho_mix_v2

echo "########## STEP 0: setup ##########"
bash "$CHAR/v2_step0_setup.sh"

cd "$RVC"

echo ""
echo "########## STEP 1: preprocess (slice + 40k) ##########"
$PYTHON -u infer/modules/train/preprocess.py \
    "dataset_raw/$EXP" 40000 4 "logs/$EXP" False 3.7
echo "slices: $(ls logs/$EXP/0_gt_wavs 2>/dev/null | wc -l)"

echo ""
echo "########## STEP 2: F0 (RMVPE) ##########"
$PYTHON -u infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 "logs/$EXP" True
echo "f0 files: $(ls logs/$EXP/2a_f0 2>/dev/null | wc -l)"

echo ""
echo "########## STEP 3: ContentVec features ##########"
$PYTHON -u infer/modules/train/extract_feature_print.py cuda:0 1 0 "logs/$EXP" v2 True
echo "feature files: $(ls logs/$EXP/3_feature768 2>/dev/null | wc -l)"

echo ""
echo "########## PREP COMPLETE ##########"
ls -la "logs/$EXP/" | head -12
