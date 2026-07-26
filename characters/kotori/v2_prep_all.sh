#!/bin/bash
# kotori_v2: setup + slice + F0 + features + score/select(180) + filelist.
# Training launched separately (pretrain-load confirmation gate).
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHAR=$PROJECT/characters/kotori
RVC=$PROJECT/Retrieval-based-Voice-Conversion-WebUI
PYTHON=$PROJECT/.venv/bin/python
EXP=kotori_v2
TARGET_MIN="${TARGET_MIN:-180}"

echo "########## STEP 0: setup ##########"
bash "$CHAR/v2_step0_setup.sh"

cd "$RVC"
echo ""; echo "########## STEP 1: preprocess ##########"
$PYTHON -u infer/modules/train/preprocess.py "dataset_raw/$EXP" 40000 4 "logs/$EXP" False 3.7
echo "slices: $(ls logs/$EXP/0_gt_wavs 2>/dev/null | wc -l)"

echo ""; echo "########## STEP 2: F0 (RMVPE) ##########"
$PYTHON -u infer/modules/train/extract/extract_f0_rmvpe.py 1 0 0 "logs/$EXP" True

echo ""; echo "########## STEP 3: features ##########"
$PYTHON -u infer/modules/train/extract_feature_print.py cuda:0 1 0 "logs/$EXP" v2 True
echo "features: $(ls logs/$EXP/3_feature768 2>/dev/null | wc -l)"

echo ""; echo "########## STEP 3a: score + select (${TARGET_MIN}min) ##########"
cd "$CHAR"
$PYTHON -u _score_slices.py --exp "$EXP" --target-min "$TARGET_MIN"

echo ""; echo "########## STEP 3b: filelist (selected only) ##########"
$PYTHON -u v2_step3b_filelist.py --exp "$EXP" --keep-list "$RVC/logs/$EXP/selected_slices.txt"
echo ""; echo "########## PREP COMPLETE ##########"
