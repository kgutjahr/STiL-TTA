#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local DATASETS=("${!1}")
  local MODALITIES=("${!2}")
  local PRETRAIN=("${!3}")
  local TAG=$4

  for ds in "${DATASETS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
      CONFIG="config_dvm_STiL_consent"
      EXP="latent_augmentation_results/input_only/STiL_all_labelled_${ds}_${mod}"
      LOG="error_logs/TIP_${ds}_${mod}.log"
      #
      echo ">>> Running $TAG: dataset=$ds, modality=$mod"
      CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --config-name "$CONFIG" \
        dataset="shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_${ds}" \
        exp_name="$EXP" \
        evaluate=True \
        pretrain="${PRETRAIN[@]}"
        2> "$LOG"
      echo ">>> Finished $TAG: dataset=$ds, modality=$mod"
    done
  done
}
#

# Experiment
DATASETS2=("black" "miles" "normal" "color_miles")
MODALITIES2=("")
PRETRAIN=(FALSE)
run_experiment DATASETS2[@] MODALITIES2[@] PRETRAIN[@] "STiL-all-labelled"
