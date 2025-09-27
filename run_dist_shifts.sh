#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local DATASETS=("${!1}")
  local MODALITIES=("${!2}")
  local TAG=$3

  for ds in "${DATASETS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
      CONFIG="config_dvm_TIP_input_${mod}"
      EXP="latent_augmentation_results/input_only/TIP_${ds}_${mod}"
      LOG="error_logs/TIP_${ds}_${mod}.log"
      #
      echo ">>> Running $TAG: dataset=$ds, modality=$mod"
      CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --config-name "$CONFIG" \
        dataset="shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_${ds}" \
        exp_name="$EXP" \
        evaluate=True \
        pretrain=True
        2> "$LOG"
      echo ">>> Finished $TAG: dataset=$ds, modality=$mod"
    done
  done
}
#
## Experiment 1
#DATASETS1=("black" "miles" "normal")
#MODALITIES1=("image")
#AUGS1=("noise")
#run_experiment DATASETS1[@] MODALITIES1[@] AUGS1[@] "EXP1"


# Experiment 2
DATASETS2=("black" "miles" "normal" "color_miles")
MODALITIES2=("both" "image_only" "tabular_only" "nothing")
run_experiment DATASETS2[@] MODALITIES2[@] "TIP"
