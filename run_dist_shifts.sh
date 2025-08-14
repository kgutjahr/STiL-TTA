#!/bin/bash

# Define parameters
DATASETS=("black" "miles" "normal")
MODALITIES=("image" "tabular" "multi")
AUGS=("delta" "extrapolation" "mixstyle" "noise")

# Create log directory if it doesn't exist
mkdir -p error_logs

# Loop over all combinations
for ds in "${DATASETS[@]}"; do
  for mod in "${MODALITIES[@]}"; do
    for aug in "${AUGS[@]}"; do

      CONFIG="augment_configs/config_dvm_STiL_input_nothing_latent_${mod}_${aug}"
      EXP="/latent_augmentation_results/latent_only/${ds}_${mod}_${aug}"
      LOG="error_logs/${ds}_${mod}_${aug}.log"

      CUDA_VISIBLE_DEVICES=1 python -u run.py \
        --config-name "$CONFIG" \
        dataset="shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_${ds}" \
        exp_name="$EXP" \
        evaluate=True \
        2> "$LOG"

    done
  done
done