#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local DATASETS=("${!1}")
  local MODALITIES=("${!2}")
  local AUGS=("${!3}")
  local TAG=$4

  for ds in "${DATASETS[@]}"; do
    for mod in "${MODALITIES[@]}"; do
      for aug in "${AUGS[@]}"; do

        CONFIG="config_dvm_STiL_input_nothing_latent_${mod}_${aug}"
        EXP="latent_augmentation_results/latent_only_before_classifier_whole/${ds}_${mod}_${aug}"
        LOG="error_logs/${ds}_${mod}_${aug}.log"

        echo ">>> Running $TAG: dataset=$ds, modality=$mod, aug=$aug"
        CUDA_VISIBLE_DEVICES=1 python -u run.py \
          --config-name "$CONFIG" \
          dataset="shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_${ds}" \
          exp_name="$EXP" \
          evaluate=True \
          2> "$LOG"
        echo ">>> Finished $TAG: dataset=$ds, modality=$mod, aug=$aug"

      done
    done
  done
}

# Experiment 1
DATASETS1=("black" "miles" "normal")
MODALITIES1=("image")
AUGS1=("noise")
run_experiment DATASETS1[@] MODALITIES1[@] AUGS1[@] "EXP1"


# Experiment 2
DATASETS2=("all")
MODALITIES2=("tabular" "image" "multi")
AUGS2=("delta" "extrapolation" "mixstyle" "noise")
run_experiment DATASETS2[@] MODALITIES2[@] AUGS2[@] "EXP2"
