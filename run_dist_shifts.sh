#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local CONFIGS=("${!1}")
  local DATASETS=("${!2}")
  local PRETRAIN=("${!3}")
  local DEVICE=$4
  local REPEAT=$5
  local TAG=$6

  for ((i=1; i<=$REPEAT; i++)); do
    for con in "${CONFIGS[@]}"; do
      for ds in "${DATASETS[@]}"; do
        CONFIG="config_dvm_STiL_consent"
        EXP="ADNI/baseline/${ds}"
        LOG="error_logs/${ds}.log"
        #
        echo ">>> Running $TAG: dataset=$ds, modality=$con"
        CUDA_VISIBLE_DEVICES=$DEVICE python -u run.py \
          --config-name "$CONFIG" \
          dataset="shifted_configs/ADNI/adni_${ds}" \
          exp_name="$EXP" \
          evaluate=True \
          pretrain="${PRETRAIN[@]}"
          2> "$LOG"
        echo ">>> Finished $TAG: dataset=$ds, modality=$con"
      done
    done
  done
}
#

# Experiment
CONFIGS=("config_dvm_STiL_consent")
DATASETS=("normal" "weight" "age" "TE")
DEVICE=1
REPEAT=8

PRETRAIN=(FALSE)
run_experiment CONFIGS[@] DATASETS[@] PRETRAIN[@] $DEVICE $REPEAT "STiL-all-labelled"
