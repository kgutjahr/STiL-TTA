#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local CONFIGS=("${!1}")
  local DATASETS=("${!2}")
  local PRETRAIN=("${!3}")
  local DEVICE=$4
  local REPEAT=$5
  local RESULT_DIR=$6

  for ((i=1; i<=$REPEAT; i++)); do
    for con in "${CONFIGS[@]}"; do
      for ds in "${DATASETS[@]}"; do
        CONFIG="${con}"
        EXP="${RESULT_DIR}/${con}_${ds}"
        LOG="error_logs/${ds}.log"
        #
        echo ">>> Running: dataset=$ds, modality=$con"
        CUDA_VISIBLE_DEVICES=$DEVICE python -u run.py \
          --config-name "$CONFIG" \
          dataset="shifted_configs/ADNI/adni_${ds}" \
          exp_name="$EXP" \
          evaluate=True \
          pretrain="${PRETRAIN[@]}"
          2> "$LOG"
        echo ">>> Finished: dataset=$ds, modality=$con"
      done
    done
  done
}
#

# Experiment
CONFIGS=("config_dvm_STiL_consent_0" "config_dvm_STiL_consent_0.1" "config_dvm_STiL_consent_0.5" "config_dvm_STiL_consent_1" "config_dvm_STiL_consent_2" "config_dvm_STiL_consent_3" "config_dvm_STiL_consent_4")
DATASETS=("normal" "weight" "age" "TE")
DEVICE=1
REPEAT=10
RESULT_DIR="ADNI/train_consent_loss"

PRETRAIN=(FALSE)
run_experiment CONFIGS[@] DATASETS[@] PRETRAIN[@] $DEVICE $REPEAT $RESULT_DIR
