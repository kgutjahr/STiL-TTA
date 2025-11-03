#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p error_logs

run_experiment () {
  local CONFIGS=("${!1}")
  local DATASETS=("${!2}")
  local PRETRAIN=("${!3}")
  local BATCHSIZES=("${!4}")
  local DEVICE=$5
  local REPEAT=$6
  local RESULT_DIR=$7
  declare -n PARAMS_REF=$8

  for ((i=1; i<=$REPEAT; i++)); do
    for con in "${CONFIGS[@]}"; do
      for ds in "${DATASETS[@]}"; do
        for b in "${BATCHSIZES[@]}"; do
          CONFIG="${con}"

          if (( ${#BATCHSIZES[@]} > 1 )); then
            EXP="${RESULT_DIR}/${con}_${ds}_${b}"
          else
            EXP="${RESULT_DIR}/${con}_${ds}"
          fi

          LOG="error_logs/${ds}_${b}.log"

          EXTRA_ARGS=""
          for key in "${!PARAMS_REF[@]}"; do
            EXTRA_ARGS+=" ${key}=${PARAMS_REF[$key]}"
          done

          #
          echo ">>> Running: dataset=$ds, modality=$con"
          CUDA_VISIBLE_DEVICES=$DEVICE python -u run.py \
            --config-name "$CONFIG" \
            dataset="shifted_configs/${ds}" \
            exp_name="$EXP" \
            evaluate=True \
            pretrain="${PRETRAIN[@]}" \
            $EXTRA_ARGS \
            2> "$LOG"
          echo ">>> Finished: dataset=$ds, modality=$con"
        done
      done
    done
  done
}
#

# Experiment
CONFIGS=("config_dvm_STiL_consent_0.1")
DATASETS=("configs/augment_configs/dataset/shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_normal" "configs/augment_configs/dataset/shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_black" "configs/augment_configs/dataset/shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_miles" "configs/augment_configs/dataset/shifted_configs/TIP/dvm_all_server_reordered_SemiPseudo_TIP_color_miles")
BATCHSIZES=(512)
DEVICE=1
REPEAT=10
RESULT_DIR="ADNI/final_dataset/baseline"
declare -A EXTRA_PARAMS=(
  ["train_logit_consent"]=False
  ["cut_classifier_input"]=False
  ["replace_ce_loss"]=False
)


PRETRAIN=(FALSE)
run_experiment CONFIGS[@] DATASETS[@] PRETRAIN[@] BATCHSIZES[@] $DEVICE $REPEAT $RESULT_DIR EXTRA_PARAMS
