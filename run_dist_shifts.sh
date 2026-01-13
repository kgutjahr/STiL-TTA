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
            EXP="${RESULT_DIR}/${con}_$(basename "$ds")_${b}"
          else
            EXP="${RESULT_DIR}/${con}_$(basename "$ds")"
          fi

          LOG="error_logs/$(basename "$ds")_${b}.log"

          EXTRA_ARGS=""
          for key in "${!PARAMS_REF[@]}"; do
            EXTRA_ARGS+=" ${key}=${PARAMS_REF[$key]}"

            # If this key is "seed", append its value to EXP
            if [[ "$key" == "seed" ]]; then
                EXP="${EXP}_seed_${PARAMS_REF[$key]}"
            fi
          done

          #
          echo ">>> Running: dataset=$ds, modality=$con"
          CUDA_VISIBLE_DEVICES=$DEVICE python -u run.py \
            --config-name "$CONFIG" \
            dataset="shifted_configs/${ds}" \
            exp_name="$EXP" \
            evaluate=True \
            pretrain="${PRETRAIN[@]}" \
            batch_size="$b" \
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
CONFIGS=("config_dvm_STiL_MoE_0.1_3" "config_dvm_STiL_MoE_0.5_3" "config_dvm_STiL_MoE_1_3" "config_dvm_STiL_MoE_2_3" "config_dvm_STiL_MoE_3_3")
DATASETS=("ADNI/adni_normal_final" "ADNI/adni_age_final" "ADNI/adni_weight_final" "ADNI/adni_TE_final")
BATCHSIZES=(32)
DEVICE=0
REPEAT=1
RESULT_DIR="ADNI/final_dataset/MoE3-linear"
declare -A EXTRA_PARAMS=(
  ["max_epochs"]=350
  ["cut_classifier_input"]=False
  ["train_logit_consent"]=False
  ["replace_ce_loss"]=False
  ["seed"]=2024
  ["linear_gate"]=True
)


PRETRAIN=(FALSE)
run_experiment CONFIGS[@] DATASETS[@] PRETRAIN[@] BATCHSIZES[@] $DEVICE $REPEAT $RESULT_DIR EXTRA_PARAMS