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
CONFIGS=("config_dvm_STiL_input_nothing_latent_all_noise.yaml" "config_dvm_STiL_input_nothing_latent_image_noise.yaml" "config_dvm_STiL_input_nothing_latent_tabular_noise.yaml" "config_dvm_STiL_input_nothing_latent_multi_noise.yaml")
DATASETS=("dvm_all_server_reordered_SemiPseudo_0.1_black.yaml" "dvm_all_server_reordered_SemiPseudo_0.1_miles.yaml" "dvm_all_server_reordered_SemiPseudo_0.1_normal.yaml" "dvm_all_server_reordered_SemiPseudo_0.1_color_miles.yaml")
BATCHSIZES=(512)
DEVICE=1
REPEAT=1
RESULT_DIR="latent_augmentation_results/latent_only_before_classifier_whole"
declare -A EXTRA_PARAMS=(
  ["max_epochs"]=500
)


PRETRAIN=(FALSE)
run_experiment CONFIGS[@] DATASETS[@] PRETRAIN[@] BATCHSIZES[@] $DEVICE $REPEAT $RESULT_DIR EXTRA_PARAMS