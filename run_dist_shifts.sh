#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_image_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_50_image_only_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_image_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_50_image_only_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_image_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_normal exp_name=normal_50_image_only_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_tabular_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_50_table_only_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_tabular_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_50_table_only_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_tabular_only dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_normal exp_name=normal_50_table_only_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_nothing dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_50_nothing_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_nothing dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_50_nothing_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL_augment_nothing dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_normal exp_name=normal_50_nothing_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_50_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_50_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_normal exp_name=normal_50_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_labels exp_name=labels_50_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_labels_black.yaml exp_name=test_label_black_50_0.1 test=True checkpoint=/mnt/data/kgutjahr/results/test/runs/eval/labels_0.1_dvm_0729_1517/checkpoint_best_acc.ckpt
