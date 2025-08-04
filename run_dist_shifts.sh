#!/bin/bash

#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_normal exp_name=normal_0.1 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_normal exp_name=normal_0.01 evaluate=True
#
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_0.1 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_black exp_name=black_0.01 evaluate=True
#
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_labels exp_name=labels_0.1 evaluate=True
CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_labels_black.yaml exp_name=test_label_black test=True checkpoint=/mnt/data/kgutjahr/results/test/runs/eval/labels_0.1_dvm_0729_1517/checkpoint_best_acc.ckpt
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_labels exp_name=labels_0.01 evaluate=True
#
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_b-s-b-g-w exp_name=b-s-b-g-w_0.1 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_b-s-b-g-w exp_name=b-s-b-g-w_0.01 evaluate=True

#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_b-s-g exp_name=b-s-g_0.1 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_b-s-g exp_name=b-s-g_0.01 evaluate=True

#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_adv_year exp_name=year_0.1 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_0.1 evaluate=True
#
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_adv_year exp_name=year_0.01 evaluate=True
#CUDA_VISIBLE_DEVICES=1 python -u run.py --config-name config_dvm_STiL dataset=shifted_configs/dvm_all_server_reordered_SemiPseudo_0.01_miles exp_name=miles_0.01 evaluate=True
