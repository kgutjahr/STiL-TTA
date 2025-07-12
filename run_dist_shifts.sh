#!/bin/bash

#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_0.1 evaluate=True
#
#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_black exp_name=black_0.01 evaluate=True
#
#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_b-s-b-g-w exp_name=b-s-b-g-w_0.1 evaluate=True
#
#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_b-s-b-g-w exp_name=b-s-b-g-w_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_b-s-g exp_name=b-s-g_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_b-s-g exp_name=b-s-g_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_year exp_name=year_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_year exp_name=year_0.01 evaluate=True

#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_miles exp_name=miles_0.1 evaluate=True
#
#CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_miles exp_name=miles_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_small exp_name=small_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_small exp_name=small_0.01 evaluate=True
