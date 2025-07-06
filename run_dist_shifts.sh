#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_black exp_name=black_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_black exp_name=black_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_b-s-b-g-w-r exp_name=b-s-b-g-w-r_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_b-s-b-g-w-r exp_name=b-s-b-g-w-r_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_b-s-g exp_name=b-s-g_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_b-s-g exp_name=b-s-g_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_gearbox exp_name=gearbox_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_gearbox exp_name=gearbox_0.01 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.1_price exp_name=price_0.1 evaluate=True

CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01_price exp_name=price_0.01 evaluate=True
