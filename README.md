## Requirements
This code is implemented using Python 3.9.15, PyTorch 1.11.0, PyTorch-lighting 1.6.4, CUDA 13.0, and CuDNN 8.

```sh
cd STiL/
conda env create --file environment.yaml
conda activate stil
```

## Data
Download DVM data from [here][1]

### Preparation
We conduct the same data preprocessing process as [siyi-wind/TIP](https://github.com/siyi-wind/TIP).

## Training

### Training
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01 exp_name=train evaluate=True checkpoint={YOUR_PRETRAINED_CKPT_PATH}
```

### Testing
```sh
CUDA_VISIBLE_DEVICES=0 python -u run.py --config-name config_dvm_STiL dataset=dvm_all_server_reordered_SemiPseudo_0.01 exp_name=test test=True checkpoint={YOUR_TRAINED_CKPT_PATH}
```

## Acknowledgements
We would like to thank the following repositories for their great works:
* [TIP](https://github.com/siyi-wind/TIP)
* [STiL](https://github.com/siyi-wind/STiL)




[1]: https://deepvisualmarketing.github.io/