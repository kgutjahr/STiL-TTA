import os 
from os.path import join

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from torch.utils.data.sampler import WeightedRandomSampler
from torch import cuda
import pandas as pd
import numpy as np

from datasets.ImageDataset import ImageDataset
from datasets.StrongWeakImageDataset import StrongWeakImageDataset
from datasets.TabularDataset import TabularDataset
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.StrongWeakImagingAndTabularDataset import StrongWeakImagingAndTabularDataset
from utils.utils import grab_arg_from_checkpoint, grab_image_augmentations, grab_hard_eval_image_augmentations, grab_wids, create_logdir

def load_datasets(hparams):
    if hparams.eval_datatype=='imaging':
        train_dataset = ImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task,
                                        dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
        val_dataset = ImageDataset(hparams.data_val_eval_imaging, hparams.labels_val_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                    dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
    elif hparams.eval_datatype == 'tabular':
        train_dataset = TabularDataset(hparams.data_train_eval_tabular, hparams.labels_train_eval_tabular, hparams.eval_train_augment_rate, hparams.corruption_rate, train=True, 
                                    eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                    strategy=hparams.strategy, target=hparams.target)
        val_dataset = TabularDataset(hparams.data_val_eval_tabular, hparams.labels_val_eval_tabular, hparams.eval_train_augment_rate, hparams.corruption_rate, train=False, 
                                    eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular,
                                    strategy=hparams.strategy, target=hparams.target)
        hparams.input_size = train_dataset.get_input_size()
    elif hparams.eval_datatype in set(['imaging_and_tabular', 'multimodal']):
        transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
        hparams.transform = transform.__repr__()
        train_dataset = ImagingAndTabularDataset(
                    hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
                    hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, train=True, target=hparams.target, corruption_rate=hparams.corruption_rate, augmentation_speedup=hparams.augmentation_speedup,)
        val_dataset = ImagingAndTabularDataset(
                    hparams.data_val_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, hparams.data_val_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
                    hparams.labels_val_eval_imaging, hparams.img_size, hparams.live_loading, train=False, target=hparams.target, corruption_rate=hparams.corruption_rate, augmentation_speedup=hparams.augmentation_speedup,)
        hparams.input_size = train_dataset.get_input_size()
    else:
        raise Exception('argument dataset must be set to imaging, tabular, multimodal or imaging_and_tabular')
    return train_dataset, val_dataset


def load_datasets_separate(hparams):
    '''Create a combined dataloader, get labelled and unlabelled data separately'''
    # If the algorithm is CoMatch, then we need to have two strong augmentations
    two_strong = True if hparams.algorithm_name == 'CoMatch' else False
    if hparams.eval_datatype == 'imaging':
        labelled_dataset = ImageDataset(hparams.data_train_eval_imaging, hparams.labels_train_eval_imaging, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task,
                                        dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup, return_index=True)
        unlabelled_dataset = StrongWeakImageDataset(hparams.data_train_eval_imaging_unlabelled, hparams.labels_train_eval_imaging_unlabelled, hparams.delete_segmentation, hparams.eval_train_augment_rate, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=True, live_loading=hparams.live_loading, task=hparams.task,
                                    dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup, two_strong=two_strong, sweep=hparams.sweep)
        if hparams.algorithm_name == 'SimMatch':
            hparams.K = len(labelled_dataset)
    elif hparams.eval_datatype == 'imaging_and_tabular':
        if hparams.algorithm_name in set(['CoMatch', 'SimMatch', 'FreeMatch']):
            labelled_dataset = ImagingAndTabularDataset(
                hparams.data_train_eval_imaging, hparams.delete_segmentation, hparams.augmentation_rate, hparams.data_train_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
                hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, train=True, target=hparams.target, corruption_rate=hparams.corruption_rate, augmentation_speedup=hparams.augmentation_speedup,return_index=True)
            unlabelled_dataset = StrongWeakImagingAndTabularDataset(
                hparams.data_train_eval_imaging_unlabelled, hparams.delete_segmentation, hparams.augmentation_rate, hparams.data_train_eval_tabular_unlabelled, hparams.field_lengths_tabular, hparams.eval_one_hot,
                hparams.labels_train_eval_imaging_unlabelled, hparams.img_size, hparams.live_loading, train=True, target=hparams.target, corruption_rate=hparams.corruption_rate, augmentation_speedup=hparams.augmentation_speedup, two_strong=two_strong)
            if hparams.algorithm_name == 'SimMatch':
                hparams.K = len(labelled_dataset)
        else:
            transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
            hparams.transform = transform.__repr__()
            labelled_dataset = ContrastiveImagingAndTabularDataset(
                    hparams.data_train_eval_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, hparams.data_train_eval_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
                    hparams.labels_train_eval_imaging, hparams.img_size, hparams.live_loading, hparams.target, hparams.augmentation_speedup, labelled=True)
            unlabelled_dataset = ContrastiveImagingAndTabularDataset(
                    hparams.data_train_eval_imaging_unlabelled, hparams.delete_segmentation, transform, hparams.augmentation_rate, hparams.data_train_eval_tabular_unlabelled, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
                    hparams.labels_train_eval_imaging_unlabelled, hparams.img_size, hparams.live_loading, hparams.target, hparams.augmentation_speedup, labelled=False, sweep=hparams.sweep)
    l_N, u_N = len(labelled_dataset), len(unlabelled_dataset)
    # As number of labelled and unlabelled data are different, calculate the repeat time for labelled data. This will be used for prototype calculation
    hparams.repeat_ratio = max(u_N//(hparams.unlabelled_ratio*l_N)-1, 1)
    l_batch_size = hparams.batch_size//(1+hparams.unlabelled_ratio)
    u_batch_size = hparams.batch_size - l_batch_size
    l_loader = DataLoader(
        labelled_dataset, num_workers=hparams.num_workers, batch_size=l_batch_size, pin_memory=True, shuffle=True, persistent_workers=True)
    u_loader = DataLoader(unlabelled_dataset, num_workers=hparams.num_workers, batch_size=u_batch_size, pin_memory=True, shuffle=True, persistent_workers=True)
    print(f'Repeat ratio: {hparams.repeat_ratio}. Load labelled and unlabelled data separately, Labelled data: {len(l_loader)}, Unlabelled data: {len(u_loader)}')
    return l_loader, u_loader
  

def evaluate(hparams, wandb_logger):
    """
    Evaluates trained contrastive models. 
    
    IN
    hparams:      All hyperparameters
    wandb_logger: Instantiated weights and biases logger
    """
    pl.seed_everything(hparams.seed)
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    trainer_accelerator = "gpu" if cuda_visible_devices else "cpu"
    cuda_visible_devices = int(cuda_visible_devices) if cuda_visible_devices else None
    
    train_dataset, val_dataset = load_datasets(hparams)
    
    drop = ((len(train_dataset)%hparams.batch_size)==1)

    sampler = None
    if hparams.weights:
        print('Using weighted random sampler(')
        weights_list = [hparams.weights[int(l)] for l in train_dataset.labels]
        sampler = WeightedRandomSampler(weights=weights_list, num_samples=len(weights_list), replacement=True)
  
    if hparams.strategy == 'semisl':
        l_loader, u_loader = load_datasets_separate(hparams)
        train_loader = {'l': l_loader, 'u': u_loader}
        print(f"Number of training batches: {len(u_loader)}")
    else:
        train_loader = DataLoader(
        train_dataset,
        num_workers=hparams.num_workers, batch_size=hparams.batch_size, sampler=sampler,
        pin_memory=True, shuffle=(sampler is None), drop_last=drop, persistent_workers=True)
        print(f"Number of training batches: {len(train_loader)}")
  
    print(f'Train shuffle is: {sampler is None}')

    val_loader = DataLoader(
        val_dataset,
        num_workers=hparams.num_workers, batch_size=hparams.batch_size,
        pin_memory=True, shuffle=False, persistent_workers=True)
  
  
    print(f"Number of validation batches: {len(val_loader)}")
    print(f'Valid batch size: {hparams.batch_size*cuda.device_count()}')

    logdir = create_logdir('eval', hparams.resume_training, wandb_logger)
    hparams.logdir = logdir


    if hparams.algorithm_name == 'STiL':
        from models.Disentangle.STiLModel import STiLModel
        model = STiLModel(hparams)
    elif hparams.algorithm_name == 'STiL_SAINT':
        from models.Disentangle.STiLModel_SAINT import STiLModel
        model = STiLModel(hparams)
    elif hparams.algorithm_name == 'MMatch':
        from models.SemiMultimodal.MMatch import MMatch
        model = MMatch(hparams)
    elif hparams.algorithm_name == 'SimMatch':
        from models.MatchModel.SimMatch import SimMatch 
        model = SimMatch(hparams)
    elif hparams.algorithm_name == 'CoMatch':
        from models.MatchModel.CoMatch import CoMatch
        model = CoMatch(hparams)
    elif hparams.algorithm_name == 'FreeMatch':
        from models.MatchModel.FreeMatchFolder.FreeMatch import FreeMatch
        model = FreeMatch(hparams)
    elif hparams.algorithm_name == 'CoTrain_Pseudo':
        from models.SemiMultimodal.CoTraining import CoTraining
        model = CoTraining(hparams)
    elif hparams.algorithm_name == 'CoTrain_Pseudo_SAINT':
        from models.SemiMultimodal.CoTraining_SAINT import CoTraining
        model = CoTraining(hparams)
    else:
        print('Algorithm name not found')
  
    mode = 'max'
  
    callbacks = []
    callbacks.append(ModelCheckpoint(monitor=f'eval.val.{hparams.eval_metric}', mode=mode, filename=f'checkpoint_best_{hparams.eval_metric}', dirpath=logdir))
    scale = 40 if hparams.sweep==True else 100
    callbacks.append(EarlyStopping(monitor=f'eval.val.{hparams.eval_metric}', min_delta=0.0001, patience=hparams.early_stop_patience, verbose=False, mode=mode))
    if hparams.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    trainer = Trainer.from_argparse_args(hparams, accelerator=trainer_accelerator, gpus=cuda_visible_devices, callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, val_check_interval=hparams.val_check_interval, limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, limit_test_batches=hparams.limit_test_batches)
    trainer.fit(model, train_loader, val_loader)
    eval_df = pd.DataFrame(trainer.callback_metrics, index=[0])
    eval_df.to_csv(join(logdir, 'eval_results.csv'), index=False)
    
    # Get the best epoch
    wandb_logger.log_metrics({f'best.val.{hparams.eval_metric}': model.best_val_score})

    if hparams.test_and_eval:
        if hparams.eval_datatype == 'imaging':
            test_dataset = ImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task=hparams.task,
                                        dataset_name=hparams.dataset_name, augmentation_speedup=hparams.augmentation_speedup)
            hparams.transform_test = test_dataset.transform_val.__repr__()
        elif hparams.eval_datatype in set(['multimodal', 'imaging_and_tabular']):
            test_dataset = ImagingAndTabularDataset(
                    hparams.data_test_eval_imaging, hparams.delete_segmentation, 0, hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot,
                    hparams.labels_test_eval_imaging, grab_arg_from_checkpoint(hparams, 'img_size'), hparams.live_loading, train=False, target=hparams.target, corruption_rate=0,
                    augmentation_speedup=hparams.augmentation_speedup)
            hparams.input_size = test_dataset.get_input_size()
        elif hparams.eval_datatype == 'tabular':
            test_dataset = TabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, 0, 0, train=False, 
                                        eval_one_hot=hparams.eval_one_hot, field_lengths_tabular=hparams.field_lengths_tabular, data_base=hparams.data_base, 
                                        strategy=hparams.strategy,target=hparams.target)
            hparams.input_size = test_dataset.get_input_size()
        else:
            raise Exception('argument dataset must be set to imaging, tabular or multimodal')
    
    drop = ((len(test_dataset)%hparams.batch_size)==1)

    test_loader = DataLoader(
      test_dataset,
      num_workers=hparams.num_workers, batch_size=512,  
      pin_memory=True, shuffle=False, drop_last=drop, persistent_workers=True)
  
    print(f"Number of testing batches: {len(test_loader)}")

    model.freeze()

    trainer = Trainer.from_argparse_args(hparams, accelerator=trainer_accelerator, gpus=cuda_visible_devices, logger=wandb_logger)
    test_results = trainer.test(model, test_loader, ckpt_path=os.path.join(logdir,f'checkpoint_best_{hparams.eval_metric}.ckpt'))
    df = pd.DataFrame(test_results)
    df.to_csv(join(logdir, 'test_results.csv'), index=False)
