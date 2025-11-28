import os 
import sys

from torch import cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.utils import grab_image_augmentations, grab_wids, create_logdir
#from utils.ssl_online_custom import SSLOnlineEvaluator

from datasets.ContrastiveImagingAndTabularDataset import ContrastiveImagingAndTabularDataset
from datasets.ContrastiveReconstructImagingAndTabularDataset import ContrastiveReconstructImagingAndTabularDataset

from models.TIP.TipModel3Loss import TIP3Loss


def load_datasets(hparams):
  if hparams.datatype == 'multimodal':
    transform = grab_image_augmentations(hparams.img_size, hparams.target, hparams.augmentation_speedup)
    hparams.transform = transform.__repr__()
    if hparams.strategy == 'tip':
      # for TIP
      train_dataset = ContrastiveReconstructImagingAndTabularDataset(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_train_tabular, hparams.corruption_rate, hparams.replace_random_rate, hparams.replace_special_rate,
        hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
      val_dataset = ContrastiveReconstructImagingAndTabularDataset(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_val_tabular, hparams.corruption_rate,  hparams.replace_random_rate, hparams.replace_special_rate, 
        hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
    else:
      # for MMCL
      train_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_train_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_train_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_train, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
      val_dataset = ContrastiveImagingAndTabularDataset(
        hparams.data_val_imaging, hparams.delete_segmentation, transform, hparams.augmentation_rate, 
        hparams.data_val_tabular, hparams.corruption_rate, hparams.field_lengths_tabular, hparams.one_hot,
        hparams.labels_val, hparams.img_size, hparams.live_loading, hparams.augmentation_speedup)
    hparams.input_size = train_dataset.get_input_size()
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return train_dataset, val_dataset


def select_model(hparams, train_dataset):
  if hparams.datatype == 'multimodal':
    if hparams.strategy == 'tip':
      # TIP
      model = TIP3Loss(hparams)
      print('Using TIP3Loss')
  else:
    raise Exception(f'Unknown datatype {hparams.datatype}')
  return model


def pretrain(hparams, wandb_logger):
  """
  Train code for pretraining or supervised models. 
  
  IN
  hparams:      All hyperparameters
  wandb_logger: Instantiated weights and biases logger
  """
  pl.seed_everything(hparams.seed)

  # Load appropriate dataset
  train_dataset, val_dataset = load_datasets(hparams)

  
  train_loader = DataLoader(
    train_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=True, persistent_workers=True)

  val_loader = DataLoader(
    val_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True)


  print(f"Number of training batches: {len(train_loader)}")
  print(f"Number of validation batches: {len(val_loader)}")
  print(f'Valid batch size: {hparams.batch_size*cuda.device_count()}')

  # Create logdir based on WandB run name
  logdir = create_logdir(hparams.datatype, hparams.resume_training, wandb_logger)
  
  model = select_model(hparams, train_dataset)
  
  callbacks = []

  #if hparams.online_mlp:
  #  model.hparams.classifier_freq = float('Inf')
  #  z_dim =  hparams.multimodal_embedding_dim if hparams.strategy=='tip' else model.pooled_dim
  #  callbacks.append(SSLOnlineEvaluator(z_dim = z_dim, hidden_dim = hparams.embedding_dim, num_classes = hparams.num_classes, swav = False, multimodal = (hparams.datatype=='multimodal'), 
  #                                      strategy=hparams.strategy))
  callbacks.append(ModelCheckpoint(filename='checkpoint_last_epoch_{epoch:02d}', dirpath=logdir, save_on_train_epoch_end=True, auto_insert_metric_name=False))
  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  trainer = Trainer.from_argparse_args(hparams, gpus=cuda.device_count(), 
                                       callbacks=callbacks, logger=wandb_logger, max_epochs=hparams.max_epochs, check_val_every_n_epoch=hparams.check_val_every_n_epoch, 
                                       limit_train_batches=hparams.limit_train_batches, limit_val_batches=hparams.limit_val_batches, enable_progress_bar=hparams.enable_progress_bar)

  if hparams.resume_training:
    trainer.fit(model, train_loader, val_loader, ckpt_path=hparams.checkpoint)
  else:
    trainer.fit(model, train_loader, val_loader)