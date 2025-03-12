from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image

from utils.utils import grab_strong_image_augmentations, grab_weak_image_augmentations

import numpy as np
import albumentations as A

def convert_to_float(x):
  return x.float()

def convert_to_ts(x, **kwargs):
  x = np.clip(x, 0, 255) / 255
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x

def convert_to_ts_01(x, **kwargs):
  x = torch.from_numpy(x).float()
  x = x.permute(2,0,1)
  return x


class StrongWeakImageDataset(Dataset):
  """
  Dataset for getting strong and weak augmentations of an image.
  if two_strong is True:
    - [im_weak, im_strong, im_strong], label
  else:
    - [im_weak, im_strong], label
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str,
               dataset_name:str='dvm', augmentation_speedup:bool=False, return_index:bool=False, sweep=False, two_strong=False) -> None:
    super(StrongWeakImageDataset, self).__init__()
    print('Use StrongWeakImageDataset for SemiSL')
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task
    self.return_index = return_index
    # two strong augmentation
    self.two_strong = two_strong
    self.target = target
    print(f'Two strong augmentation: {self.two_strong}')

    self.dataset_name = dataset_name
    self.augmentation_speedup = augmentation_speedup

    self.data = torch.load(data)
    self.labels = torch.load(labels)

    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.transform_strong = grab_strong_image_augmentations(img_size, target, augmentation_speedup=self.augmentation_speedup)
    self.transform_weak = grab_weak_image_augmentations(img_size, target, augmentation_speedup=self.augmentation_speedup)

    if sweep:
      min_num = min(5000, len(self.labels))
      print(f'Only use {min_num} samples for sweep')
      self.data = self.data[:min_num]
      self.labels = self.labels[:min_num]
      assert len(self.data) == len(self.labels)
    else:
      print(f'Num of data: {len(self.labels)}')

  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255
    
    ims = [self.transform_weak(image=im)['image'] if self.augmentation_speedup else self.transform_strong(im)]
    ims.append(self.transform_strong(image=im)['image'] if self.augmentation_speedup else self.transform_strong(im))
    if self.two_strong:
      ims.append(self.transform_strong(image=im)['image'] if self.augmentation_speedup else self.transform_strong(im))
    
    label = self.labels[indx]

    if self.return_index:
      return ims, label, indx
    else:
      return ims, label

  def __len__(self) -> int:
    return len(self.labels)
