from typing import List, Tuple
import random
import csv
import copy

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import albumentations as A
import numpy as np
import os
import sys
from os.path import join

current_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(current_path)))
from utils.utils import grab_hard_eval_image_augmentations, grab_weak_image_augmentations

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


class ImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that imaging and tabular data for downstream tasks.

  The imaging view has {eval_train_augment_rate} chance of being augmented.
  The tabular view corruption rate to be augmented.
  """
  def __init__(
      self,
      data_path_imaging: str, delete_segmentation: bool, eval_train_augment_rate: float, 
      data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool,
      labels_path: str, img_size: int, live_loading: bool, train: bool, target: str,
      corruption_rate: float, augmentation_speedup: bool=False, return_index=False,
      visualization=False) -> None:

    # Imaging
    self.data_imaging = torch.load(data_path_imaging)
    self.delete_segmentation = delete_segmentation
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.augmentation_speedup = augmentation_speedup
    self.dataset_name = data_path_tabular.split('/')[-1].split('_')[0]
    self.target = target
    self.visualization = visualization
    self.return_index = return_index

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target, augmentation_speedup=augmentation_speedup)

    if augmentation_speedup:
      if self.target == 'dvm':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts)
        ])
        print('Using dvm transform for default transform')
      elif self.target == 'CAD' or self.target == 'Infarction':
        self.default_transform = A.Compose([
          A.Resize(height=img_size, width=img_size),
          A.Lambda(name='convert2tensor', image=convert_to_ts_01)
        ])
        print('Using cardiac transform for default transform in ImagingAndTabularDataset')
      else:
        raise print('Only support dvm and cardiac datasets in ImagingAndTabularDataset')
    else:
      self.default_transform = transforms.Compose([
        transforms.Resize(size=(img_size,img_size)),
        transforms.Lambda(convert_to_float)
      ])

    # Tabular
    self.data_tabular = np.array(self.read_and_parse_csv(data_path_tabular))
    self.generate_marginal_distributions()
    self.field_lengths_tabular = np.array(torch.load(field_lengths_tabular))
    self.eval_one_hot = eval_one_hot
    self.c = corruption_rate if corruption_rate else None

    # Classifier
    self.labels = torch.load(labels_path)

    self.train = train
    assert len(self.data_imaging) == len(self.data_tabular) == len(self.labels) 
  
  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def generate_marginal_distributions(self) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data = np.array(self.data_tabular)
    self.marginal_distributions = np.transpose(data)
    # data_df = pd.read_csv(data_path)
    # self.marginal_distributions = np.array(data_df.transpose().values.tolist())

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)
    subject = np.array(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      marg_dist = self.marginal_distributions[i]
      if marg_dist.size != 0:
        value = np.random.choice(marg_dist, size=1)
        subject[i] = value
    return subject

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(np.sum(self.field_lengths_tabular))
    else:
      return len(self.field_lengths_tabular)

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths_tabular[i]-1).long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    im = self.data_imaging[index]
    path = im
    if self.live_loading:
      if self.augmentation_speedup:
        im = np.load(im[:-4]+'.npy', allow_pickle=True)
      else:
        im = read_image(im)
        im = im / 255

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(image=im)['image'] if self.augmentation_speedup else self.transform_train(im)
      if self.c > 0:
        tab = torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)
      else:
        tab = torch.tensor(self.data_tabular[index], dtype=torch.float)
    else:
      im = self.default_transform(image=im)['image'] if self.augmentation_speedup else self.default_transform(im)
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    if self.eval_one_hot:
      tab = self.one_hot_encode(tab).to(torch.float)

    label = torch.tensor(self.labels[index], dtype=torch.long)


    if self.return_index:
      return (im, tab), label, index
    else:
      if self.visualization:
        return (im, tab, path), label
      else:
        return (im, tab), label
   

  def __len__(self) -> int:
    return len(self.data_tabular)
  
  
if __name__ == '__main__':
  dataset = ImagingAndTabularDataset(
    data_path_imaging='/mnt/data/kgutjahr/datasets/DVM/images/val_paths_all_views.pt', delete_segmentation=False, eval_train_augment_rate=0.8, 
          data_path_tabular='/mnt/data/kgutjahr/datasets/DVM/images/dvm_features_val_noOH_all_views_physical_jittered_50_reordered.csv', 
          field_lengths_tabular='/mnt/data/kgutjahr/datasets/DVM/images/tabular_lengths_all_views_physical_reordered.pt', eval_one_hot=False,
          labels_path='/mnt/data/kgutjahr/datasets/DVM/images/labels_model_all_val_all_views.pt', img_size=128, live_loading=True, train=True, target='dvm',
          corruption_rate=0.3, augmentation_speedup=True, 
  )
  print(dataset[0][0][0].shape, dataset[0][0][1].shape, dataset[0][1].shape)

