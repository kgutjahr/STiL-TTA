from typing import List, Tuple
import random
import csv
import copy
from os.path import join

import torch
from torch.utils.data import Dataset
import numpy as np

class TabularDataset(Dataset):
  """"
  Dataset for the evaluation of tabular data
  """
  def __init__(self, data_path: str, labels_path: str, eval_train_augment_rate: float, corruption_rate:float, train: bool, eval_one_hot: bool, field_lengths_tabular: str,
               strategy:str='tip', augmentation_speedup: bool=False, target: str=None
               ):
    super(TabularDataset, self).__init__()
    self.data = self.read_and_parse_csv(data_path)
    self.raw_data = np.array(self.data)
    self.labels = torch.load(labels_path)
    self.eval_one_hot = eval_one_hot
    self.field_lengths = torch.load(field_lengths_tabular)
    self.generate_marginal_distributions()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.c = corruption_rate
    self.strategy = strategy

    assert len(self.labels) == len(self.data)


  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths))
    else:
      return len(self.field_lengths) #len(self.data[0])
  
  def read_and_parse_csv(self, path: str):
    """
    Does what it says on the box
    """
    with open(path,'r') as f:
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
    data = np.array(self.data)
    self.marginal_distributions = np.transpose(data)
    

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
  

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(torch.clamp(subject[i],min=0,max=self.field_lengths[i]-1).long(), num_classes=int(self.field_lengths[i])))
    return torch.cat(out)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if self.train and random.random() < self.eval_train_augment_rate:
      tab = torch.tensor(self.corrupt(self.data[index]), dtype=torch.float)
    else:
      tab = torch.tensor(self.data[index], dtype=torch.float)

    if self.eval_one_hot:
      tab = self.one_hot_encode(tab)
      
    label = torch.tensor(self.labels[index], dtype=torch.long)

    return (tab), label

  def __len__(self) -> int:
    return len(self.data)


if __name__ == '__main__':
  dataset = TabularDataset(data_path='/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final/cardiac_features_train_imputed_noOH_tabular_imaging_CAD_balanced_reordered.csv',
                           eval_train_augment_rate=1.0, labels_path='/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final/cardiac_labels_CAD_train_balanced.pt',
                           corruption_rate=0.3, train=True, eval_one_hot=False,
                           field_lengths_tabular='/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final/tabular_lengths_reordered.pt',
                           strategy='comparison')
  x = dataset[2]
  print(x)