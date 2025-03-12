'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2025
'''

from typing import Tuple, List

import torch
from torch import nn
import sys
sys.path.append('/home/siyi/project/mm/STiL')
import torch.nn.functional as F

class PrototypeLoss(torch.nn.Module):
    '''
    Push samples to the positive prototype and push away from negative prototypes
    Use label probability to define the loss
    '''
    def __init__(self, temperature, threshold) -> None:
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
    
    def forward(self, label: torch.Tensor, prototypes: torch.Tensor, feat: torch.Tensor):
        # calculate similarity between prototypes and features
        sim = torch.mm(feat, prototypes.t())/self.temperature
        sim = torch.softmax(sim, dim=1)
        log_sim = torch.log(sim + 1e-7)

        # get confident samples
        max_prob, max_id = torch.max(label, dim=1)
        conf_mask = max_prob.ge(self.threshold)
        # hard label
        hard_label = torch.zeros_like(label, device=label.device)
        hard_label[torch.arange(len(label)), max_id] = 1
        
        loss = -torch.sum(log_sim * hard_label, dim=1)
        # loss = (loss * conf_mask * max_prob).mean()
        loss = (loss * conf_mask).mean()
        return loss 
    
if __name__ == '__main__':
    label = torch.tensor([[0,1],[1,0],[0,0]])
    prototypes = F.normalize(torch.randn(2, 128))
    feat = F.normalize(torch.randn(3, 128))
    loss_func = PrototypeLoss(0.1, 0.9)
    loss = loss_func(label, prototypes, feat)
    print(loss)
