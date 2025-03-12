'''
SimMatch pytorch lightning module
'''
from typing import List, Tuple, Dict
import sys
import time

import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.MatchModel.simmatch_model import SimMatchModel


class SimMatch(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = SimMatchModel(hparams)
        print('Use SimMatch.py')
        nclasses = hparams.num_classes

        self.initialize_metrics(nclasses, nclasses)
    
        self.best_val_score = 0
        self.criterion = nn.CrossEntropyLoss()
        self.threshold = hparams.sim_threshold
        self.lambda_u = hparams.lambda_u
        self.lambda_in = hparams.lambda_in
        self.use_pseudo = False
        self.start_epoch = hparams.start_epoch
        
        print(f'SimMatch threshold: {self.threshold}, lambda_u: {self.lambda_u}, lambda_in: {self.lambda_in}')

        print(f'Model backbone: {self.model.main}')


    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    
    def initialize_metrics(self, nclasses_train, nclasses_val):

        # classification metrics
        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
        
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_train_unlabelled = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_train_pseudo = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_unlabelled = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_pseudo = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        current_epoch = self.current_epoch
        batch_l, batch_u = batch['l'], batch['u']
        images_x, targets_x, index = batch_l 
        images_u, targets_u = batch_u
        B_l, B_u = len(targets_x), len(targets_u)
        images_u_w, images_u_s = images_u


        logits_x, pseudo_label, logits_u_s, loss_in = self.model(images_x, images_u_w, images_u_s, labels=targets_x, index=index, start_unlabel=True)
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.threshold).float()

        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss_u = (torch.sum(-F.log_softmax(logits_u_s,dim=1) * pseudo_label.detach(), dim=1) * mask).mean()
        loss_in = loss_in.mean()

        if current_epoch <= self.start_epoch:
            loss = loss_x
        else:
            loss = loss_x + self.lambda_u * loss_u + self.lambda_in * loss_in

        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.threshold1_ratio", torch.sum(mask)/len(mask), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        
        with torch.no_grad():
            prob_x = torch.softmax(logits_x.detach(), dim=1)
            prob_u = torch.softmax(logits_u_s.detach(), dim=1)
        if self.hparams.num_classes==2:
            prob_x = prob_x[:,1]
            prob_u = prob_u[:,1]
            pseudo_label = pseudo_label[:,1]
        
        self.acc_train(prob_x, targets_x)
        self.auc_train(prob_x, targets_x)
        self.acc_train_unlabelled(prob_u, targets_u)
        self.auc_train_unlabelled(prob_u, targets_u)

        # Comment. May cause ddp stuck
        # if torch.sum(mask) > 0:
        #     self.use_pseudo = True
        #     mask = mask.bool()
        #     self.acc_train_pseudo(pseudo_label[mask], targets_u[mask])
        #     self.auc_train_pseudo(pseudo_label[mask], targets_u[mask])
        
        return loss
        

    def training_epoch_end(self, _) -> None:
        """
        Compute training epoch metrics and check for new best values
        """
        self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
        self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)
        self.log('eval.train_unlabelled.acc', self.acc_train_unlabelled, on_epoch=True, on_step=False, metric_attribute=self.acc_train_unlabelled)
        self.log('eval.train_unlabelled.auc', self.auc_train_unlabelled, on_epoch=True, on_step=False, metric_attribute=self.auc_train_unlabelled)
        # if self.use_pseudo:
        #     self.log('eval.train_pseudo.acc', self.acc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.acc_train_pseudo)
        #     self.log('eval.train_pseudo.auc', self.auc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.auc_train_pseudo)
        #     self.use_pseudo = False
        
        print(f'Epoch {self.current_epoch}: train.acc: {self.acc_train.compute()}, train.auc: {self.auc_train.compute()}, train.acc_unlabelled: {self.acc_train_unlabelled.compute()}, train.auc_unlabelled: {self.auc_train_unlabelled.compute()}')


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)
        self.log(f"multimodal.val.loss", loss, on_epoch=True, on_step=False)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]
        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

        return loss


    def validation_epoch_end(self, _) -> None:
        """
        Compute validation epoch metrics and check for new best values
        """
        if self.trainer.sanity_checking:
            return  

        epoch_acc_val = self.acc_val.compute()
        epoch_auc_val = self.auc_val.compute()

        self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
        self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)

        print(f'Epoch {self.current_epoch}: val.acc: {epoch_acc_val}, val.auc: {epoch_auc_val}')

        self.acc_val.reset()
        self.auc_val.reset()


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Runs test step
        """
        x,y = batch
        y_hat = self.model(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """
        Test epoch end
        """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()

        self.log('test.acc', test_acc)
        self.log('test.auc', test_auc)
    
    def calc_and_log_train_embedding_acc(self, logits, labels, modality: str) -> None:
        self.top1_acc_train(logits, labels)
        self.top5_acc_train(logits, labels)
        self.log(f"{modality}.train.top1", self.top1_acc_train, on_epoch=True, on_step=False, batch_size=logits.size(0))
        self.log(f"{modality}.train.top5", self.top5_acc_train, on_epoch=True, on_step=False, batch_size=logits.size(0))

    def calc_and_log_val_embedding_acc(self, logits, labels, modality: str) -> None:
        self.top1_acc_val(logits, labels)
        self.top5_acc_val(logits, labels)
        self.log(f"{modality}.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)
        self.log(f"{modality}.val.top5", self.top5_acc_val, on_epoch=True, on_step=False)
        
    def configure_optimizers(self):
        """
        Sets optimizer and scheduler.
        Must use strict equal to false because if check_val_n_epochs is > 1
        because val metrics not defined when scheduler is queried
        """
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
        scheduler = self.initialize_scheduler(optimizer)
        return (
        { # Contrastive
            "optimizer": optimizer, 
            "lr_scheduler": scheduler
        }
        )

    def initialize_scheduler(self, optimizer: torch.optim.Optimizer):
        if self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(self.hparams.dataset_length*self.hparams.cosine_anneal_mult), eta_min=0, last_epoch=-1)
        elif self.hparams.scheduler == 'anneal':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs = self.hparams.max_epochs)
        elif self.hparams.scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
        else:
            raise ValueError('Valid schedulers are "cosine" and "anneal"')
        
        return scheduler
