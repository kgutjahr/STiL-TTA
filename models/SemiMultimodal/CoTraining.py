'''
CoTraining pytorch lightning module
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

# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from models.SemiMultimodal.Multimodal_model import MultimodalBackbone


class CoTraining(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = MultimodalBackbone(hparams)
        print('Use CoTraining.py')
        nclasses = hparams.num_classes

        self.initialize_metrics(nclasses, nclasses)
    
        self.best_val_score = 0
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = hparams.alpha
        self.rate_uce = hparams.rate_uce
        self.threshold = hparams.co_threshold
        self.use_pseudo_i = False
        self.use_pseudo_t = False
        self.start_epoch = hparams.start_epoch
        
        print(f'CoTraining threshold: {self.threshold}, alpha: {self.alpha}, rate_uce: {self.rate_uce}')

        # teacher model
        self.use_ema = self.hparams.use_ema
        if self.use_ema:
            print('Use EMA as teacher model')
            self.eman = self.hparams.eman
            self.momentum = self.hparams.ema_momentum
            self.ema = MultimodalBackbone(self.hparams)
            for param_model, param_ema in zip(self.model.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_model.data)
                param_ema.requires_grad = False

        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.use_ddp = torch.cuda.device_count() > 1

        print(f'Model backbone: {self.model}')


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
        self.acc_train_pseudo_i = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_train_pseudo_t = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_imaging = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_unlabelled = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_pseudo_i = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_pseudo_t = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_imaging = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_tabular = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)


    @torch.no_grad()
    def momentum_update_ema(self):
        if self.eman:
            state_dict_main = self.model.state_dict()
            state_dict_ema = self.ema.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                if 'num_batches_tracked' in k_ema:
                    v_ema.data.copy_(v_main.data)
                else:
                    v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
                    # v_ema.data.mul_(self.momentum).add_((1. - self.momentum) * v_main.data)
        else:
            for param_q, param_k in zip(self.model.parameters(), self.ema.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        current_epoch = self.current_epoch
        batch_l, batch_u = batch['l'], batch['u']
        im_views_l, tab_views_l, y_l, _, label_identify_l = batch_l
        im_views_u, tab_views_u, y_u, _, label_identify_u = batch_u
        B_l, B_u = len(y_l), len(y_u)
        assert torch.sum(label_identify_l) == len(label_identify_l)
        assert torch.sum(label_identify_u) == 0

        y_hat_m, y_hat_i, y_hat_t, _ = self.model.forward([torch.cat((im_views_l[1], im_views_u[1])), torch.cat((tab_views_l[1], tab_views_u[1]))])
        prob_m, prob_i, prob_t = torch.softmax(y_hat_m.detach(), dim=1), torch.softmax(y_hat_i.detach(), dim=1), torch.softmax(y_hat_t.detach(), dim=1)
        prob_m_l, prob_m_u = prob_m[:B_l], prob_m[B_l:]

        if self.use_ema:
            self.ema.eval()
        with torch.no_grad():
            if self.use_ema:
                self.momentum_update_ema()
                y_hat_m_e, y_hat_i_e, y_hat_t_e, _ = self.ema.forward([torch.cat((im_views_l[1], im_views_u[1])), torch.cat((tab_views_l[1], tab_views_u[1]))])
            else:
                y_hat_m_e, y_hat_i_e, y_hat_t_e = y_hat_m.clone(), y_hat_i.clone(), y_hat_t.clone()

        # supervised loss
        loss_ce = self.criterion_ce(y_hat_m[:B_l], y_l) + self.criterion_ce(y_hat_i[:B_l], y_l) + self.criterion_ce(y_hat_t[:B_l], y_l)

        # unsupervised loss
        pseudo_label_i = torch.softmax(y_hat_i_e[B_l:].detach(), dim=1)
        pseudo_label_t = torch.softmax(y_hat_t_e[B_l:].detach(), dim=1)
        max_prob_i, _ = torch.max(pseudo_label_i, dim=1)
        max_prob_t, _ = torch.max(pseudo_label_t, dim=1)
        mask_i = max_prob_i.ge(self.threshold)
        mask_t = max_prob_t.ge(self.threshold)

        loss_i_u = (F.cross_entropy(y_hat_i[B_l:], pseudo_label_t, reduction='none')*mask_t).mean()
        loss_t_u = (F.cross_entropy(y_hat_t[B_l:], pseudo_label_i, reduction='none')*mask_i).mean()

        self.log(f"multimodal.train.CEloss_unlabelled_i", loss_i_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CEloss_unlabelled_t", loss_t_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.threshold_i_ratio', torch.sum(mask_i)/len(mask_i), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.threshold_t_ratio', torch.sum(mask_t)/len(mask_t), on_epoch=True, on_step=False, batch_size=B_l+B_u)

        if current_epoch > self.start_epoch:
            loss = self.alpha*loss_ce + self.rate_uce*(loss_i_u + loss_t_u)
        else:
            loss = self.alpha*loss_ce
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False, batch_size=B_l+B_u)

        if self.hparams.num_classes==2:
            prob_m_l = prob_m_l[:,1]
            prob_m_u = prob_m_u[:,1]
            pseudo_label_i = pseudo_label_i[:,1]
            pseudo_label_t = pseudo_label_t[:,1]
        self.acc_train(prob_m_l, y_l)
        self.auc_train(prob_m_l, y_l)
        self.acc_train_unlabelled(prob_m_u, y_u)
        self.auc_train_unlabelled(prob_m_u, y_u)
        
        return loss
        

    def on_train_epoch_end(self) -> None:
        """
        Compute training epoch metrics and check for new best values
        """
        self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
        self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)
        self.log('eval.train_unlabelled.acc', self.acc_train_unlabelled, on_epoch=True, on_step=False, metric_attribute=self.acc_train_unlabelled)
        self.log('eval.train_unlabelled.auc', self.auc_train_unlabelled, on_epoch=True, on_step=False, metric_attribute=self.auc_train_unlabelled)
        
        print(f'Epoch {self.current_epoch}: train.acc: {self.acc_train.compute()}, train.auc: {self.auc_train.compute()}, train.acc_unlabelled: {self.acc_train_unlabelled.compute()}, train.auc_unlabelled: {self.auc_train_unlabelled.compute()}')


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        x, y = batch
        # use augmented image and tabular views
        y_hat, y_i_hat, y_t_hat, x_m = self.model.forward(x)
        # =============================  classification ======================================
        loss_ce = self.criterion_ce(y_hat, y)
        self.log(f"multimodal.val.CEloss", loss_ce, on_epoch=True, on_step=False)

        loss = self.alpha*loss_ce
        self.log(f"multimodal.val.loss", loss, on_epoch=True, on_step=False)

        # task accuracy
        y_hat = torch.softmax(y_hat.detach(), dim=1)
        y_i_hat = torch.softmax(y_i_hat.detach(), dim=1)
        y_t_hat = torch.softmax(y_t_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]
            y_i_hat = y_i_hat[:,1]
            y_t_hat = y_t_hat[:,1]
        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)
        self.acc_val_imaging(y_i_hat, y)
        self.auc_val_imaging(y_i_hat, y)
        self.acc_val_tabular(y_t_hat, y)
        self.auc_val_tabular(y_t_hat, y)
        return loss


    def on_validation_epoch_end(self) -> None:
        """
        Compute validation epoch metrics and check for new best values
        """
        if self.trainer.sanity_checking:
            return  

        epoch_acc_val = self.acc_val.compute()
        epoch_auc_val = self.auc_val.compute()
        epoch_acc_val_imaging = self.acc_val_imaging.compute()
        epoch_auc_val_imaging = self.auc_val_imaging.compute()
        epoch_acc_val_tabular = self.acc_val_tabular.compute()
        epoch_auc_val_tabular = self.auc_val_tabular.compute()

        self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
        self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
        self.log('eval.val.acc_imaging', epoch_acc_val_imaging, on_epoch=True, on_step=False, metric_attribute=self.acc_val_imaging)
        self.log('eval.val.auc_imaging', epoch_auc_val_imaging, on_epoch=True, on_step=False, metric_attribute=self.auc_val_imaging)
        self.log('eval.val.acc_tabular', epoch_acc_val_tabular, on_epoch=True, on_step=False, metric_attribute=self.acc_val_tabular)
        self.log('eval.val.auc_tabular', epoch_auc_val_tabular, on_epoch=True, on_step=False, metric_attribute=self.auc_val_tabular)
      
        if self.hparams.target == 'dvm':
            self.best_val_score = max(self.best_val_score, epoch_acc_val)
        else:
            self.best_val_score = max(self.best_val_score, epoch_auc_val)

        self.acc_val.reset()
        self.auc_val.reset()
        self.acc_val_imaging.reset()
        self.auc_val_imaging.reset()
        self.acc_val_tabular.reset()
        self.auc_val_tabular.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Runs test step
        """
        x,y = batch
        y_hat, _, _, _ = self.model.forward(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_hat = y_hat[:,1]

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def on_test_epoch_end(self) -> None:
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
