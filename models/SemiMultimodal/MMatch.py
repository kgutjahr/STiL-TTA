'''
MMatch pytorch lightning module
'''
from typing import List, Tuple, Dict
import sys

import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.SemiMultimodal.Multimodal_model import MultimodalBackbone


class MMatch(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = MultimodalBackbone(self.hparams)
        print('Use concatenation model in MMatch')
        
        self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
        self.hidden_dim = self.hparams.multimodal_embedding_dim
        # for itc, club, and classification
        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma
        self.rate_uce = self.hparams.rate_uce
        self.mmatch_lambda = self.hparams.mmatch_lambda
        self.th1 = self.hparams.th1
        self.th2 = self.hparams.th2
        self.T = self.hparams.temperature
        self.prototype_momentum = self.hparams.prototype_momentum
        self.rate_pseudo = self.hparams.rate_pseudo
        self.start_epoch = self.hparams.start_epoch   # start using pseudo label
        self.th_contrast = self.hparams.th_contrast
        print(f'threshold: {self.th1}')
        print(f'MMatch Lambda: {self.mmatch_lambda}')
 
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.use_pseudo = False
        self.use_ddp = torch.cuda.device_count() > 1

        # memory bank size
        self.K = 640

        self.initialize_metrics()
    
        self.best_val_score = 0

        # prototypes
        self.register_buffer("embed_queue", torch.randn((self.hparams.projection_dim, self.K)))
        self.embed_queue = F.normalize(self.embed_queue, dim=0)
        self.register_buffer("embed_queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("probs_queue", torch.zeros((self.hparams.num_classes, self.K)))
        
        # distribution alignment
        if self.hparams.DA == True:
            self.DA_len = 256
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, self.hparams.num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))

        print(f'Model backbone: {self.model}')

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0
    
    def initialize_metrics(self):
        # classification metrics
        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'
        
        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_train_unlabelled = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_train_pseudo = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_imaging = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_unlabelled = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_train_pseudo = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_imaging = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_tabular = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, z, t, ws):
        if self.use_ddp:
            z = concat_all_gather(z)
            t = concat_all_gather(t)

        batch_size = z.shape[0]
        ptr = int(self.embed_queue_ptr)
        if (ptr + batch_size) > self.K:
            batch_size = self.K - ptr
            z = z[:batch_size]
            t = t[:batch_size]
        # replace the samples at ptr (dequeue and enqueue)
        self.embed_queue[:, ptr:ptr + batch_size] = z.T
        self.probs_queue[:, ptr:ptr + batch_size] = t.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.embed_queue_ptr[0] = ptr

    @torch.no_grad()
    def momentum_update_ema(self):
        if self.eman:
            state_dict_main = self.model.state_dict()
            state_dict_ema = self.ema.state_dict()
            for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
                assert k_main == k_ema, "state_dict names are different!"
                assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
                if 'num_batches_tracked' in k_ema:
                    v_ema.copy_(v_main)
                else:
                    v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
        else:
            for param_q, param_k in zip(self.model.parameters(), self.ema.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        if self.use_ddp:
            torch.distributed.all_reduce(probs_bt_mean)
        ptr = int(self.DA_ptr)
        if self.use_ddp:
            self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
        else:
            self.DA_queue[ptr] = probs_bt_mean
        self.DA_ptr[0] = (ptr + 1) % self.DA_len
        probs = probs / self.DA_queue.mean(0)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs.detach()
    
    def project_3features(self, feat_m=None, feat_i=None, feat_t=None):
        if feat_m is not None:
            feat_m = self.projector_multimodal(feat_m)
            feat_m = F.normalize(feat_m)
        if feat_i is not None:
            feat_i = self.projector_imaging(feat_i)
            feat_i = F.normalize(feat_i)
        if feat_t is not None:
            feat_t = self.projector_tabular(feat_t)
            feat_t = F.normalize(feat_t)
        return feat_m, feat_i, feat_t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_m_hat, y_i_hat, y_t_hat, x_m = self.model(x)
        return y_m_hat, y_i_hat, y_t_hat, x_m

    def sharpen_predictions(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=1)

    def cal_predictions_diff(self, p1, p2, p3):
        # return a value to indicate the difference between the predictions
        p_mean = (p1 + p2 + p3) / 3.0   # (B, D)
        diff = (torch.sqrt(torch.mean((p1-p_mean)**2, dim=1)) + torch.sqrt(torch.mean((p2-p_mean)**2, dim=1)) + torch.sqrt(torch.mean((p3-p_mean)**2, dim=1)))/3
        return diff

    def cal_prototypes(self, label, feat):
        '''
        Calculate prototypes for each class
        Only use confident samples
        '''
        max_prob, max_id = torch.max(label, dim=1)
        conf_mask = max_prob.ge(self.th1)
        with torch.no_grad():
            # hard label
            hard_label = torch.zeros_like(label).to(label.device)
            hard_label[torch.arange(len(label)), max_id] = 1
        hard_label, feat = hard_label[conf_mask], feat[conf_mask]
        class_sum = hard_label.t() @ feat
        class_count = torch.sum(hard_label, dim=0, keepdim=True).t()
        return class_sum, class_count

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
        # use augmented image and tabular views
        y_hat_m, y_hat_i, y_hat_t, x_m = self.forward([torch.cat((im_views_l[1], im_views_u[1])), torch.cat((tab_views_l[1], tab_views_u[1]))])
        prob_m, prob_i, prob_t = torch.softmax(y_hat_m.detach(), dim=1), torch.softmax(y_hat_i.detach(), dim=1), torch.softmax(y_hat_t.detach(), dim=1)
        prob_m_l, prob_m_u = prob_m[:B_l], prob_m[B_l:]
        feat_m = F.normalize(x_m.detach(), dim=1)
        feat_m_u = feat_m[B_l:]

        # ============================= classification =====================================
        # labelled loss
        loss_ce = self.criterion_ce(y_hat_m[:B_l], y_l) + self.criterion_ce(y_hat_i[:B_l], y_l) + self.criterion_ce(y_hat_t[:B_l], y_l)
        
        # unlabelled loss
        # pseudo label generation
        pseudo_label = self.distribution_alignment(torch.softmax(y_hat_m[B_l:], dim=1))
        pseudo_label_orig = pseudo_label.clone()
        if current_epoch > 0:
            with torch.no_grad():
                feat_bank = self.embed_queue.clone().detach()
                probs_bank = self.probs_queue.clone().detach()
                A = torch.exp(torch.mm(feat_m_u, feat_bank)/self.T)   
                A = A/A.sum(dim=1, keepdim=True)
                pseudo_label = 0.9 * pseudo_label_orig + 0.1 * torch.mm(A, probs_bank.t())
        
        max_prob, max_idx = torch.max(pseudo_label, dim=1)
        mask1 = max_prob.ge(self.th1)
        hard_label = torch.zeros_like(pseudo_label).to(pseudo_label.device)
        hard_label[torch.arange(len(pseudo_label)), max_idx] = 1
        loss_i_u = (F.cross_entropy(y_hat_i[B_l:], hard_label, reduction='none')*mask1).mean()
        loss_t_u = (F.cross_entropy(y_hat_t[B_l:], hard_label, reduction='none')*mask1).mean()
        
        self.log(f"multimodal.train.CEloss_unlabelled_i", loss_i_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CEloss_unlabelled_t", loss_t_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.threshold1_ratio', torch.sum(mask1)/len(mask1), on_epoch=True, on_step=False, batch_size=B_l+B_u)

        if current_epoch > self.start_epoch:
            loss = self.alpha*loss_ce + self.mmatch_lambda*(loss_i_u + loss_t_u)
        else:
            loss = self.alpha*loss_ce
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False, batch_size=B_l+B_u)


        # labelled task accuracy
        pseudo_label_all = torch.cat((F.one_hot(y_l, self.hparams.num_classes).float(), pseudo_label), dim=0)
        if self.hparams.num_classes==2:
            prob_m_l = prob_m_l[:,1]
            prob_m_u = prob_m_u[:,1]
            pseudo_label = pseudo_label[:,1]
        self.acc_train(prob_m_l, y_l)
        self.auc_train(prob_m_l, y_l)
        self.acc_train_unlabelled(prob_m_u, y_u)
        self.auc_train_unlabelled(prob_m_u, y_u)

        # if torch.sum(mask1) > 0:
        #     self.use_pseudo = True
        #     self.acc_train_pseudo(pseudo_label[mask1], y_u[mask1])
        #     self.auc_train_pseudo(pseudo_label[mask1], y_u[mask1])
        
        self._dequeue_and_enqueue(feat_m, pseudo_label_all, mask1)

        torch.cuda.empty_cache()
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


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        x, y = batch
        # use augmented image and tabular views
        y_hat, y_i_hat, y_t_hat, x_m = self.forward(x)
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

    def validation_epoch_end(self, _) -> None:
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
        y_hat, _, _, _ = self.forward(x)

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



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output