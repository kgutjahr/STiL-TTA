'''
Use MultiAttention_model
Contrastive regularization and pseudo-labelling
Use label smoothing (teacher-student, prototypes)
''''''
Semi-supervised learning
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

from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.Disentangle.utils.STiLModel_backbone import DisCoAttentionBackbone
from utils.clip_loss import CLIPLoss
from utils.prototype_loss import PrototypeLoss
from models.Disentangle.utils.club import CLUBMean


class STiLModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = DisCoAttentionBackbone(self.hparams)
        print('Use STiLModel.py')
        
        self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
        self.hidden_dim = self.hparams.multimodal_embedding_dim
        # for itc, club, and classification
        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma
        self.rate_uce = self.hparams.rate_uce
        self.th1 = self.hparams.th1
        self.th2 = self.hparams.th2
        self.T = self.hparams.temperature
        self.rate_pseudo = self.hparams.rate_pseudo
        self.start_epoch = self.hparams.start_epoch   # start epoch using pseudo label
        self.th_contrast = self.hparams.th_contrast
        self.rate_pt = self.hparams.rate_pt
        self.repeat_ratio = self.hparams.repeat_ratio
        print('Start pseudo label from epoch:', self.start_epoch)
        print(f'Semi Pseudo training weights. alpha: {self.alpha}, beta: {self.beta}, gamma: {self.gamma}, rate_pt: {self.rate_pt}, unlabelled ce: {self.rate_uce}')
        print(f'rate_pseudo: {self.rate_pseudo}')
        # contrastive loss
        self.projector_multimodal = SimCLRProjectionHead(self.hparams.multimodal_embedding_dim*3, self.hparams.multimodal_embedding_dim*3, self.hparams.projection_dim)
        if self.hparams.target == 'dvm':
            self.projector_imaging = nn.Linear(self.hparams.multimodal_embedding_dim, self.hparams.projection_dim)
            self.projector_tabular = nn.Linear(self.hparams.multimodal_embedding_dim, self.hparams.projection_dim)
            print('DVM use linear projection heads for itc')
        else:
            self.projector_imaging = SimCLRProjectionHead(self.hparams.multimodal_embedding_dim, self.hparams.multimodal_embedding_dim, self.hparams.projection_dim)
            self.projector_tabular = SimCLRProjectionHead(self.hparams.multimodal_embedding_dim, self.hparams.multimodal_embedding_dim, self.hparams.projection_dim)
            print('Use SimCLR projection heads for itc')
        nclasses = hparams.batch_size
        # MI loss
        self.CLUB_imaging = CLUBMean(self.hidden_dim, self.hidden_dim)
        self.CLUB_tabular = CLUBMean(self.hidden_dim, self.hidden_dim)

        # classification loss
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.criterion_itc = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
        self.criterion_pt = PrototypeLoss(temperature=self.hparams.temperature, threshold=self.hparams.th1)
        self.use_pseudo = False
        self.use_ddp = torch.cuda.device_count() > 1
        print(f'Use DDP: {self.use_ddp}')

        self.initialize_metrics(nclasses, nclasses)
    
        self.best_val_score = 0

        # teacher model
        self.use_ema = self.hparams.use_ema
        if self.use_ema:
            print('Use EMA as teacher model')
            self.eman = self.hparams.eman
            self.momentum = self.hparams.ema_momentum
            self.ema = DisCoAttentionBackbone(self.hparams)
            for param_model, param_ema in zip(self.model.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_model.data)
                param_ema.requires_grad = False

        # prototypes
        self.register_buffer("prototypes", torch.zeros(self.hparams.num_classes, self.hparams.projection_dim))
        self.register_buffer("prototypes_sum", torch.zeros(self.hparams.num_classes, self.hparams.projection_dim))
        self.register_buffer("prototypes_count_sum", torch.zeros(self.hparams.num_classes, 1))
        self.logdir = self.hparams.logdir
        
        # distribution alignment
        if self.hparams.DA == True:
            self.DA_len = 256
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, self.hparams.num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))
            print('Use distribution alignment')
        else:
            print('Do not use distribution alignment')

        print(f'Model backbone: {self.model}')
        print(f'ITC imaging head: {self.projector_imaging}')
        print(f'ITC tabular head: {self.projector_tabular}')
        print(f'ITC multimodal head: {self.projector_multimodal}')

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0
    
    def initialize_metrics(self, nclasses_train, nclasses_val):
        # contrastive loss metrics
        self.top1_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_train)
        self.top1_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=1, num_classes=nclasses_val)

        self.top5_acc_train = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_train)
        self.top5_acc_val = torchmetrics.Accuracy(task='multiclass', top_k=5, num_classes=nclasses_val)

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
        # self.auc_train_pseudo_prototypes = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

        # self.acc_train_labelled_prototypes = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        # self.acc_train_unlabelled_prototypes = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        # self.acc_train_pseudo_prototypes = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

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
                    v_ema.data.mul_(self.momentum).add_((1. - self.momentum) * v_main.data)
        else:
            for param_q, param_k in zip(self.model.parameters(), self.ema.parameters()):
                param_k.data.mul_(self.momentum).add_((1. - self.momentum) * param_q.data)


    @torch.no_grad()
    def distribution_alignment(self, probs):
        probs_bt_mean = probs.mean(0)
        torch.distributed.all_reduce(probs_bt_mean)
        ptr = int(self.DA_ptr)
        self.DA_queue[ptr] = probs_bt_mean / torch.distributed.get_world_size()
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
    

    def sharpen_predictions(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=1)


    def cal_prototypes(self, label, feat):
        '''
        Calculate prototypes for each class
        Only use confident samples
        '''
        max_prob, max_id = torch.max(label, dim=1)
        conf_mask = max_prob.ge(self.th1)
        # print(f'Conf mask ratio: {torch.sum(conf_mask)/len(conf_mask)}')
        with torch.no_grad():
            # hard label
            hard_label = torch.zeros_like(label, device=label.device)
            hard_label[torch.arange(len(label)), max_id] = 1
        hard_label, feat = hard_label[conf_mask], feat[conf_mask]
        class_sum = hard_label.t() @ feat
        class_count = torch.sum(hard_label, dim=0, keepdim=True).t()
        return class_sum, class_count

    def cal_prototypes_separate(self, label, feat, B_l):
        '''
        Consider the repeat of labelled data
        '''
        l_label, u_label = label[:B_l], label[B_l:]
        l_feat, u_feat = feat[:B_l], feat[B_l:]
        l_class_sum, l_class_count = self.cal_prototypes(l_label, l_feat)
        u_class_sum, u_class_count = self.cal_prototypes(u_label, u_feat)
        class_sum = l_class_sum/self.repeat_ratio + u_class_sum
        class_count = l_class_count/self.repeat_ratio + u_class_count
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
        y_hat_m, y_hat_i, y_hat_t, x_si_enhance, x_si, x_ai, x_st_enhance, x_st, x_at, x_c = self.model.forward_all([torch.cat((im_views_l[1], im_views_u[1])), torch.cat((tab_views_l[1], tab_views_u[1]))]) 
        prob_m = torch.softmax(y_hat_m.detach(), dim=1)
        prob_m_l, prob_m_u = prob_m[:B_l], prob_m[B_l:]
        feat_m = torch.cat((x_si_enhance, x_c, x_st_enhance), dim=1)
        feat_m, feat_i, feat_t = self.project_3features(feat_m, x_ai, x_at)

        # teacher model generate pseudo label, mask, and case identification
        if self.use_ema:
            self.ema.eval()
        with torch.no_grad():
            if self.use_ema:
                self.momentum_update_ema()
                y_hat_m_e, y_hat_i_e, y_hat_t_e, x_si_enhance_e, _, _, x_st_enhance_e, _, _, x_c_e = self.ema.forward_all([torch.cat((im_views_l[1], im_views_u[1])), torch.cat((tab_views_l[1], tab_views_u[1]))])
                feat_m_e = torch.cat((x_si_enhance_e, x_c_e, x_st_enhance_e), dim=1)
                feat_m_e, _, _ = self.project_3features(feat_m_e, None, None)
            else:
                y_hat_m_e, y_hat_i_e, y_hat_t_e = y_hat_m, y_hat_i, y_hat_t
                feat_m_e = feat_m
            feat_m_e = feat_m_e.detach()
            feat_m_le, feat_m_ue = feat_m_e[:B_l], feat_m_e[B_l:]
            y_hat_m_ue, y_hat_i_ue, y_hat_t_ue = y_hat_m_e[B_l:], y_hat_i_e[B_l:], y_hat_t_e[B_l:]
            # case identification. case1: all the same, case2: two the same, case3: else
            prob_m_ue, prob_i_ue, prob_t_ue = torch.softmax(y_hat_m_ue.detach(), dim=1), torch.softmax(y_hat_i_ue.detach(), dim=1), torch.softmax(y_hat_t_ue.detach(), dim=1)
            top1_m, top1_i, top1_t = torch.argmax(prob_m_ue, dim=1), torch.argmax(prob_i_ue, dim=1), torch.argmax(prob_t_ue, dim=1)
            case1 = ((top1_m == top1_i) & (top1_m == top1_t))
            case2_i = ((top1_m == top1_i) & (top1_m != top1_t))
            case2_t = (top1_m == top1_t) & (top1_m != top1_i)
            case3 = ~(case1 | case2_i | case2_t)
            assert ((case1.float()+case2_i.float()+case2_t.float()+case3.float()) == torch.ones_like(case1, device=case1.device).float()).all()
            # pseudo label for different cases
            case1_pseudo_label = self.sharpen_predictions((y_hat_m_ue + y_hat_i_ue + y_hat_t_ue)/3.0, 1.0)
            case2_i_pseudo_label = self.sharpen_predictions((y_hat_m_ue + y_hat_i_ue)/2.0, 1.0)
            case2_t_pseudo_label = self.sharpen_predictions((y_hat_m_ue + y_hat_t_ue)/2.0, 1.0)
            case3_pseudo_label = self.sharpen_predictions(y_hat_m_ue, 1.0)
            pseudo_label_orig = case1[:,None]*case1_pseudo_label + case2_i[:,None]*case2_i_pseudo_label + case2_t[:,None]*case2_t_pseudo_label + case3[:,None]*case3_pseudo_label
            # get prediction for threshold of pseudo label
            if self.hparams.DA == True:
                prediction = self.distribution_alignment(torch.softmax(y_hat_m_ue, dim=1))
            else:
                prediction = self.sharpen_predictions(y_hat_m_ue, 1.0)

            
        # =============================  classification ======================================
        # student labelled CE loss
        loss_ce = self.criterion_ce(y_hat_m[:B_l], y_l) + self.criterion_ce(y_hat_i[:B_l], y_l) + self.criterion_ce(y_hat_t[:B_l], y_l)
        prob_m_l = torch.softmax(y_hat_m[:B_l].detach(), dim=1)
        max_prob_l, max_idx_l = torch.max(prob_m_l, dim=1)
        self.log(f"multimodal.train.CEloss", loss_ce, on_epoch=True, on_step=False, batch_size=B_l+B_u)

        # student pseudo label loss
        # final pseudo label = rate_pseudo*pseudo_label_orig + (1-rate_pseudo)*teacher_probs
        prototypes = self.prototypes.clone().detach()
        with torch.no_grad():
            teacher_logits = feat_m_ue @ prototypes.t()
            teacher_probs = torch.softmax(teacher_logits/self.T, dim=1)
            pseudo_label = self.rate_pseudo*pseudo_label_orig + (1-self.rate_pseudo)*teacher_probs
            prediction = self.rate_pseudo*prediction + (1-self.rate_pseudo)*teacher_probs
            max_prob, max_idx = torch.max(prediction, dim=1)
            mask1 = max_prob.ge(self.th1)
            mask_random = torch.rand_like(mask1.float(), device=mask1.device).ge(0.5)

        loss_m_u = (F.cross_entropy(y_hat_m[B_l:], pseudo_label, reduction='none')*mask1*(case1)).mean()
        loss_i_u = (F.cross_entropy(y_hat_i[B_l:], pseudo_label, reduction='none')*mask1*(case1+case2_t+case3*mask_random)).mean()
        loss_t_u = (F.cross_entropy(y_hat_t[B_l:], pseudo_label, reduction='none')*mask1*(case1+case2_i+case3*(~mask_random))).mean()
        self.log(f"multimodal.train.CEloss_unlabelled_m", loss_m_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CEloss_unlabelled_i", loss_i_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CEloss_unlabelled_t", loss_t_u, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.threshold1_ratio', torch.sum(mask1)/len(mask1), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.case1_ratio', torch.sum(case1)/len(case1), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.case2_i_ratio', torch.sum(case2_i)/len(case2_i), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.case2_t_ratio', torch.sum(case2_t)/len(case2_t), on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f'multimodal.train.case3_ratio', torch.sum(case3)/len(case3), on_epoch=True, on_step=False, batch_size=B_l+B_u)


        # ============================= itc loss =======================================
        # update based on feat_i, feat_t
        # multimodal contrast probs = softmax(sim_embed_m)*pos_mask, normalize
        if current_epoch > self.start_epoch:
            pass 
        else:
            prediction = torch.zeros_like(prediction, device=prediction.device)
        pseudo_label_all = torch.cat((F.one_hot(y_l, self.hparams.num_classes).float(), prediction), dim=0)
        loss_itc, logits, labels = self.criterion_itc(feat_i, feat_t)
        self.log(f"multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False, batch_size=B_l+B_u)

        
        # ==================================== disentangle loss ===========================================
        loss_clubi = self.CLUB_imaging(x_si, x_ai)
        loss_club_i_est = self.CLUB_imaging.learning_loss(x_si, x_ai)
        loss_club_t = self.CLUB_tabular(x_st, x_at)
        loss_club_t_est = self.CLUB_tabular.learning_loss(x_st, x_at)
        self.log(f"multimodal.train.CLUBloss_imaging", loss_clubi, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CLUBloss_imaging_est", loss_club_i_est, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CLUBloss_tabular", loss_club_t, on_epoch=True, on_step=False, batch_size=B_l+B_u)
        self.log(f"multimodal.train.CLUBloss_tabular_est", loss_club_t_est, on_epoch=True, on_step=False, batch_size=B_l+B_u)


        # ============================= prototype loss ============================================
        # update based on feat_m
        loss_pt = self.criterion_pt(pseudo_label_all, prototypes, feat_m)
        self.log(f"multimodal.train.PTloss", loss_itc, on_epoch=True, on_step=False, batch_size=B_l+B_u)

        if  current_epoch <= self.start_epoch:
            loss = self.alpha*loss_ce + self.beta*loss_itc + self.gamma*(loss_clubi + loss_club_i_est + loss_club_t + loss_club_t_est)
        else:
            loss = self.alpha*loss_ce + self.beta*loss_itc + self.gamma*(loss_clubi + loss_club_i_est + loss_club_t + loss_club_t_est) + self.rate_pt*loss_pt + self.rate_uce*(loss_m_u + loss_i_u + loss_t_u)
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False, batch_size=B_l+B_u)


        # labelled task accuracy
        l_teacher_logits = feat_m_le @ prototypes.t()
        l_teacher_probs = torch.softmax(l_teacher_logits/self.T, dim=1)
        if self.hparams.num_classes==2:
            prob_m_l = prob_m_l[:,1]
            prob_m_u = prob_m_u[:,1]
            pseudo_label = pseudo_label[:,1]
            teacher_probs = teacher_probs[:,1]
            l_teacher_probs = l_teacher_probs[:,1]
            
        self.acc_train(prob_m_l, y_l)
        self.auc_train(prob_m_l, y_l)
        self.acc_train_unlabelled(prob_m_u, y_u)
        self.auc_train_unlabelled(prob_m_u, y_u)
        # self.acc_train_labelled_prototypes(l_teacher_probs, y_l)
        # self.acc_train_unlabelled_prototypes(teacher_probs, y_u)

        # Comment. May cause ddp stuck
        # if torch.sum(mask1) > 0:
        #     self.use_pseudo = True
        #     self.acc_train_pseudo(pseudo_label[mask1], y_u[mask1])
        #     self.auc_train_pseudo(pseudo_label[mask1], y_u[mask1])
            # self.acc_train_pseudo_prototypes(teacher_probs[mask1], y_u[mask1])
            # self.auc_train_pseudo_prototypes(teacher_probs[mask1], y_u[mask1])

        with torch.no_grad():
            # update prototypes
            class_sum, class_count = self.cal_prototypes_separate(pseudo_label_all, feat_m_e, B_l)
            if self.use_ddp:
                dist.all_reduce(class_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(class_count, op=dist.ReduceOp.SUM)
            self.prototypes_sum.data.add_(class_sum.detach())
            self.prototypes_count_sum.data.add_(class_count.detach())
        
        del prototypes, class_sum, class_count
        
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
        # self.log('eval.train.l_prot_acc', self.acc_train_labelled_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_labelled_prototypes)
        # self.log('eval.train.u_prot_acc', self.acc_train_unlabelled_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_unlabelled_prototypes)
        # if self.use_pseudo:
        #     self.log('eval.train_pseudo.acc', self.acc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.acc_train_pseudo)
        #     self.log('eval.train_pseudo.auc', self.auc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.auc_train_pseudo)
            # self.log('eval.train_pseudo_prototypes.acc', self.acc_train_pseudo_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_pseudo_prototypes)
            # self.log('eval.train_pseudo_prototypes.auc', self.auc_train_pseudo_prototypes, on_epoch=True, on_step=False, metric_attribute=self.auc_train_pseudo_prototypes)
            # self.use_pseudo = False
        
        self.print(f'Epoch {self.current_epoch}: train.acc: {self.acc_train.compute()}, train.auc: {self.auc_train.compute()}, train.acc_unlabelled: {self.acc_train_unlabelled.compute()}, train.auc_unlabelled: {self.auc_train_unlabelled.compute()}')

        with torch.no_grad():
            prototypes_count_sum = self.prototypes_count_sum.detach()
            prototypes_sum = self.prototypes_sum.detach()
            zero_count = torch.where(prototypes_count_sum < 1)[0]
            assert len(zero_count) == 0
            self.prototypes.data.copy_(prototypes_sum / prototypes_count_sum)
            self.prototypes_sum.zero_()
            self.prototypes_count_sum.zero_()

        del prototypes_count_sum, prototypes_sum, zero_count

        if self.use_ddp:
            dist.barrier()
        torch.cuda.empty_cache()
        

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Train and log.
        """
        # im_views, tab_views, y, original_im, _ = batch
        x, y = batch
        # use augmented image and tabular views
        y_hat, y_i_hat, y_t_hat, x_si_enhance, x_si, x_ai, x_st_enhance, x_st, x_at, x_c = self.model.forward_all(x)
        feat_m = torch.cat((x_si_enhance, x_c, x_st_enhance), dim=1)
        feat_m, feat_i, feat_t = self.project_3features(feat_m, x_ai, x_at)
        # =============================  itc ======================================
        loss_itc, logits, labels = self.criterion_itc(feat_i, feat_t)
        self.log(f"multimodal.val.ITCloss", loss_itc, on_epoch=True, on_step=False)
        if len(x[0])==self.hparams.batch_size:
            self.calc_and_log_val_embedding_acc(logits=logits, labels=labels, modality='multimodal')
        # =============================  club ======================================
        loss_club_i = self.CLUB_imaging(x_si, x_ai)
        loss_club_i_est = self.CLUB_imaging.learning_loss(x_si, x_ai)

        loss_club_t = self.CLUB_tabular(x_st, x_at)
        loss_club_t_est = self.CLUB_tabular.learning_loss(x_st, x_at)
        self.log(f"multimodal.val.CLUBloss_imaging", loss_club_i, on_epoch=True, on_step=False)
        self.log(f"multimodal.val.CLUBloss_imaging_est", loss_club_i_est, on_epoch=True, on_step=False)
        self.log(f"multimodal.val.CLUBloss_tabular", loss_club_t, on_epoch=True, on_step=False)
        self.log(f"multimodal.val.CLUBloss_tabular_est", loss_club_t_est, on_epoch=True, on_step=False)
        # =============================  classification ======================================
        loss_ce = self.criterion_ce(y_hat, y)
        self.log(f"multimodal.val.CEloss", loss_ce, on_epoch=True, on_step=False)

        # loss = self.alpha*loss_ce + self.beta*loss_itc
        loss = self.alpha*loss_ce + self.beta*loss_itc + self.gamma*(loss_club_i + loss_club_i_est + loss_club_t + loss_club_t_est)
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

        torch.cuda.empty_cache()
        
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

        self.print(f'Epoch {self.current_epoch}: val.acc: {epoch_acc_val}, val.auc: {epoch_auc_val}, val.acc_imaging: {epoch_acc_val_imaging}, val.auc_imaging: {epoch_auc_val_imaging}, val.acc_tabular: {epoch_acc_val_tabular}, val.auc_tabular: {epoch_auc_val_tabular}')
      
        if self.hparams.target == 'dvm':
            if epoch_acc_val > self.best_val_score:
                self.print(f'Best epoch: {self.current_epoch}')
            self.best_val_score = max(self.best_val_score, epoch_acc_val)
        else:
            if epoch_auc_val > self.best_val_score:
                self.print(f'Best epoch: {self.current_epoch}')
            self.best_val_score = max(self.best_val_score, epoch_auc_val)

        self.acc_val.reset()
        self.auc_val.reset()
        self.acc_val_imaging.reset()
        self.auc_val_imaging.reset()
        self.acc_val_tabular.reset()
        self.auc_val_tabular.reset()

        torch.cuda.empty_cache()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Runs test step
        """
        x,y = batch
        y_hat, _, _, _, _, _, _, _ = self.model.forward(x)

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
            {'params': self.projector_imaging.parameters()},
            {'params': self.projector_tabular.parameters()},
            {'params': self.projector_multimodal.parameters()},
            {'params': self.CLUB_imaging.parameters()},
            {'params': self.CLUB_tabular.parameters()}
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