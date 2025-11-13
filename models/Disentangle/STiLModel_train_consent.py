'''
Use MultiAttention_model
Contrastive regularization and pseudo-labelling
Use label smoothing (teacher-student, prototypes)
''''''
Semi-supervised learning
'''
from typing import Tuple
import sys

import torch
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from .metrics import balanced_accuracy_by_hand

from lightly.models.modules import SimCLRProjectionHead
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/kgutjahr/STiL-TTA')
from models.Disentangle.utils.STiLModel_backbone_train_consent import DisCoAttentionBackbone
from utils.clip_loss import CLIPLoss
from utils.prototype_loss import PrototypeLoss
from utils.AugmentSummarizer import AugmentSummarizer
from models.Disentangle.utils.club import CLUBMean


class STiLModel_Consent(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        
        self.aug_summarizer = AugmentSummarizer()
        self.teacher_aug_summarizer = AugmentSummarizer()

        self.model = DisCoAttentionBackbone(self.hparams, self.aug_summarizer)
        print('Use STiLModel.py')
        
        self.pooled_dim = 2048 if self.hparams.model=='resnet50' else 512
        self.hidden_dim = self.hparams.multimodal_embedding_dim
        # for itc, club, and classification
        self.alpha = self.hparams.alpha
        self.beta = self.hparams.beta
        self.gamma = self.hparams.gamma
        self.epsilon = self.hparams.epsilon
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
            self.ema = DisCoAttentionBackbone(self.hparams, self.teacher_aug_summarizer)
            for param_model, param_ema in zip(self.model.parameters(), self.ema.parameters()):
                param_ema.data.copy_(param_model.data)
                param_ema.requires_grad = False

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
        
        self.w_m = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_i = nn.Parameter(torch.tensor(1.0 / 3))
        self.w_t = nn.Parameter(torch.tensor(1.0 / 3))
        
        self.train_logit_consent = self.hparams.train_logit_consent
        self.replace_ce_loss = self.hparams.replace_ce_loss
        self.num_classes = self.hparams.num_classes

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
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_multimodal = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_imaging = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test_multimodal = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test_imaging = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_multimodal = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_imaging = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val_tabular = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test_multimodal = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test_imaging = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test_tabular = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        
        self.acc_classifier_multi = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_classifier_image = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_classifier_tabular = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        
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
        x, y = batch
        im_views, tab_views = x
        B_l = len(y)
        # use augmented image and tabular views
        y_hat_m, y_hat_i, y_hat_t, x_si_enhance, x_si, x_ai, x_st_enhance, x_st, x_at, x_c = self.model.forward_all(x=[im_views, tab_views], y=y)

        feat_m = torch.cat((x_si_enhance, x_c, x_st_enhance), dim=1)
        feat_m, feat_i, feat_t = self.project_3features(feat_m, x_ai, x_at)

        # teacher model generate pseudo label, mask, and case identification
        if self.use_ema:
            self.ema.eval()
        with torch.no_grad():
            if self.use_ema:
                self.momentum_update_ema()
                y_hat_m_e, y_hat_i_e, y_hat_t_e, x_si_enhance_e, _, _, x_st_enhance_e, _, _, x_c_e = self.ema.forward_all(x=[im_views, tab_views], y=y)
                feat_m_e = torch.cat((x_si_enhance_e, x_c_e, x_st_enhance_e), dim=1)
                feat_m_e, _, _ = self.project_3features(feat_m_e, None, None)
            else:
                y_hat_m_e, y_hat_i_e, y_hat_t_e = y_hat_m, y_hat_i, y_hat_t
                feat_m_e = feat_m
            feat_m_e = feat_m_e.detach()
        #    # case identification. case1: all the same, case2: two the same, case3: else
            prob_m_e, prob_i_e, prob_t_e = torch.softmax(y_hat_m_e.detach(), dim=1), torch.softmax(y_hat_i_e.detach(), dim=1), torch.softmax(y_hat_t_e.detach(), dim=1)
            top1_m, top1_i, top1_t = torch.argmax(prob_m_e, dim=1), torch.argmax(prob_i_e, dim=1), torch.argmax(prob_t_e, dim=1)
            
            self.acc_classifier_multi(top1_m, y)
            self.acc_classifier_image(top1_i, y)
            self.acc_classifier_tabular(top1_t, y)
            
            entropy_m = -torch.sum(prob_m_e * torch.log(prob_m_e + 1e-9), dim=1)
            entropy_i = -torch.sum(prob_i_e * torch.log(prob_i_e + 1e-9), dim=1)
            entropy_t = -torch.sum(prob_t_e * torch.log(prob_t_e + 1e-9), dim=1)
            
            self.log(f'multimodal.classifier.entropy', entropy_m, on_epoch=True, on_step=False)
            self.log(f'image.classifier.entropy', entropy_i, on_epoch=True, on_step=False)
            self.log(f'tabular.classifier.entropy', entropy_t, on_epoch=True, on_step=False)
            
            
            balanced_acc_m = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_m.cpu(), num_classes=self.num_classes)
            balanced_acc_i = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_i.cpu(), num_classes=self.num_classes)
            balanced_acc_t = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_t.cpu(), num_classes=self.num_classes)
            
            self.log(f'multimodal.train.balanced_acc', balanced_acc_m, on_epoch=True, on_step=False)
            self.log(f'image.train.balanced_acc', balanced_acc_i, on_epoch=True, on_step=False)
            self.log(f'tabular.train.balanced_acc', balanced_acc_t, on_epoch=True, on_step=False)
            
            with torch.no_grad():
                
                case1 = ((top1_m == top1_i) & (top1_m == top1_t))
                case2_i = ((top1_m == top1_i) & (top1_m != top1_t))
                case2_t = (top1_m == top1_t) & (top1_m != top1_i)
                case3 = ~(case1 | case2_i | case2_t)
                assert ((case1.float()+case2_i.float()+case2_t.float()+case3.float()) == torch.ones_like(case1, device=case1.device).float()).all()
                
                self.log(f'multimodal.train.case1_ratio', torch.sum(case1)/len(case1), on_epoch=True, on_step=False, batch_size=B_l)
                self.log(f'multimodal.train.case2_i_ratio', torch.sum(case2_i)/len(case2_i), on_epoch=True, on_step=False, batch_size=B_l)
                self.log(f'multimodal.train.case2_t_ratio', torch.sum(case2_t)/len(case2_t), on_epoch=True, on_step=False, batch_size=B_l)
                self.log(f'multimodal.train.case3_ratio', torch.sum(case3)/len(case3), on_epoch=True, on_step=False, batch_size=B_l)         
            
        # Weighted combination of logits
        if self.train_logit_consent:
            w = F.softmax(torch.stack([self.w_m, self.w_i, self.w_t]), dim=0)
            p_prime = w[0] * y_hat_m + w[1] * y_hat_i + w[2] * y_hat_t
            # Cross-entropy loss with ground truth labels
            loss_p_prime = self.criterion_ce(p_prime, y)
            self.log(f"multimodal.train.p_prime_loss", loss_p_prime, on_epoch=True, on_step=False, batch_size=B_l)
            
            prob_prime = torch.softmax(p_prime.detach(), dim=1)
            top1 = torch.argmax(prob_prime, dim=1)
            balanced_acc = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1.cpu(), num_classes=self.num_classes)
            self.log(f'train.balanced_acc', balanced_acc, on_epoch=True, on_step=False)
            self.acc_train(prob_prime, y)
            self.auc_train(prob_prime, y)
        else:
            loss_p_prime = 0.0
            prob_m_l = torch.softmax(y_hat_m.detach(), dim=1)
            top1 = torch.argmax(prob_m_l, dim=1)
            balanced_acc = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1.cpu(), num_classes=self.num_classes)
            self.log(f'train.balanced_acc', balanced_acc, on_epoch=True, on_step=False)
            self.acc_train(prob_m_l, y)
            self.auc_train(prob_m_l, y)

        # ============================= itc loss =======================================
        loss_itc, logits, labels = self.criterion_itc(feat_i, feat_t)
        self.log(f"multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False, batch_size=B_l)

        
        # ==================================== disentangle loss ===========================================
        loss_clubi = self.CLUB_imaging(x_si, x_ai)
        loss_club_i_est = self.CLUB_imaging.learning_loss(x_si, x_ai)
        loss_club_t = self.CLUB_tabular(x_st, x_at)
        loss_club_t_est = self.CLUB_tabular.learning_loss(x_st, x_at)
        self.log(f"multimodal.train.CLUBloss_imaging", loss_clubi, on_epoch=True, on_step=False, batch_size=B_l)
        self.log(f"multimodal.train.CLUBloss_imaging_est", loss_club_i_est, on_epoch=True, on_step=False, batch_size=B_l)
        self.log(f"multimodal.train.CLUBloss_tabular", loss_club_t, on_epoch=True, on_step=False, batch_size=B_l)
        self.log(f"multimodal.train.CLUBloss_tabular_est", loss_club_t_est, on_epoch=True, on_step=False, batch_size=B_l)
        
        # =============================  classification ======================================
        # student labelled CE loss
        loss_ce = self.criterion_ce(y_hat_m, y) + self.criterion_ce(y_hat_i, y) + self.criterion_ce(y_hat_t, y)
        self.log(f"multimodal.train.CEloss", loss_ce, on_epoch=True, on_step=False, batch_size=B_l)
        
        if self.train_logit_consent and self.replace_ce_loss:
            self.alpha = 0.0

        loss = self.alpha*loss_ce + self.epsilon*loss_p_prime + self.beta*loss_itc + self.gamma*(loss_clubi + loss_club_i_est + loss_club_t + loss_club_t_est)
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False, batch_size=B_l)
        
        torch.cuda.empty_cache()
        return loss
        

    def training_epoch_end(self, _) -> None:
        """
        Compute training epoch metrics and check for new best values
        """
        aug_sum = self.aug_summarizer.summarize()
        teacher_aug_sum = self.teacher_aug_summarizer.summarize()
        self.aug_summarizer.reset()
        self.teacher_aug_summarizer.reset()

        self.log('eval.train.latent.multi.aug_rate', aug_sum["multi_rate"], on_epoch=True, on_step=False)
        self.log('eval.train.latent.image.aug_rate', aug_sum["image_rate"], on_epoch=True, on_step=False)
        self.log('eval.train.latent.table.aug_rate', aug_sum["table_rate"], on_epoch=True, on_step=False)
        
        self.log('eval.train.teacher.latent.multi.aug_rate', teacher_aug_sum["multi_rate"], on_epoch=True, on_step=False)
        self.log('eval.train.teacher.latent.image.aug_rate', teacher_aug_sum["image_rate"], on_epoch=True, on_step=False)
        self.log('eval.train.teacher.latent.table.aug_rate', teacher_aug_sum["table_rate"], on_epoch=True, on_step=False)
        
        self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False, metric_attribute=self.acc_train)
        self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False, metric_attribute=self.auc_train)
        self.log('classifier.multi.acc', self.acc_classifier_multi, on_epoch=True, on_step=False, metric_attribute=self.acc_classifier_multi)
        self.log('classifier.image.acc', self.acc_classifier_image, on_epoch=True, on_step=False, metric_attribute=self.acc_classifier_image)
        self.log('classifier.tab.acc', self.acc_classifier_tabular, on_epoch=True, on_step=False, metric_attribute=self.acc_classifier_tabular)
        # self.log('eval.train.l_prot_acc', self.acc_train_labelled_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_labelled_prototypes)
        # self.log('eval.train.u_prot_acc', self.acc_train_unlabelled_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_unlabelled_prototypes)
        # if self.use_pseudo:
        #     self.log('eval.train_pseudo.acc', self.acc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.acc_train_pseudo)
        #     self.log('eval.train_pseudo.auc', self.auc_train_pseudo, on_epoch=True, on_step=False, metric_attribute=self.auc_train_pseudo)
            # self.log('eval.train_pseudo_prototypes.acc', self.acc_train_pseudo_prototypes, on_epoch=True, on_step=False, metric_attribute=self.acc_train_pseudo_prototypes)
            # self.log('eval.train_pseudo_prototypes.auc', self.auc_train_pseudo_prototypes, on_epoch=True, on_step=False, metric_attribute=self.auc_train_pseudo_prototypes)
            # self.use_pseudo = False
        
        self.print(f'Epoch {self.current_epoch}: train.acc: {self.acc_train.compute()}, train.auc: {self.auc_train.compute()}')

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
        y_m_hat, y_i_hat, y_t_hat, x_si_enhance, x_si, x_ai, x_st_enhance, x_st, x_at, x_c = self.model.forward_all(x=x, y=y)
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
        loss_ce = self.criterion_ce(y_m_hat, y)
        self.log(f"multimodal.val.CEloss", loss_ce, on_epoch=True, on_step=False)
        
        # Weighted combination of logits
        if self.train_logit_consent:
            w = F.softmax(torch.stack([self.w_m, self.w_i, self.w_t]), dim=0)
            p_prime = w[0] * y_m_hat + w[1] * y_i_hat + w[2] * y_t_hat
            # Cross-entropy loss with ground truth labels
            loss_p_prime = self.criterion_ce(p_prime, y)
            self.log(f"multimodal.val.p_prime_loss", loss_p_prime, on_epoch=True, on_step=False)

            prob_prime = torch.softmax(p_prime.detach(), dim=1)
            top1 = torch.argmax(prob_prime, dim=1)
            balanced_acc = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1.cpu(), num_classes=self.num_classes)
            self.log(f'eval.val.balanced_acc', balanced_acc, on_epoch=True, on_step=False)
            self.acc_val(prob_prime, y)
            self.auc_val(prob_prime, y)
        else:
            loss_p_prime = 0.0
            prob_m_l = torch.softmax(y_m_hat.detach(), dim=1)
            top1 = torch.argmax(prob_m_l, dim=1)
            balanced_acc = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1.cpu(), num_classes=self.num_classes)
            self.log(f'eval.val.balanced_acc', balanced_acc, on_epoch=True, on_step=False)
            self.acc_val(prob_m_l, y)
            self.auc_val(prob_m_l, y)
        
        if self.train_logit_consent and self.replace_ce_loss:
            self.alpha = 0.0

        # loss = self.alpha*loss_ce + self.beta*loss_itc
        loss = self.alpha*loss_ce + self.beta*loss_itc + self.gamma*(loss_club_i + loss_club_i_est + loss_club_t + loss_club_t_est) + self.epsilon*loss_p_prime
        self.log(f"multimodal.val.loss", loss, on_epoch=True, on_step=False)
        
        prob_m_e, prob_i_e, prob_t_e = torch.softmax(y_m_hat.detach(), dim=1), torch.softmax(y_i_hat.detach(), dim=1), torch.softmax(y_t_hat.detach(), dim=1)
        top1_m, top1_i, top1_t = torch.argmax(prob_m_e, dim=1), torch.argmax(prob_i_e, dim=1), torch.argmax(prob_t_e, dim=1)
        
        balanced_acc_m = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_m.cpu(), num_classes=self.num_classes)
        balanced_acc_i = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_i.cpu(), num_classes=self.num_classes)
        balanced_acc_t = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_t.cpu(), num_classes=self.num_classes)
            
        self.log(f'multimodal.val.balanced_acc', balanced_acc_m, on_epoch=True, on_step=False)
        self.log(f'image.val.balanced_acc', balanced_acc_i, on_epoch=True, on_step=False)
        self.log(f'tabular.val.balanced_acc', balanced_acc_t, on_epoch=True, on_step=False)

        # task accuracy
        y_m_hat = torch.softmax(y_m_hat.detach(), dim=1)
        y_i_hat = torch.softmax(y_i_hat.detach(), dim=1)
        y_t_hat = torch.softmax(y_t_hat.detach(), dim=1)
        if self.hparams.num_classes==2:
            y_m_hat = y_m_hat[:,1]
            y_i_hat = y_i_hat[:,1]
            y_t_hat = y_t_hat[:,1]
        self.acc_val_multimodal(y_m_hat, y)
        self.auc_val_multimodal(y_m_hat, y)
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
        epoch_acc_val_multimodal = self.acc_val_multimodal.compute()
        epoch_auc_val_multimodal = self.auc_val_multimodal.compute()
        epoch_acc_val_imaging = self.acc_val_imaging.compute()
        epoch_auc_val_imaging = self.auc_val_imaging.compute()
        epoch_acc_val_tabular = self.acc_val_tabular.compute()
        epoch_auc_val_tabular = self.auc_val_tabular.compute()

        self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
        self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
        self.log('eval.val.acc_multimodal', epoch_acc_val_multimodal, on_epoch=True, on_step=False, metric_attribute=self.acc_val_multimodal)
        self.log('eval.val.auc_multimodal', epoch_auc_val_multimodal, on_epoch=True, on_step=False, metric_attribute=self.auc_val_multimodal)
        
        self.log('eval.val.acc_imaging', epoch_acc_val_imaging, on_epoch=True, on_step=False, metric_attribute=self.acc_val_imaging)
        self.log('eval.val.auc_imaging', epoch_auc_val_imaging, on_epoch=True, on_step=False, metric_attribute=self.auc_val_imaging)
        self.log('eval.val.acc_tabular', epoch_acc_val_tabular, on_epoch=True, on_step=False, metric_attribute=self.acc_val_tabular)
        self.log('eval.val.auc_tabular', epoch_auc_val_tabular, on_epoch=True, on_step=False, metric_attribute=self.auc_val_tabular)

        self.print(f'Epoch {self.current_epoch}: val.acc: {epoch_acc_val}, val.auc: {epoch_auc_val}, val.acc_multimodal: {epoch_acc_val_multimodal}, val.auc_multimodal: {epoch_auc_val_multimodal}, val.acc_imaging: {epoch_acc_val_imaging}, val.auc_imaging: {epoch_auc_val_imaging}, val.acc_tabular: {epoch_acc_val_tabular}, val.auc_tabular: {epoch_auc_val_tabular}')
      
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
        self.acc_val_multimodal.reset()
        self.auc_val_multimodal.reset()
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
        
        y_hat_m, y_hat_i, y_hat_t, _, _, _, _, _ = self.model.forward(x)     

        y_hat_m_s = torch.softmax(y_hat_m.detach(), dim=1)
        y_hat_i_s = torch.softmax(y_hat_i.detach(), dim=1)
        y_hat_t_s = torch.softmax(y_hat_t.detach(), dim=1)
        
        entropy_m = -torch.sum(y_hat_m_s * torch.log(y_hat_m_s + 1e-9), dim=1)
        entropy_i = -torch.sum(y_hat_i_s * torch.log(y_hat_i_s + 1e-9), dim=1)
        entropy_t = -torch.sum(y_hat_t_s * torch.log(y_hat_t_s + 1e-9), dim=1)
        
        self.log(f'multimodal.test.classifier.entropy', entropy_m, on_epoch=True, on_step=False, batch_size=x[0].size()[0])
        self.log(f'image.test.classifier.entropy', entropy_i, on_epoch=True, on_step=False, batch_size=x[0].size()[0])
        self.log(f'tabular.test.classifier.entropy', entropy_t, on_epoch=True, on_step=False, batch_size=x[0].size()[0])
        
        top1_m, top1_i, top1_t = torch.argmax(y_hat_m_s, dim=1), torch.argmax(y_hat_i_s, dim=1), torch.argmax(y_hat_t_s, dim=1)
        balanced_acc_m = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_m.cpu(), num_classes=self.num_classes)
        balanced_acc_i = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_i.cpu(), num_classes=self.num_classes)
        balanced_acc_t = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1_t.cpu(), num_classes=self.num_classes)
            
        self.log(f'multimodal.test.balanced_acc', balanced_acc_m, on_epoch=True, on_step=False)
        self.log(f'image.test.balanced_acc', balanced_acc_i, on_epoch=True, on_step=False)
        self.log(f'tabular.test.balanced_acc', balanced_acc_t, on_epoch=True, on_step=False)
        
        if self.hparams.num_classes==2:
            y_hat = y_hat_m[:,1]
            
        if self.train_logit_consent:
            p_prime = self.w_m * y_hat_m + self.w_i * y_hat_i + self.w_t * y_hat_t
            y_hat = torch.softmax(p_prime.detach(), dim=1)
        else:
            y_hat = torch.softmax(y_hat_m.detach(), dim=1)

        top1 = torch.argmax(y_hat, dim=1)
        balanced_acc = balanced_accuracy_by_hand(y_true=y.cpu(), y_pred=top1.cpu(), num_classes=self.num_classes)
        self.log(f'test.balanced_acc', balanced_acc, on_epoch=True, on_step=False)

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)
        
        self.acc_test_multimodal(y_hat_m, y)
        self.auc_test_multimodal(y_hat_m, y)
        
        self.acc_test_imaging(y_hat_i, y)
        self.auc_test_imaging(y_hat_i, y)
        
        self.acc_test_tabular(y_hat_t, y)
        self.auc_test_tabular(y_hat_t, y)

    def test_epoch_end(self, _) -> None:
        """
        Test epoch end
        """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()
        test_acc_multimodal = self.acc_test_multimodal.compute()
        test_auc_multimodal = self.auc_test_multimodal.compute()
        test_acc_imaging = self.acc_test_imaging.compute()
        test_auc_imaging = self.auc_test_imaging.compute()
        test_acc_tabular = self.acc_test_tabular.compute()
        test_auc_tabular = self.auc_test_tabular.compute()

        self.log('test.acc', test_acc)
        self.log('test.auc', test_auc)
        self.log('test.acc_multimodal', test_acc_multimodal)
        self.log('test.auc_multimodal', test_auc_multimodal)
        self.log('test.acc_imaging', test_acc_imaging)
        self.log('test.auc_imaging', test_auc_imaging)
        self.log('test.acc_tabular', test_acc_tabular)
        self.log('test.auc_tabular', test_auc_tabular)
    
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
            {'params': self.CLUB_tabular.parameters()},
            {'params': self.w_m},
            {'params': self.w_i},
            {'params': self.w_t},
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