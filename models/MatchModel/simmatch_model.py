'''
Built from SimMatch https://github.com/mingkai-zheng/SimMatch
'''
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.self_supervised import torchvision_ssl_encoder
from models.MatchModel.multimodal_backbone import MultimodalBackbone
from models.pieces import DotDict


class ResNet(nn.Module):
    def __init__(self, args, num_classes, out_channels, dim=128):
        super(ResNet, self).__init__()
        self.backbone = torchvision_ssl_encoder(args.model, return_all_feature_maps=False)
        # assert not hasattr(self.backbone, 'fc'), "fc should not in backbone"
        self.classifier = nn.Linear(out_channels, num_classes)
        self.head = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, dim),
        )

    def forward(self, x):
        x = self.backbone(x)[0].squeeze()
        embedding = self.head(x)
        logits = self.classifier(x)
        return logits, F.normalize(embedding)


class SimMatchModel(nn.Module):
    def __init__(self, args):
        super(SimMatchModel, self).__init__()
        self.eman = False
        self.momentum = args.ema_momentum
        self.num_classes = args.num_classes
        self.pooled_dim = args.embedding_dim
        self.dim = args.projection_dim
        K = args.K
        self.eval_datatype = args.eval_datatype

        print("using EMAN as techer model")
        if self.eval_datatype == 'imaging':
            self.main = ResNet(args, self.num_classes, out_channels=self.pooled_dim, dim=self.dim)
            self.ema = ResNet(args, self.num_classes, out_channels=self.pooled_dim, dim=self.dim)
            print(f'Using imaging encoder for SimMatch')
        elif self.eval_datatype == 'imaging_and_tabular':
            self.main = MultimodalBackbone(args)
            self.ema = MultimodalBackbone(args)
            print(f'Using multimodal encoder for SimMatch')
        else:
            assert False, f'Unknown eval datatype {self.eval_datatype}'
        # build ema model
        
        for param_main, param_ema in zip(self.main.parameters(), self.ema.parameters()):
            param_ema.data.copy_(param_main.data)  # initialize
            param_ema.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("bank", torch.randn(self.dim, K))
        self.bank = nn.functional.normalize(self.bank, dim=0)
        self.register_buffer("labels", torch.zeros(K, dtype=torch.long))

        if args.DA:
            self.DA_len = 256
            self.register_buffer("DA_queue", torch.zeros(self.DA_len, self.num_classes, dtype=torch.float))
            self.register_buffer("DA_ptr", torch.zeros(1, dtype=torch.long))
        self.DA = args.DA
        self.tt = args.tt
        self.c_smooth = args.c_smooth
        self.st = args.st
        self.use_ddp = torch.cuda.device_count() > 1

        if args.checkpoint and self.eval_datatype == 'imaging':
            print(f'Load weights for the image backbone from {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint)
            state_dict = checkpoint['state_dict']
            self.load_weights(self.main.backbone, 'encoder_imaging.', state_dict)
            if args.finetune_strategy == 'frozen':
                for _, param in self.model.main.backbone.named_parameters():
                    param.requires_grad = False
                parameters = list(filter(lambda p: p.requires_grad, self.main.backbone.parameters()))
                assert len(parameters)==0
                print(f'Freeze the backbone of the image encoder')
            elif args.finetune_strategy == 'trainable':
                print(f'Full finetune the backbone of the image encoder')
            else:
                assert False, f'Unknown finetune strategy {args.finetune_strategy}'

        for param, param_m in zip(self.main.parameters(), self.ema.parameters()):
            param_m.data.copy_(param.data)  
            param_m.requires_grad = False
        # check main and ema have the same weights
        for param, param_m in zip(self.main.parameters(), self.ema.parameters()):
            assert param.data.numel() == param_m.data.numel()
            assert torch.allclose(param.data, param_m.data)

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    def momentum_update_ema(self):
        # if self.eman:
        #     state_dict_main = self.main.state_dict()
        #     state_dict_ema = self.ema.state_dict()
        #     for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
        #         assert k_main == k_ema, "state_dict names are different!"
        #         assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
        #         if 'num_batches_tracked' in k_ema:
        #             v_ema.copy_(v_main)
        #         else:
        #             v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
        # else:
        #     for param_q, param_k in zip(self.main.parameters(), self.ema.parameters()):
        #         param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

        state_dict_main = self.main.state_dict()
        state_dict_ema = self.ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * self.momentum + (1. - self.momentum) * v_main)
    
    @torch.no_grad()
    def _update_bank(self, k, labels, index):
        if self.use_ddp:
            k = concat_all_gather(k)
            labels = concat_all_gather(labels)
            index = concat_all_gather(index)
        self.bank[:, index] = k.t()
        self.labels[index] = labels


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
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        if self.use_ddp:
            x_gather = concat_all_gather(x)
        else:
            x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        if self.use_ddp:
            torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        if self.use_ddp:
            gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        if self.use_ddp:
            x_gather = concat_all_gather(x)
        else:
            x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        if self.use_ddp:
            gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_x, im_u_w=None, im_u_s=None, labels=None, index=None, start_unlabel=False):
        if im_u_w is None and im_u_s is None:
            logits, _ = self.main(im_x)
            return logits

        if self.eval_datatype == 'imaging':
            batch_x = im_x.shape[0]
            batch_u = im_u_w.shape[0]
        else:
            batch_x = im_x[0].shape[0]
            batch_u = im_u_w[0].shape[0]
        bank = self.bank.clone().detach()

        if self.eval_datatype == 'imaging':
            logits_q, feat_q = self.main(torch.cat((im_x, im_u_s)))
        else:
            logits_q, feat_q = self.main((torch.cat((im_x[0], im_u_s[0])), torch.cat((im_x[1], im_u_s[1]))))
        logits_qx, logits_qu = logits_q[:batch_x], logits_q[batch_x:]
        feat_qx, feat_qu = feat_q[:batch_x], feat_q[batch_x:]

        self.ema.eval()
        with torch.no_grad():
            self.momentum_update_ema()
            if self.eval_datatype == 'imaging':
                im_k = torch.cat((im_x, im_u_w))
            else:
                im_k = (torch.cat((im_x[0], im_u_w[0])), torch.cat((im_x[1], im_u_w[1])))
            # if self.eman:
            #     logits_k, feat_k = self.ema(im_k)
            # else:
            #     im, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            #     logits_k, feat_k = self.ema(im)
            #     feat_k = self._batch_unshuffle_ddp(feat_k, idx_unshuffle)
            #     logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle)
            logits_k, feat_k = self.ema(im_k)
            
            logits_kx, logits_ku = logits_k[:batch_x], logits_k[batch_x:]
            feat_kx, feat_ku = feat_k[:batch_x], feat_k[batch_x:]
            prob_ku_orig = F.softmax(logits_ku, dim=-1)
            if self.DA:
                prob_ku_orig = self.distribution_alignment(prob_ku_orig)
            
        if start_unlabel:
            with torch.no_grad():
                teacher_logits = feat_ku @ bank
                teacher_prob_orig = F.softmax(teacher_logits / self.tt, dim=1)
                factor = prob_ku_orig.gather(1, self.labels.expand([batch_u, -1]))
                teacher_prob = teacher_prob_orig * factor
                teacher_prob /= torch.sum(teacher_prob, dim=1, keepdim=True)

                if self.c_smooth < 1:
                    bs = teacher_prob_orig.size(0)
                    aggregated_prob = torch.zeros([bs, self.num_classes], device=teacher_prob_orig.device)
                    aggregated_prob = aggregated_prob.scatter_add(1, self.labels.expand([bs,-1]) , teacher_prob_orig)
                    prob_ku = prob_ku_orig * self.c_smooth + aggregated_prob * (1-self.c_smooth)
                else:
                    prob_ku = prob_ku_orig

            student_logits = feat_qu @ bank
            student_prob = F.softmax(student_logits / self.st, dim=1)
            loss_in = torch.sum(-teacher_prob.detach() * torch.log(student_prob), dim=1)
        else:
            loss_in = torch.tensor(0, dtype=torch.float).cuda()
            prob_ku = prob_ku_orig

        self._update_bank(feat_kx, labels, index)
        return logits_qx, prob_ku, logits_qu, loss_in


# def get_simmatch_model(model):
#     if isinstance(model, str):
#         model = {
#             "SimMatch": SimMatch,
#         }[model]
#     return model



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == "__main__":
    args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'SimMatch',
                  'num_classes': 286, 
                  'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                    'imaging_pretrained': False, 
                    'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                    'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0,
                    'finetune_strategy':'trainable', 'checkpoint': False, 'K': 2,'eval_datatype': 'imaging',
                    'pretrained_model': 'TIP',
                    'DA': True, 'c_smooth': 0.1, 'st': 0.05, 'tt': 0.05, 'ema_momentum': 0.999, 'projection_dim': 128})
    # '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt'
    print(args.multimodal_embedding_dim)
    model = SimMatchModel(args)
    x = torch.randn(2, 3, 128, 128)
    x_u_w = torch.randn(3, 3, 128, 128)
    x_u_s = torch.randn(3, 3, 128, 128)
    labels = torch.tensor([0,1])
  
  
    # y = model(x, x_u_w, x_u_s, labels, [0,1], True)
    # for item in y:
    #     print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')
    num_infer_params = sum(p.numel() for p in model.main.parameters() if p.requires_grad) - sum(p.numel() for p in model.main.head.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_infer_params/1e6}M')

# =====================================  multimodal =====================================

    args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'SimMatch',
                  'num_classes': 286, 
                  'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                    'imaging_pretrained': False, 
                    'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                    'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0,
                    'finetune_strategy':'trainable', 'checkpoint': False, 'K': 2,'eval_datatype': 'imaging_and_tabular',
                    'pretrained_model': 'TIP',
                    'DA': True, 'c_smooth': 0.1, 'st': 0.05, 'tt': 0.05, 'ema_momentum': 0.999, 'projection_dim': 128})
    # '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt'
    print(args.multimodal_embedding_dim)
    model = SimMatchModel(args)
    x = torch.randn(2, 3, 128, 128)
    x_u_w = torch.randn(3, 3, 128, 128)
    x_u_s = torch.randn(3, 3, 128, 128)
    x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
    x_t_u = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
    labels_l = torch.tensor([0,1])
    labels_u = torch.tensor([0,1,0])
  
  
    # y = model((x,x_t), (x_u_w,x_t_u), (x_u_s,x_t_u), labels_l, [0,1], True)
    # for item in y:
    #     print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')
    num_infer_params = sum(p.numel() for p in model.main.parameters() if p.requires_grad) - sum(p.numel() for p in model.main.head.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_infer_params/1e6}M')