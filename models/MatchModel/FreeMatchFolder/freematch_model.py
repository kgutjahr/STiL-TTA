'''
Built from FreeMatch https://github.com/microsoft/Semi-supervised-learning 
'''
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.utils.self_supervised import torchvision_ssl_encoder

import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.MatchModel.multimodal_backbone import MultimodalBackbone
from models.pieces import DotDict
from models.MatchModel.FreeMatchFolder.freematch_utils import entropy_loss


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


class FreeMatchModel(nn.Module):
    def __init__(self, args):
        super(FreeMatchModel, self).__init__()
        self.eman = False
        self.momentum = args.ema_momentum
        self.num_classes = args.num_classes
        self.eval_datatype = args.eval_datatype
        self.pooled_dim = args.embedding_dim
        self.dim = args.projection_dim

        self.m = 0.999
        self.clip_thresh = 0.0
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()

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
    def update(self, probs_x_ulb):
        if self.use_ddp:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        # if algorithm.use_quantile:
        #     self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        # else:
        self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if self.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(probs_x_ulb)

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask

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

        if self.eval_datatype == 'imaging':
            logits_q, _ = self.main(torch.cat((im_x, im_u_s)))
        else:
            logits_q, _ = self.main((torch.cat((im_x[0], im_u_s[0])), torch.cat((im_x[1], im_u_s[1]))))
        logits_x_lb, logits_x_ulb_s = logits_q[:batch_x], logits_q[batch_x:]

        self.ema.eval()
        with torch.no_grad():
            self.momentum_update_ema()
            logits_x_ulb_w, feat_k = self.ema(im_u_w)
            mask = self.masking(logits_x_ulb_w)
            max_probs, max_idx = torch.max(torch.softmax(logits_x_ulb_w.detach(), dim=-1), dim=-1)
            pseudo_label = torch.zeros_like(logits_x_ulb_w)
            pseudo_label[torch.arange(batch_u), max_idx] = 1

        # calculate entropy loss
        if mask.sum() > 0:
            ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
        else:
            ent_loss = 0.0

        return logits_x_lb, pseudo_label, logits_x_ulb_s, mask, ent_loss


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
                  'num_cat': 26, 'num_con': 49, 'num_classes': 2, 
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
    model = FreeMatchModel(args)
    x = torch.randn(2, 3, 128, 128)
    x_u_w = torch.randn(3, 3, 128, 128)
    x_u_s = torch.randn(3, 3, 128, 128)
    labels = torch.tensor([0,1])
  
  
    y = model(x, x_u_w, x_u_s, labels, [0,1], True)
    for item in y:
        print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')


    args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'SimMatch',
                  'num_cat': 26, 'num_con': 49, 'num_classes': 2, 
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
    model = FreeMatchModel(args)
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
  
  
    y = model((x,x_t), (x_u_w,x_t_u), (x_u_s,x_t_u), labels_l, [0,1], True)
    for item in y:
        print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')