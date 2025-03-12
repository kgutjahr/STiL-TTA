'''
Built from CoMatch https://github.com/salesforce/CoMatch/tree/main
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CoMatchModel(nn.Module):

    def __init__(self, args):

        super(CoMatchModel, self).__init__()
        
        self.K = args.K
        self.momentum = args.ema_momentum
        self.num_classes = args.num_classes
        self.pooled_dim = args.embedding_dim
        self.low_dim = args.projection_dim
        self.temperature = args.co_temperature
        self.alpha = args.alpha
        self.start_epoch = args.start_epoch
        
        if args.eval_datatype == 'imaging':
            self.encoder = ResNet(args, self.num_classes, out_channels=self.pooled_dim, dim=self.low_dim)
            self.m_encoder = ResNet(args, self.num_classes, out_channels=self.pooled_dim, dim=self.low_dim)
            print(f'Using imaging encoder for CoMatch')
        elif args.eval_datatype == 'imaging_and_tabular':
            self.encoder = MultimodalBackbone(args)
            self.m_encoder = MultimodalBackbone(args)
            print(f'Using multimodal encoder for CoMatch')
        else:
            assert False, f'Unknown eval_datatype {args.eval_datatype}'
        self.eval_datatype = args.eval_datatype

        if args.checkpoint and args.eval_datatype == 'imaging':
            print(f'Load weights for the image backbone from {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint)
            state_dict = checkpoint['state_dict']
            self.load_weights(self.encoder.backbone, 'encoder_imaging.', state_dict)
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
        
        for param, param_m in zip(self.encoder.parameters(), self.m_encoder.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient
        
        # queue to store momentum feature for strong augmentations
        self.register_buffer("queue_s", torch.randn(self.low_dim, self.K))        
        self.queue_s = F.normalize(self.queue_s, dim=0)
        self.register_buffer("queue_ptr_s", torch.zeros(1, dtype=torch.long))       
        # queue to store momentum probs for weak augmentations (unlabeled)
        self.register_buffer("probs_u", torch.zeros(self.num_classes, self.K)) 
        
        # queue (memory bank) to store momentum feature and probs for weak augmentations (labeled and unlabeled)
        self.register_buffer("queue_w", torch.randn(self.low_dim, self.K))  
        self.register_buffer("queue_ptr_w", torch.zeros(1, dtype=torch.long))
        self.register_buffer("probs_xu", torch.zeros(self.num_classes, self.K)) 
        
        # for distribution alignment
        self.hist_prob = []

        self.use_ddp = torch.cuda.device_count() > 1
        # self.use_ddp = False
        

    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    @torch.no_grad()
    def _update_momentum_encoder(self,m):
        """
        Update momentum encoder
        """
        for param, param_m in zip(self.encoder.parameters(), self.m_encoder.parameters()):
            param_m.data = param_m.data * m + param.data * (1. - m)
            
    @torch.no_grad()
    def _dequeue_and_enqueue(self, z, t, ws):
        if self.use_ddp:
            z = concat_all_gather(z)
            t = concat_all_gather(t)

        batch_size = z.shape[0]
  
        if ws=='s':
            ptr = int(self.queue_ptr_s)
            if (ptr + batch_size) > self.K:
                batch_size = self.K-ptr
                z = z[:batch_size]
                t = t[:batch_size]            
            # replace the samples at ptr (dequeue and enqueue)
            self.queue_s[:, ptr:ptr + batch_size] = z.T
            self.probs_u[:, ptr:ptr + batch_size] = t.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr_s[0] = ptr
            
        elif ws=='w':
            ptr = int(self.queue_ptr_w)
            if (ptr + batch_size) > self.K:
                batch_size = self.K-ptr
                z = z[:batch_size]
                t = t[:batch_size]               
            # replace the samples at ptr (dequeue and enqueue)
            self.queue_w[:, ptr:ptr + batch_size] = z.T
            self.probs_xu[:, ptr:ptr + batch_size] = t.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr_w[0] = ptr
        
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
    
    def forward(self, labeled_batch, unlabeled_batch=None, is_eval=False, epoch=0):    
        
        # img_x = labeled_batch[0].cuda(args.gpu, non_blocking=True)  
        # labels_x = labeled_batch[1].cuda(args.gpu, non_blocking=True)  
        img_x = labeled_batch[0]
        labels_x = labeled_batch[1]
        
        if is_eval:        
            outputs_x, _ = self.encoder(img_x)      
            return outputs_x
        
        if self.eval_datatype == 'imaging':
            btx = img_x.size(0)
        else:
            btx = img_x[0].size(0)
        
        # img_u_w = unlabeled_batch[0][0].cuda(args.gpu, non_blocking=True)  
        # img_u_s0 = unlabeled_batch[0][1].cuda(args.gpu, non_blocking=True)   
        # img_u_s1 = unlabeled_batch[0][2].cuda(args.gpu, non_blocking=True)   
        img_u_w, img_u_s0, img_u_s1 = unlabeled_batch[0][0], unlabeled_batch[0][1], unlabeled_batch[0][2]
        
        if self.eval_datatype == 'imaging':
            btu = img_u_w.size(0)
        else:
            btu = img_u_w[0].size(0)
        
        if self.eval_datatype == 'imaging':
            imgs = torch.cat([img_x, img_u_s0], dim=0)
        else:
            imgs = (torch.cat([img_x[0], img_u_s0[0]], dim=0), torch.cat([img_x[1], img_u_s0[1]], dim=0))
        outputs, features = self.encoder(imgs)

        outputs_x = outputs[:btx]
        outputs_u_s0 = outputs[btx:]      
        features_u_s0 = features[btx:]
        
        with torch.no_grad(): 
            self._update_momentum_encoder(self.momentum)
            # forward through the momentum encoder
            if self.eval_datatype == 'imaging':
                imgs_m = torch.cat([img_x, img_u_w, img_u_s1], dim=0)       
            else:
                imgs_m = (torch.cat([img_x[0], img_u_w[0], img_u_s1[0]], dim=0), torch.cat([img_x[1], img_u_w[1], img_u_s1[1]], dim=0))     
            # imgs_m, idx_unshuffle = self._batch_shuffle_ddp(imgs_m)
            
            outputs_m, features_m = self.m_encoder(imgs_m)
            # outputs_m = self._batch_unshuffle_ddp(outputs_m, idx_unshuffle)
            # features_m = self._batch_unshuffle_ddp(features_m, idx_unshuffle)
            
            outputs_u_w = outputs_m[btx:btx+btu]
            
            feature_u_w = features_m[btx:btx+btu]
            feature_xu_w = features_m[:btx+btu]
            features_u_s1 = features_m[btx+btu:]
            
            outputs_u_w = outputs_u_w.detach()
            feature_u_w = feature_u_w.detach()
            feature_xu_w = feature_xu_w.detach()
            features_u_s1 = features_u_s1.detach()
            
            probs = torch.softmax(outputs_u_w, dim=1)         
            
            # distribution alignment
            probs_bt_avg = probs.mean(0)
            if self.use_ddp:
                torch.distributed.all_reduce(probs_bt_avg,async_op=False)
                world_size = torch.distributed.get_world_size()
            else:
                world_size = 1
            self.hist_prob.append(probs_bt_avg/world_size)

            if len(self.hist_prob)>128:
                self.hist_prob.pop(0)

            probs_avg = torch.stack(self.hist_prob,dim=0).mean(0)
            probs = probs / probs_avg
            probs = probs / probs.sum(dim=1, keepdim=True)             
            probs_orig = probs.clone()
            
            # memory-smoothed pseudo-label refinement (starting from 2nd epoch)
            if epoch>self.start_epoch:                   
                m_feat_xu = self.queue_w.clone().detach()
                m_probs_xu = self.probs_xu.clone().detach()
                A = torch.exp(torch.mm(feature_u_w, m_feat_xu)/self.temperature)       
                A = A/A.sum(1,keepdim=True)                    
                probs = self.alpha*probs + (1-self.alpha)*torch.mm(A, m_probs_xu.t())  
            
            # construct pseudo-label graph
            
            # similarity with current batch
            Q_self = torch.mm(probs,probs.t())  
            Q_self.fill_diagonal_(1)    
            
            # similarity with past samples
            m_probs_u = self.probs_u.clone().detach()
            Q_past = torch.mm(probs,m_probs_u)  

            # concatenate them
            Q = torch.cat([Q_self,Q_past],dim=1)
        
        # construct embedding graph for strong augmentations
        sim_self = torch.exp(torch.mm(features_u_s0, features_u_s1.t())/self.temperature)         
        m_feat = self.queue_s.clone().detach()
        sim_past = torch.exp(torch.mm(features_u_s0, m_feat)/self.temperature)                 
        sim = torch.cat([sim_self,sim_past],dim=1)      
        
        # store strong augmentation features and probs (unlabeled) into momentum queue 
        self._dequeue_and_enqueue(features_u_s1, probs, 's') 
        
        # store weak augmentation features and probs (labeled and unlabeled) into memory bank
        onehot = torch.zeros(btx,self.num_classes, device=labels_x.device).scatter(1,labels_x.view(-1,1),1)
        probs_xu = torch.cat([onehot, probs_orig],dim=0)
        
        self._dequeue_and_enqueue(feature_xu_w, probs_xu, 'w') 
        
        return outputs_x, outputs_u_s0, labels_x, probs, Q, sim
    
    
    

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




if __name__ == "__main__":
    args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'SimMatch',
                  'num_classes': 286, 
                  'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                    'imaging_pretrained': False, 
                    'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                    'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0,
                    'finetune_strategy':'trainable', 'checkpoint': False,
                    'pretrained_model': 'TIP', 'eval_datatype':'imaging', 'start_epoch': 0,
                    'alpha': 0.9, 'co_temperature': 0.1, 'ema_momentum': 0.999, 'projection_dim': 128, 'K': 2560})
    # '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt'
    # '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt'
    # '/bigdata/siyi/data/UKBB/cardiac_segmentations/projects/SelfSuperBio/18545/final/tabular_lengths_reordered.pt'
    print(args.multimodal_embedding_dim)
    model = CoMatchModel(args)
    x = torch.randn(2, 3, 128, 128)
    x_u_w = torch.randn(3, 3, 128, 128)
    x_u_s0 = torch.randn(3, 3, 128, 128)
    x_u_s1 = torch.randn(3, 3, 128, 128)
    labels_l = torch.tensor([0,1])
    labels_u = torch.tensor([0,1,0])
  
  
    # y = model((x, labels_l), ((x_u_w, x_u_s0, x_u_s1),labels_u), is_eval=False, epoch=0)
    # for item in y:
    #     print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')
    # get ResNet backbone and classifier params 
    num_params = sum(p.numel() for p in model.encoder.backbone.parameters() if p.requires_grad) + sum(p.numel() for p in model.encoder.classifier.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')


    args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'SimMatch',
                  'num_classes': 286, 
                  'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                    'imaging_pretrained': False, 
                    'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                    'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0,
                    'finetune_strategy':'trainable', 'checkpoint': False, 
                    'pretrained_model': 'TIP', 'eval_datatype':'imaging_and_tabular', 'start_epoch': 0,
                    'alpha': 0.9, 'co_temperature': 0.1, 'ema_momentum': 0.999, 'projection_dim': 128, 'K': 2560})
    # '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt'
    print(args.multimodal_embedding_dim)
    model = CoMatchModel(args)
    x = torch.randn(2, 3, 128, 128)
    x_u_w = torch.randn(3, 3, 128, 128)
    x_u_s0 = torch.randn(3, 3, 128, 128)
    x_u_s1 = torch.randn(3, 3, 128, 128)
    x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
    x_t_u = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
    labels_l = torch.tensor([0,1])
    labels_u = torch.tensor([0,1,0])
  
  
    # y = model(((x,x_t), labels_l), (((x_u_w,x_t_u), (x_u_s0,x_t_u), (x_u_s1,x_t_u)), labels_u), is_eval=False, epoch=0)
    # for item in y:
    #     print(item.shape)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6}M')
    print(f'Number of trainable parameters: {num_trainable_params/1e6}M')

    num_infer_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad) - sum(p.numel() for p in model.encoder.head.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_infer_params/1e6}M')