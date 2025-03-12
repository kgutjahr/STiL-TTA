'''
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2025
'''

import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig, open_dict, OmegaConf

import sys
# TODO: Change the path to your own project directory if you want to run this file alone for debugging 
sys.path.append('/home/siyi/project/mm/STiL')
from models.self_supervised import torchvision_ssl_encoder

from models.Disentangle.utils.SAINT.Tabular_Encoder import SAINT
from models.pieces import DotDict
from models.Disentangle.utils.disentangle_transformer import MITransformerLayer


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class DisCoAttentionBackbone(nn.Module):
    '''
    Input: image, tabular
    Output: 
        image features: x_si, x_ai
        tabular features: x_st, x_at
        prediction
    '''
    def __init__(self, args) -> None:
        super(DisCoAttentionBackbone, self).__init__()

        self.create_imaging_model(args)
        self.create_tabular_model(args)
        self.pooled_dim = args.embedding_dim
        self.hidden_dim = args.multimodal_embedding_dim
        # disentangle projection
        self.projection_si = MLP(self.pooled_dim, self.hidden_dim, self.hidden_dim)
        self.projection_ai = MLP(self.pooled_dim, self.hidden_dim, self.hidden_dim)
        self.projection_st = MLP(self.tabular_embedding_dim, self.tabular_embedding_dim, self.hidden_dim)
        self.projection_at = MLP(self.tabular_embedding_dim, self.tabular_embedding_dim, self.hidden_dim)
        self.reduce = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.transformer = nn.ModuleList([
                            MITransformerLayer(dim=self.hidden_dim, num_heads=4, mlp_ratio=1.0, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, drop_path=0.1)
                            for i in range(args.multimodal_transformer_num_layers)
                            ])
        if args.pretrain == True and args.checkpoint is None:
            print('Pretrain model does not have aggregation and classifier')
        else:
            self.classifier_multimodal = nn.Linear(self.hidden_dim*3, args.num_classes)
            self.classifier_imaging = nn.Linear(self.hidden_dim*2, args.num_classes)
            self.classifier_tabular = nn.Linear(self.hidden_dim*2, args.num_classes)
        if args.checkpoint: 
            print(f'Checkpoint name: {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint)
            original_args = OmegaConf.create(checkpoint['hyper_parameters'])
            state_dict = checkpoint['state_dict']
            if args.pretrained_model == 'TIP':
                # load image encoder
                for module, module_name in zip([self.encoder_imaging], 
                                                ['encoder_imaging.']):
                    self.load_weights(module, module_name, state_dict)
                    if args.finetune_strategy == 'frozen':
                        for _, param in module.named_parameters():
                            param.requires_grad = False
                        parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
                        assert len(parameters)==0
                        print(f'Freeze {module_name}')
                    elif args.finetune_strategy == 'trainable':
                        print(f'Full finetune {module_name}')
                    else:
                        assert False, f'Unknown finetune strategy {args.finetune_strategy}'
            else:
                raise Exception(f'Unknown pretrain model: {args.pretrained_model}')
    
    def create_imaging_model(self, args):
        self.encoder_imaging = torchvision_ssl_encoder(args.model, return_all_feature_maps=True)

    def create_tabular_model(self, args):
        self.field_lengths_tabular = torch.load(args.field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        self.cat_cols = []
        self.con_cols = []
        for id, x in enumerate(self.field_lengths_tabular):
            if x == 1:
                self.con_lengths_tabular.append(x) 
                self.con_cols.append(id)
            else:
                self.cat_lengths_tabular.append(x)
                self.cat_cols.append(id)

        nfeat = len(self.cat_cols)+len(self.con_cols)+1
        transformer_depth = 6
        embedding_size = 32
        attention_heads = 8
        attentiontype = 'colrow'
        cont_embeddings = 'MLP'
        final_mlp_style = 'sep'
        if nfeat > 100:
            embedding_size = min(8,embedding_size)
        if attentiontype != 'col':
            transformer_depth = 1
            attention_heads = min(4, attention_heads)
            attention_dropout = 0.8
            embedding_size = min(32,embedding_size)
            ff_dropout = 0.8
        
        self.tabular_embedding_dim = embedding_size
        print(f'Tabular embedding dimension {self.tabular_embedding_dim}')

        self.encoder_tabular = SAINT(
            categories = tuple(self.cat_lengths_tabular), 
            num_continuous = len(self.con_cols),                
            dim = embedding_size,                           
            dim_out = 1,                       
            depth = transformer_depth,                       
            heads = attention_heads,                         
            attn_dropout = attention_dropout,             
            ff_dropout = ff_dropout,                  
            mlp_hidden_mults = (4, 2),       
            cont_embeddings = cont_embeddings,
            attentiontype = attentiontype,
            final_mlp_style = final_mlp_style,
            y_dim = args.num_classes,
            num_special_tokens=1
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1))
        if args.checkpoint_SAINT:
            self.encoder_tabular.load_state_dict(torch.load(args.checkpoint_SAINT))
            print(f'Load SAINT weights from {args.checkpoint_SAINT}')


    def load_weights(self, module, module_name, state_dict):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0

    
    def forward_tabular(self, x_t):
        cls_tokens = self.cls_token.expand(x_t.shape[0], -1)
        x_categ = x_t[:, self.cat_cols]
        x_categ = torch.cat((cls_tokens, x_categ), dim=1).long()
        x_cont = x_t[:, self.con_cols]
        x_categ = x_categ + self.encoder_tabular.categories_offset
        x_categ_enc = self.encoder_tabular.embeds(x_categ)

        n1,n2 = x_cont.shape
        _, n3 = x_categ.shape

        if self.encoder_tabular.cont_embeddings == 'MLP':
            x_cont_enc = torch.empty(n1,n2, self.encoder_tabular.dim, device=x_categ.device)
            for i in range(self.encoder_tabular.num_continuous):
                x_cont_enc[:,i,:] = self.encoder_tabular.simple_MLP[i](x_cont[:,i])
        else:
            raise Exception('This case should not work!')    

        pos = np.tile(np.arange(x_categ.shape[-1]),(x_categ.shape[0],1))
        pos =  torch.from_numpy(pos).to(x_categ.device)
        pos_enc = self.encoder_tabular.pos_encodings(pos)
        x_categ_enc += pos_enc

        x_t = self.encoder_tabular.transformer(x_categ_enc, x_cont_enc)

        return x_t

    def forward_encoding_feature(self, x: torch.Tensor, visualize=False) -> torch.Tensor:
        '''Get disentangled features'''
        x_i, x_t = x[0], x[1]
        x_i = self.encoder_imaging(x_i)[-1]   # (B,C,H,W)
        if len(x_i.shape) == 4:
            B,C,H,W = x_i.shape
            x_i = x_i.reshape(B,C,H*W).permute(0,2,1)
        
        x_t = self.forward_tabular(x_t)
        # disentangle image
        x_si = self.projection_si(x_i)  # (B, N_i, C)
        x_ai = self.projection_ai(torch.mean(x_i, dim=1))  # (B, C)
        # disentangle tabular
        x_st = self.projection_st(x_t[:,1:,:])   # (B, N_t, C)
        x_at = self.projection_at(x_t[:,0,:])   # (B, N_t, C)
        return x_si, x_ai, x_st, x_at

    def forward_multimodal_feature(self, x_si: torch.Tensor, x_ai: torch.Tensor, x_st: torch.Tensor, x_at: torch.Tensor, visualize=False) -> torch.Tensor:
        '''Get disentangled and multimodal features'''
        # x_si, x_ai, x_st, x_at = self.forward_feature(x)
        # reduce
        x_c = self.reduce(torch.cat([x_ai, x_at], dim=1)).unsqueeze(1)   # (B, C)
        # attention
        for block in self.transformer:
            x_si, x_st, x_c = block(x_si, x_st, x_c)
        # aggregate 
        x_si = torch.mean(x_si, dim=1)
        x_st = torch.mean(x_st, dim=1)
        x_c = torch.mean(x_c, dim=1)
        return x_si, x_ai, x_st, x_at, x_c
    
    def forward_all(self, x: torch.Tensor, visualize=False) -> torch.Tensor:
        x_si, x_ai, x_st, x_at = self.forward_encoding_feature(x)
        x_si_enhance, x_ai, x_st_enhance, x_at, x_c = self.forward_multimodal_feature(x_si, x_ai, x_st, x_at)
        out_m = self.classifier_multimodal(torch.cat([x_si_enhance, x_c, x_st_enhance], dim=1))
        out_i = self.classifier_imaging(torch.cat([x_si_enhance, x_ai], dim=1))
        out_t = self.classifier_tabular(torch.cat([x_st_enhance, x_at], dim=1))
        return out_m, out_i, out_t, x_si_enhance, torch.mean(x_si,dim=1), x_ai, x_st_enhance, torch.mean(x_st,dim=1), x_at, x_c
    
    def forward(self, x: torch.Tensor, visualize=False) -> torch.Tensor:
        x_si, x_ai, x_st, x_at = self.forward_encoding_feature(x)
        x_si_enhance, x_ai, x_st_enhance, x_at, x_c = self.forward_multimodal_feature(x_si, x_ai, x_st, x_at)
        out_m = self.classifier_multimodal(torch.cat([x_si_enhance, x_c, x_st_enhance], dim=1))
        out_i = self.classifier_imaging(torch.cat([x_si_enhance, x_ai], dim=1))
        out_t = self.classifier_tabular(torch.cat([x_st_enhance, x_at], dim=1))
        return out_m, out_i, out_t, x_si_enhance, x_ai, x_st_enhance, x_at, x_c



if __name__ == "__main__":
    args = DotDict({'model': 'resnet50', 'checkpoint': '/vol/biomedic3/sd1523/project/mm/result/valid/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt', 'algorithm_name': 'DISCO',
                    'num_cat': 26, 'num_con': 49, 'num_classes': 286, 
                    'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                    'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                        'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                        'imaging_pretrained': False, 
                        'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                        'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0,
                        'finetune_strategy':'frozen', 'checkpoint': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt',
                        'checkpoint_SAINT': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/FD10/3e5_dvm_0121_1649/pretrained_last_499.pth',
                        'pretrained_model': 'TIP'})
    print(args.multimodal_embedding_dim)
    model = DisCoAttentionBackbone(args)
    x_i = torch.randn(2, 3, 128, 128)
    x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                        [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
    
    y = model.forward_all(x=(x_i,x_t))
    for item in y:
        print(item.shape)

    # count num params in classifier_imaging, classifier_tabular
    count = 0
    for key, value in model.state_dict().items():
        # if 'classifier_imaging' in key or 'classifier_tabular' in key:
        count += value.numel()
        if value.dtype != torch.float32:
            print(key, value.dtype)
    print(f'{count/1e6}M parameters')
