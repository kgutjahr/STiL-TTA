'''
Multimodal backbone model
Used for multimodal CoMatch, FreeMatch, and SimMatch
* Licensed under the Apache License, Version 2.
* By Siyi Du, 2025
'''

import torch
import torch.nn as nn
from omegaconf import DictConfig, open_dict, OmegaConf
import torch.nn.functional as F
from einops import rearrange
import sys
sys.path.append('/home/siyi/project/mm/STiL')
from models.self_supervised import torchvision_ssl_encoder

from models.Transformer import TabularTransformerEncoder
from models.pieces import DotDict


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


class MultimodalBackbone(nn.Module):
    '''
    Input: image, tabular
    Output: 
        image features: x_si, x_ai
        tabular features: x_st, x_at
        prediction
    '''
    def __init__(self, args) -> None:
        super(MultimodalBackbone, self).__init__()

        self.create_imaging_model(args)
        self.create_tabular_model(args)
        self.pooled_dim = args.embedding_dim
        self.hidden_dim = args.multimodal_embedding_dim
        self.image_proj = nn.Linear(self.pooled_dim, self.hidden_dim)
        self.tabular_proj = nn.Linear(args.tabular_embedding_dim, args.multmimodal_embedding_dim) if args.tabular_embedding_dim != args.multimodal_embedding_dim else nn.Identity()
        if args.pretrain == True and args.checkpoint is None:
            print('Pretrain model does not have aggregation and classifier')
        else:
            self.head = nn.Sequential(
                        nn.Linear(self.hidden_dim*2, self.hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.hidden_dim, args.projection_dim),
                    )
            self.classifier_multimodal = nn.Linear(2*self.hidden_dim, args.num_classes)
        if args.checkpoint: 
            print(f'Checkpoint name: {args.checkpoint}')
            checkpoint = torch.load(args.checkpoint)
            original_args = OmegaConf.create(checkpoint['hyper_parameters'])
            state_dict = checkpoint['state_dict']
            if args.pretrained_model == 'TIP':
                # load image and tabular encoders
                for module, module_name in zip([self.encoder_imaging, self.encoder_tabular], 
                                                ['encoder_imaging.', 'encoder_tabular.']):
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
    
    def create_imaging_model(self, args):
        self.encoder_imaging = torchvision_ssl_encoder(args.model)

    def create_tabular_model(self, args):
        self.field_lengths_tabular = torch.load(args.field_lengths_tabular)
        self.cat_lengths_tabular = []
        self.con_lengths_tabular = []
        for x in self.field_lengths_tabular:
            if x == 1:
                self.con_lengths_tabular.append(x) 
            else:
                self.cat_lengths_tabular.append(x)
        self.encoder_tabular = TabularTransformerEncoder(args, self.cat_lengths_tabular, self.con_lengths_tabular)

    def load_weights(self, module, module_name, state_dict, finetune='trainable'):
        state_dict_module = {}
        for k in list(state_dict.keys()):
            if k.startswith(module_name) and not 'projection_head' in k and not 'prototypes' in k:
                state_dict_module[k[len(module_name):]] = state_dict[k]
        print(f'Load {len(state_dict_module)}/{len(state_dict)} weights for {module_name}')
        log = module.load_state_dict(state_dict_module, strict=True)
        assert len(log.missing_keys) == 0
        if finetune == 'frozen':
            for _, param in module.named_parameters():
                param.requires_grad = False
            parameters = list(filter(lambda p: p.requires_grad, module.parameters()))
            assert len(parameters)==0
            print(f'Freeze {module_name}')
        elif finetune == 'trainable':
            print(f'Full finetune {module_name}')
        else:
            assert False, f'Unknown finetune strategy {finetune}'

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_i, x_t = x[0], x[1]
        x_i = self.encoder_imaging(x_i)[0].squeeze()
        x_t = self.encoder_tabular(x_t)
        x_m = torch.cat([self.image_proj(x_i), self.tabular_proj(x_t[:,0,:])], dim=1)
        embedding = self.head(x_m)
        logits = self.classifier_multimodal(x_m)
        return logits, F.normalize(embedding)

    
 

if __name__ == "__main__":
  args = DotDict({'model': 'resnet50', 'checkpoint': None, 'algorithm_name': 'DISCO',
                  'num_cat': 26, 'num_con': 49, 'num_classes': 2, 
                  'field_lengths_tabular': '/vol/biomedic3/sd1523/data/mm/DVM/features/tabular_lengths_all_views_physical_reordered.pt',
                  'tabular_embedding_dim': 512, 'tabular_transformer_num_layers': 4, 'multimodal_transformer_layers': 4,'embedding_dropout': 0.0, 'drop_rate':0.0,
                    'multimodal_embedding_dim': 512, 'multimodal_transformer_num_layers': 4,
                    'imaging_pretrained': False, 
                    'img_size': 128, 'patch_size': 16, 'embedding_dim': 2048, 'mlp_ratio': 4.0, 'num_heads': 6, 'depth': 12,
                    'attention_dropout_rate': 0.0, 'imaging_dropout_rate': 0.0, 'projection_dim':128,
                    'finetune_strategy':'trainable', 'checkpoint': '/vol/biomedic3/sd1523/project/mm/result/TIP_results/D20/MaskAttn_ran00spec05_dvm_0104_0938/checkpoint_last_epoch_499.ckpt',
                    'pretrained_model': 'TIP'})
  print(args.multimodal_embedding_dim)
  model = MultimodalBackbone(args)
  x_i = torch.randn(2, 3, 128, 128)
  x_t = torch.tensor([[4.0, 3.0, 0.0, 2.0, 0.2, -0.1,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1],
                    [2.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2,  -0.5, 0.2, 0.1]], dtype=torch.float32)
  
  y = model.forward(x=(x_i,x_t))
  for item in y:
      print(item.shape)
