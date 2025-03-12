from torch.nn import Module
from torch.nn import Identity
import sys
sys.path.append('/home/siyi/project/mm/STiL')
from models import resnets


def torchvision_ssl_encoder(
    name: str,
    pretrained: bool = False,
    return_all_feature_maps: bool = False,
) -> Module:
    pretrained_model = getattr(resnets, name)(pretrained=pretrained, return_all_feature_maps=return_all_feature_maps)
    pretrained_model.fc = Identity()   
    return pretrained_model