
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
from cosplace_model.mixvpr import MixVPR
from cosplace_model.layers import Flatten, L2Norm, GeM, AdaptivePool, NetVLAD
import parser

args = parser.parse_arguments(is_training=False)

# The number of channels in the last convolutional layer, the one before average pooling
CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "ResNet152": 2048,
    "VGG16": 512,
    "regnetx_002":128
}

class Debug(nn.Module):
    def forward(self, x):
        return x

class GeoLocalizationNet(nn.Module):
    def __init__(self, backbone: str, fc_output_dim: int, aggregation_type: str = args.aggregation_type):

        """Return a model for GeoLocalization.

        Args:
            backbone (str): which backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
            aggregation_type (str): the type of aggregation layer to use. One of ['NetVLAD', 'MixVPR', 'GeM'].

        """
        super().__init__()
        assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone, features_dim = get_backbone(backbone)
        
        # Choose the aggregation layer based on the argument.
        if aggregation_type == 'NetVLAD':
            self.aggregation = nn.Sequential(
                NetVLAD(clusters_num=15, dim=features_dim),
                nn.Linear(fc_output_dim, fc_output_dim),
                L2Norm()
            )
        elif aggregation_type == 'MixVPR':
            self.aggregation = nn.Sequential(
                L2Norm(),
                Debug(),
                MixVPR(in_channels=features_dim),
                Debug(),
                Debug(),
                L2Norm()
            )
        elif aggregation_type == 'GeM':
            self.aggregation = nn.Sequential(
                L2Norm(),
                GeM(),
                Flatten(),
                nn.Linear(fc_output_dim, fc_output_dim),
                L2Norm()
            )
        else:
            raise ValueError(f"Unknown aggregation_type: {aggregation_type}")
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x

        
def get_pretrained_torchvision_model(backbone_name : str) -> torch.nn.Module:
    """This function takes the name of a backbone and returns the corresponding pretrained
    model from torchvision. Examples of backbone_name are 'VGG16' or 'ResNet18'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model

   

def get_backbone(backbone_name: str) -> Tuple[torch.nn.Module, int]:
    if backbone_name in CHANNELS_NUM_IN_LAST_CONV:
        if backbone_name.startswith("ResNet"):
            backbone = get_pretrained_torchvision_model(backbone_name)
            for name, child in backbone.named_children():
                if name == "layer3":  # Freeze layers before conv_3
                    break
                for params in child.parameters():
                    params.requires_grad = False
            logging.debug(f"Train only layer3 and layer4 of the {backbone_name}, freeze the previous ones")
            layers = list(backbone.children())[:-2]  # Remove avg pooling and FC layer

        elif backbone_name == "VGG16":
            backbone = get_pretrained_torchvision_model(backbone_name)
            layers = list(backbone.features.children())[:-2]  # Remove avg pooling and FC layer

            for layer in layers[:-5]:
                for p in layer.parameters():
                    p.requires_grad = False
            logging.debug("Train last layers of the VGG-16, freeze the previous ones")

        backbone = torch.nn.Sequential(*layers)
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    return backbone, features_dim