
import torch
import logging
import torchvision
from torch import nn
from typing import Tuple
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from cosplace_model.layers import Flatten, L2Norm, GeM, FeatureMixerLayer, MixVPR

# The number of channels in the last convolutional layer, the one before average pooling
#CHANNELS_NUM_IN_LAST_CONV = {
    #"ResNet18":512 default if you don't freeze the last four layers
    #"ResNet50": 2048,
    #"ResNet101": 2048,
    #"ResNet152": 2048,
    #"VGG16": 512,
#}

class ResNet(nn.Module):
    def __init__(self,
                 model_name='resnet18',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[4],
                 ):
      
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze

        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None

        if 'swsl' in model_name or 'ssl' in model_name:
            # These are the semi supervised and weakly semi supervised weights from Facebook
            self.model = torch.hub.load(
                'facebookresearch/semi-supervised-ImageNet1K-models', model_name)
        else:
            if 'resnext50' in model_name:
                self.model = torchvision.models.resnext50_32x4d(weights=weights)
            elif 'resnet50' in model_name:
                self.model = torchvision.models.resnet50(weights=weights)
            elif '101' in model_name:
                self.model = torchvision.models.resnet101(weights=weights)
            elif '152' in model_name:
                self.model = torchvision.models.resnet152(weights=weights)
            elif '34' in model_name:
                self.model = torchvision.models.resnet34(weights=weights)
            elif '18' in model_name:
                # self.model = torchvision.models.resnet18(pretrained=False)
                self.model = torchvision.models.resnet18(weights=weights)
            elif 'wide_resnet50_2' in model_name:
                self.model = torchvision.models.wide_resnet50_2(weights=weights)
            else:
                raise NotImplementedError(
                    'Backbone architecture not recognized!')

        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None

        out_channels = 2048
        if '34' in model_name or '18' in model_name:
            out_channels = 512
            
        self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
        self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.model.layer3 is not None:
            x = self.model.layer3(x)
        if self.model.layer4 is not None:
            x = self.model.layer4(x)
        return x

class GeoLocalizationNet(nn.Module):

    def __init__(self, backbone : nn.Module, fc_output_dim: int):
        """Return a model for GeoLocalization.
        
        Args:
            backbone (str): which torchvision backbone to use. Must be VGG16 or a ResNet.
            fc_output_dim (int): the output dimension of the last fc layer, equivalent to the descriptors dimension.
        """
        super().__init__()
        #assert backbone in CHANNELS_NUM_IN_LAST_CONV, f"backbone must be one of {list(CHANNELS_NUM_IN_LAST_CONV.keys())}"
        self.backbone=backbone
        self.fc_output_dim = backbone.out_channels //1024
       # self.aggregation = nn.Sequential(
        #    L2Norm(),
        #    GeM(),
        #    Flatten(),
        #    nn.Linear(features_dim, fc_output_dim),
        #    L2Norm()
       # )
        self.aggregation = nn.Sequential(
            #L2Norm(),
            MixVPR(in_channels=fc_output_dim, out_channels=fc_output_dim),
            #L2Norm()
        )
    
    def forward(self, x):
        x = self.backbone(x).cuda()
        x = F.interpolate(x, size=(20,20), mode="bilinear", align_corners=False).cuda()
        x = self.aggregation(x).cuda()
        return x



