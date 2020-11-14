import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

import torchvision.models.resnet as torch_resnet
from torchvision.models.resnet import BasicBlock, Bottleneck

model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

class ResNet(torch_resnet.ResNet):
    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)

    def modify(self, remove_layers=[], padding=''):
        # Set stride of layer3 and layer 4 to 1 (from 2)
        filter_layers = lambda x: [l for l in x if getattr(self, l) is not None]
        for layer in filter_layers(['layer3', 'layer4']):
            for m in getattr(self, layer).modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.stride = tuple(1 for _ in m.stride)
                    print('stride', m)
        # Set padding (zeros or reflect, doesn't change much; 
        # zeros requires lower temperature)
        if padding != '':
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d) and sum(m.padding) > 0:
                    m.padding_mode = padding
                    print('padding', m)

        # Remove extraneous layers
        remove_layers += ['fc', 'avgpool']
        for layer in filter_layers(remove_layers):
            setattr(self, layer, None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x if self.maxpool is None else self.maxpool(x) 

        x = self.layer1(x)
        x = self.layer2(x)
        x = x if self.layer3 is None else self.layer3(x) 
        x = x if self.layer4 is None else self.layer4(x) 
    
        return x        


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs) -> ResNet:
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)