import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .model_base import *
from .model_fn import *

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class ShortcutLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(ShortcutLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride

    def forward(self, x):
        if self.stride != 1 or self.in_planes != self.out_planes:
            return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, self.out_planes // 4, self.out_planes // 4), "constant", 0)
        else:
            return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, planes)

        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = ShortcutLayer(in_planes, planes, stride)
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(1, self.expansion * planes)
                )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet20(nn.Module, Model_fn):
    def __init__(self, args, num_classes=10):
        super(ResNet20, self).__init__()
        self.args = args
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, 16)
        self.layer1 = self._make_layer(BasicBlock, 16, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 3, stride=2)
        self.linear = nn.Linear(64, num_classes)
        
        Model_fn.__init__(self, args, loss_fn=torch.nn.CrossEntropyLoss())

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x['pixels']
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

a=1