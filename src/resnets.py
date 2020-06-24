
from src.models import SparseCodingLayer_First, SparseCodingLayer_AfterConv, SparseCodingLayer_AfterSparse

DEVICE = "cuda"

#
#   Taken from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
#


"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same/home/controller/Desktop/Research/SparseDebugging/src/resnets.py with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        
    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64//2

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64//2),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64//2, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128//2, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256//2, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512//2, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512//2 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        logits = self.fc(output)

        _, preds = logits.max(dim=1)

        return logits, preds, None 


def resnet34(_, __, ___):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])



# TODO: Add in usecase conditionals

class BasicBlock_Sparse(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block 
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, filter_set_mult, k_div, stride=1):
        super().__init__()

        #residual function
        self.residual_function_0 = SparseCodingLayer_AfterSparse(in_dim=in_channels, out_dim=out_channels, stride=stride, padding=1, filterset_size=round(int(out_channels*filter_set_mult)), k=round(int(out_channels/k_div)))
        self.residual_function_1 = SparseCodingLayer_AfterSparse(in_dim=out_channels, out_dim=out_channels * BasicBlock.expansion, padding=1, filterset_size=round(int((out_channels * BasicBlock.expansion)*filter_set_mult)), k=round(int((out_channels * BasicBlock.expansion)/k_div)))
        
        # TODO: THIS IS A HACK. FIGURE OUT WHY THE USUAL METHOD OF SENDING THE MODEL TO CUDA DOESN'T DETECT THESE.
        self.residual_function_0.to(DEVICE)
        self.residual_function_1.to(DEVICE)

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

            # TODO: THIS IS A HACK. FIGURE OUT WHY THE USUAL METHOD OF SENDING THE MODEL TO CUDA DOESN'T DETECT THESE.
            self.shortcut.to(DEVICE)
        
    def forward(self, x):
        out, aux_loss_0 = self.residual_function_0(x)
        out, aux_loss_1 = self.residual_function_1(out)
        shortcut = self.shortcut(x)
        out = nn.ReLU(inplace=True)(out + shortcut)
        return out, [aux_loss_0, aux_loss_1]

class ResNet_Sparse(nn.Module):

    def __init__(self, block, num_block, filter_set_mult, k_div, usecase, num_classes=100):
        super().__init__()
        self.usecase = usecase

        self.in_channels = 64

        self.conv1 = SparseCodingLayer_AfterSparse(in_dim=3, out_dim=64, stride=1, padding=1, filterset_size=round(int(64*filter_set_mult)), k=round(int(64/k_div)))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(filter_set_mult, k_div, block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(filter_set_mult, k_div, block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(filter_set_mult, k_div, block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(filter_set_mult, k_div, block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, filter_set_mult, k_div, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the 
        same as a neuron netowork layer, ex. conv layer), one layer may 
        contain more than one residual block 
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block 
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, filter_set_mult=filter_set_mult, k_div=k_div, stride=stride))
            self.in_channels = out_channels * block.expansion
        
        return SequentialCustomRes(*layers)

    def forward(self, x):
        output, aux_loss_1 = self.conv1(x)
        output, aux_losses_2 = self.conv2_x(output)
        output, aux_losses_3 = self.conv3_x(output)
        output, aux_losses_4 = self.conv4_x(output)
        output, aux_losses_5 = self.conv5_x(output)
        
        if self.usecase == "pretrain" or self.usecase == "random":
            output = output.detach().clone()
            output.requires_grad = True

        if self.usecase == "random" or self.usecase == "supervise":
            aux_loss = None

        else:
            aux_losses = [aux_loss_1] + aux_losses_2 + aux_losses_3 + aux_losses_4 + aux_losses_5
            aux_loss = torch.mean(torch.stack(aux_losses))

        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        logits = self.fc(output)

        _, preds = logits.max(dim=1)

        return logits, preds, aux_loss

class SequentialCustomRes(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        for module in modules:
            if not isinstance(module, nn.Module):
                raise Exception("Expected nn.Module, got: {}".format(type(module)))
        
        self.modules = modules
    
    def forward(self, x):
        aux_losses = []
        for module in self.modules:
            x, auxs = module(x)
            aux_losses += auxs
        return x, aux_losses

def resnet34_sparse(filter_set_mult, k_div, usecase):
    """ return a Sparse_ResNet 34 object
    """
    return ResNet_Sparse(BasicBlock_Sparse, [3, 4, 6, 3], filter_set_mult=filter_set_mult, k_div=k_div, usecase=usecase)
