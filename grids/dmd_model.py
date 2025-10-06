"""
This file (model_zoo.py) is designed for:
    models
Copyright (c) 2025, Zhiyu Pan. All rights reserved.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F
import copy
import math

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

# create a new 3x3 
def part_conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, kernel_number=1):
    # partial convolution in 
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def batch_translate_multiweight(weights, lambdas, trans): # generate the fusion weight according to the translate ratio
    """
    Let
        batch_size = b
        kernel_number = n
        kernel_size = 3
    Args:
        weights: tensor, shape = [kernel_number, Cout, Cin, k, k]
        lambdas: tensor of lambdas, shape = [batch_size, kernel_number], controlling the fusing weight
        trans: tensor of trans,  shape = [batch_size, kernel_number, 2]
    Return:
        weights_out: tensor, shape = [batch_size x Cout, Cin // groups, k, k]
    """
    assert(trans.shape[:2] == lambdas.shape)
    assert(lambdas.shape[1] == weights.shape[0])

    b = trans.shape[0]
    n = trans.shape[1]
    k = weights.shape[-1]
    _, Cout, Cin, _, _ = weights.shape

    # create the grid
    trans = trans.reshape(-1,2) # [bs * n, 2]
    # create the Translate matrix with Bs * n, 2 ,3
    T = torch.stack(
        (
            torch.stack([torch.tensor([1]*(b*n)).to(trans.device), torch.tensor([0]*(b*n)).to(trans.device) ,trans[:,0]], dim=1),
            torch.stack([torch.tensor([0]*(b*n)).to(trans.device), torch.tensor([1]*(b*n)).to(trans.device), trans[:,1]], dim=1),
        ),
        dim=1,
    )
    # create the grid
    grid = F.affine_grid(T, [b*n, Cout*Cin, k, k], align_corners=False)
    # append the weight with the b at the first dimension
    weights = weights.unsqueeze(0).repeat(b, 1, 1, 1, 1, 1).reshape(b*n, Cout * Cin, k, k)
    # grid the weight
    weights = F.grid_sample(weights, grid, mode='bilinear', align_corners=False) # translate the weight [b*n, Cout, Cin, k, k]
    weights = weights.reshape(b, n, Cout, Cin, k, k) # [b, n, Cout, Cin, k, k]

    # fusing the weight acoording to the lambdas
    weights = torch.mul(weights, lambdas.unsqueeze(2).unsqueeze(3).unsqueeze(4).unsqueeze(5)) # [b, n, Cout, Cin, k, k]
    # sum the weight
    weights = torch.sum(weights, dim=1) # [b, Cout, Cin, k, k]
    # reshape the weight
    weights = weights.reshape(b * Cout, Cin, k, k) # [b * Cout, Cin, k, k]

    return weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module): # with adaptive translate convolution
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_layers=[64, 128, 256, 512],
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        num_in=3,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = num_layers[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_layers[0], layers[0])
        self.layer2 = self._make_layer(block, num_layers[1], layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, num_layers[2], layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, num_layers[3], layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_layers[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feature0 = x.clone() # output the first feature map
        x = self.layer1(x)
        x = x / 2 ** 3
        feature1 = x.clone() # output the second feature map
        x = self.layer2(x)
        x = x / 2 ** 8
        feature2 = x.clone() # output the third feature map
        x = self.layer3(x)
        x = x / 2 ** 36
        feature3 = x.clone() # output the fourth feature map
        x = self.layer4(x)
        x = x / 2 ** 3
        feature_map = x.clone() # output the final feature map
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        output = {
            'feat_f': feature_map,
            'feat_0': feature0,
            'feat_1': feature1,
            'feat_2': feature2,
            'feat_3': feature3,
            'x': x,
        }
        return output # x, feature_map # output the feature map

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if "conv1" not in k}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicDeConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0, outpadding=0):
        super(BasicDeConv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=outpadding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class NormalizeModule(nn.Module):
    def __init__(self, m0=0.0, var0=1.0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y
    
class DoubleConv(nn.Module):
    def __init__(self, in_chn, out_chn, do_bn=True, do_res=False):
        super().__init__()
        self.conv = (
            nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_chn),
                nn.LeakyReLU(inplace=True),
            )
            if do_bn
            else nn.Sequential(
                nn.Conv2d(in_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(out_chn, out_chn, 3, padding=1),
                nn.LeakyReLU(inplace=True),
            )
        )

        self.do_res = do_res
        if self.do_res:
            if out_chn < in_chn:
                self.original = nn.Conv2d(in_chn, out_chn, 1, padding=0)
            elif out_chn == in_chn:
                self.original = nn.Identity()
            else:
                self.original = ChannelPad(out_chn - in_chn)

    def forward(self, x):
        out = self.conv(x)
        if self.do_res:
            res = self.original(x)
            out = out + res
        return out

class ChannelPad(nn.Module):
    def __init__(self, after_C, before_C=0, value=0) -> None:
        super().__init__()
        self.before_C = before_C
        self.after_C = after_C
        self.value = value

    def forward(self, x):
        prev_0 = [0] * (x.ndim - 2) * 2
        out = F.pad(x, (*prev_0, self.before_C, self.after_C), value=self.value)
        return out
    
class PositionEncoding2D(nn.Module):
    def __init__(self, in_size, ndim): #, translation=False
        super().__init__()
        # self.translation = translation
        n_encode = ndim // 2
        self.in_size = in_size
        coordinate = torch.meshgrid(torch.arange(in_size[0]), torch.arange(in_size[1]), indexing="ij")
        div_term = torch.exp(torch.arange(0, n_encode, 2).float() * (-math.log(10000.0) / n_encode)).view(-1, 1, 1) # d_model/4个
        pe = torch.cat(
            (
                torch.sin(coordinate[0].unsqueeze(0) * div_term),
                torch.cos(coordinate[0].unsqueeze(0) * div_term),
                torch.sin(coordinate[1].unsqueeze(0) * div_term),
                torch.cos(coordinate[1].unsqueeze(0) * div_term),
            ),
            dim=0,
        )
        self.div_term = div_term # B, d_model, 1, 1
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe

class NOP(nn.Module):
    def forward(self, x):
        return x  

class DMD(nn.Module):
    def __init__(self, num_in=1, ndim_feat=6, pos_embed=True, input_norm=False, tar_shape = (256, 256)):
        super().__init__()
        self.num_in = num_in  # number of input channel
        self.ndim_feat = ndim_feat  # number of latent dimension
        self.input_norm = input_norm
        self.tar_shape = tar_shape
        layers = [3, 4, 6, 3]
        self.base_width = 64
        num_layers = [64, 128, 256, 512]
        block = BasicBlock

        self.inplanes = num_layers[0]
        self.img_norm = NormalizeModule(m0=0, var0=1)
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(inplace=True),
        )
        
        self.layer1 = self._make_layers(block, num_layers[0], layers[0])
        self.layer2 = self._make_layers(block, num_layers[1], layers[1], stride=2)
        self.layer3 = self._make_layers(block, num_layers[2], layers[2], stride=2)
        self.layer4 = self._make_layers(block, num_layers[3], layers[3], stride=2)

        self.texture3 = copy.deepcopy(self.layer3)
        self.texture4 = copy.deepcopy(self.layer4)

        self.minu_map = nn.Sequential(
            DoubleConv(num_layers[2] * block.expansion, 128),
            DoubleConv(128, 128),
            DoubleConv(128, 128),
            BasicDeConv2d(128, 128, kernel_size=4, stride=2, padding=1),
            BasicConv2d(128, 128, kernel_size=3, stride=1, padding=1),
            BasicDeConv2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )  # size=(128, 128)

        self.embedding = nn.Sequential(
            PositionEncoding2D((self.tar_shape[0]//16, self.tar_shape[1]//16), num_layers[3] * block.expansion) if pos_embed else NOP(),
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], ndim_feat, kernel_size=1),
        )
        self.embedding_t = nn.Sequential(
            PositionEncoding2D((self.tar_shape[0]//16, self.tar_shape[1]//16), num_layers[3] * block.expansion) if pos_embed else NOP(),
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], ndim_feat, kernel_size=1),
        )


        self.foreground = nn.Sequential(
            nn.Conv2d(num_layers[3] * block.expansion, num_layers[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(num_layers[3]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_layers[3], 1, kernel_size=1),
            nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=stride, stride=stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                base_width=self.base_width,
                downsample=downsample,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_embedding(self, x):
        if self.input_norm:
            x = self.img_norm(x) # adding the normalization layer (no traineable parameters)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        t_x3 = self.texture3(x2)
        t_x4 = self.texture4(t_x3)
        feature_t = self.embedding_t(t_x4)
        feature_m = self.embedding(x4)
        foreground = self.foreground(t_x4)
        feature = torch.cat((feature_t, feature_m), dim=1)

        return {
            "feature": feature.flatten(1),
            "feature_t": feature_t.flatten(1),
            "feature_m": feature_m.flatten(1),
            "mask": foreground.flatten(1),
        }

    def forward(self, x):
        if self.input_norm:
            x = self.img_norm(x) # adding the normalization layer (no traineable parameters)
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        t_x3 = self.texture3(x2)
        t_x4 = self.texture4(t_x3)
        feature_t = self.embedding_t(t_x4)

        minu_map = self.minu_map(x3)
        feature = self.embedding(x4)
        foreground = self.foreground(t_x4) 

        return {
            "input": x,
            "feat_f": feature,
            "feat_t": feature_t,
            "mask_f": foreground,
            "minu_map": minu_map,
            "minu_lst": torch.split(minu_map.detach(), 3, dim=1),
            "feat_lst": torch.split(feature.detach(), 3, dim=1),
        }