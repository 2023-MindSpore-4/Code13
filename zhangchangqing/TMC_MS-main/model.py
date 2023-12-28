import mindspore.nn as nn
import mindspore as ms
import mindspore
import mindspore.ops as ops
import mindspore.ops.operations as P
import numpy as np
import math
import mindspore.common.dtype as mstype

import mindspore.context as context
from mindspore import Parameter, Tensor
from mindspore.common.initializer import Normal
from mindspore.ops import constexpr

import mindspore.context as context

# context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

#装饰器，用于静态图常量tensor,求batchsize大小
@constexpr
def construct_tensor(x):
    if x is None:
        raise ValueError("input is an unknown value")
    return x

class NormedLinear(nn.Cell):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(Tensor(np.random.uniform(-1, 1, size=(in_features, out_features)), mindspore.float32), requires_grad=True)
        self.weight.data.renorm(2, 1, 1e-5)
        self.normalize_dim0 = ops.L2Normalize(axis=0)
        self.normalize_dim1 = ops.L2Normalize(axis=1)
    def construct(self, x):
        # x: [batch_size:128, in_features:64]
        # self.weight: [in_features:64, out_features:100]
        x_normalized = self.normalize_dim1(x)
        weight_normalized = self.normalize_dim0(self.weight)
        return ops.matmul(x_normalized, weight_normalized)


# class NormedLinear(nn.Cell):
#     def __init__(self, in_features, out_features):
#         super(NormedLinear,self).__init__()
#         self.linear = nn.Dense(in_features,out_features)
#     def construct(self, x):
#         x = self.linear(x)
#         return x

class BasicBlock(nn.Cell):
    def __init__(self,in_planes,planes,stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,pad_mode='pad',padding=1,has_bias=False,weight_init=weight_init)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,pad_mode='pad',padding=1,has_bias=False,weight_init=weight_init)
        self.bn2 = nn.BatchNorm2d(planes)
        #如果特征图尺寸相同，直接输出
        self.shortcut = lambda x:x
        #如果特征图尺寸不同，使用下采样，或使用1*1卷积核，使其通道数加倍
        if stride != 1 or in_planes!=planes:
            if option == 'A':
                self.planes = planes
                self.in_planes = in_planes
                # 下采样，并对通道数填充0，使其加倍
                self.shortcut = lambda x: nn.Pad(
                    paddings=((0, 0), ((planes - in_planes) // 2, (planes - in_planes) // 2), (0, 0), (0, 0)),
                    mode="CONSTANT")(x[:, :, ::2, ::2])
            elif option == 'B':
                self.planes = planes
                self.in_planes = in_planes
                self.shortcut =nn.SequentialCell(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=False),
                     nn.BatchNorm2d(planes,gamma_init=gamma_init)
                )


    def construct(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))+self.shortcut(x)
        out = nn.ReLU()(out)
        return out




class ResNet_s(nn.Cell):

    def __init__(self,block,num_blocks,num_experts,num_classes,reweight_temperature=0.2):
        super(ResNet_s, self).__init__()

        self.in_planes = 16
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.eta = reweight_temperature


        self.conv1 = nn.Conv2d(3,16,kernel_size=3,stride=1,pad_mode='pad',padding=1,has_bias=False,weight_init=weight_init)
        self.bn1 = nn.BatchNorm2d(16)

        layer1s_list = []
        for _ in range(num_experts):
            layer1s_list.append(self._make_layer(block,16,num_blocks[0],stride=1))
        self.layer1s = nn.CellList(layer1s_list)
        self.in_planes = self.next_in_planes

        layer2s_list = []
        for _ in range(num_experts):
            layer2s_list.append(self._make_layer(block,32,num_blocks[1],stride=2))
        self.layer2s = nn.CellList(layer2s_list)
        self.in_planes = self.next_in_planes

        layer3s_list = []
        for _ in range(num_experts):
            layer3s_list.append(self._make_layer(block,64,num_blocks[2],stride=2))
        self.layer3s = nn.CellList(layer3s_list)
        self.in_planes = self.next_in_planes

        # layer4s_list = []
        # for _ in range(num_experts):
        #     layer4s_list.append(self._make_layer(block,512,num_blocks[3],stride=2))
        # self.layer4s = nn.CellList(layer4s_list)
        # self.in_planes = self.next_in_planes

        linears_list = []
        for _ in range(num_experts):
            linears_list.append(NormedLinear(64,num_classes))
        self.linears = nn.CellList(linears_list)

        self.use_experts = list(range(num_experts))
        self.AvgPool2d = nn.AvgPool2d(kernel_size=8)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        self.next_in_planes = self.in_planes
        for stride in strides:
            layers.append(block(self.next_in_planes, planes, stride))
            self.next_in_planes = planes
        return nn.SequentialCell(*layers)


    def construct(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))

        outs = []

        b0 = None
        #静态图中，len(x)为128,即batchsize,使用len(x)会报错
        batch_size = construct_tensor(x.shape[0])
        W = [ms.Tensor(np.ones(batch_size, dtype=np.float32))]

        for i in self.use_experts:
            xi = self.layer1s[i](x)
            xi = self.layer2s[i](xi)
            xi = self.layer3s[i](xi)
            # xi = self.layer4s[i](xi)
            xi = self.AvgPool2d(xi)

            xi = xi.view(xi.shape[0], -1, 1).squeeze(-1)
            xi = self.linears[i](xi)
            xi = xi * 30



            outs.append(xi)
        # return outs

            # evidential
            alpha = ops.Exp()(xi) + 1
            S = alpha.sum(axis=1, keepdims=True)
            b = (alpha - 1) / S
            u = self.num_classes / S.squeeze(-1)

            # update w
            if b0 is None:
                C = 0
            else:
                bb = ops.BatchMatMul()(b0.view(-1,b0.shape[1],1),b.view(-1,1,b.shape[1]))
                C = bb.sum(axis=(1, 2)) - bb.diagonal(offset=0, axis1=1, axis2=2).sum(axis=1)
            b0 = b
            W.append(W[-1] * u / (1 - C))

        # dynamic reweighting
        exp_w = [ops.exp(wi / self.eta) for wi in W]
        exp_w = [wi / wi.sum() for wi in exp_w]
        exp_w = [ops.ExpandDims()(wi,-1) for wi in exp_w]


        reweighted_outs = [outs[i] * exp_w[i] for i in self.use_experts]



        return sum(reweighted_outs), len(self.use_experts), outs, W

def resnet(num_blocks,num_experts,num_classes,reweight_temperature=0.2):
    return ResNet_s(BasicBlock,num_blocks=num_blocks,
            num_classes = num_classes  ,
            num_experts = num_experts,
            reweight_temperature = reweight_temperature
        )



