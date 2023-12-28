import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import numpy as np

class Fusion(nn.Cell):
    def __init__(self, channel_size=64, upblock_channel=None, add=True):
        super(Fusion, self).__init__()
        self.channel_size = channel_size
        self.add = add

        # 模态1 模态2 融合
        self.concat = P.Concat(axis=1)
        self.conv1 = nn.Conv2d(channel_size*2, channel_size*4, kernel_size=1, stride=1, has_bias=False,
                          weight_init='normal', pad_mode='same')
        self.bn1 = nn.BatchNorm2d(channel_size*4, eps=1e-5, momentum=0.9)
        self.activation1 = nn.ReLU()

        # 上一阶段特征融合
        self.conv2 = nn.Conv2d(upblock_channel, channel_size*4, kernel_size=1, stride=2, has_bias=False,
                          weight_init='normal', pad_mode='same')
        self.bn2 = nn.BatchNorm2d(channel_size*4, eps=1e-5, momentum=0.9)
        self.relu = nn.ReLU()

    def construct(self, x, x_lwir, upblock):
        x_fusion = self.concat((x, x_lwir))
        x_fusion = self.conv1(x_fusion)
        x_fusion = self.bn1(x_fusion)
        x_fusion = self.activation1(x_fusion)
        if not self.add:
            return x_fusion

        upblock = self.conv2(upblock)
        upblock = self.bn2(upblock)

        x_fusion = x_fusion + upblock
        x_fusion = self.relu(x_fusion)
        return x_fusion

# 模态1 模态2 tensor
channel_size = 64
x = np.random.rand(1, channel_size, 32, 32)
x = mindspore.Tensor(x).astype(mindspore.float32)
x_lwir = np.random.rand(1, channel_size, 32, 32)
x_lwir = mindspore.Tensor(x_lwir).astype(mindspore.float32)

# 上一阶段的特征
upblock = np.random.rand(1, channel_size * 4, 64, 64)
upblock = mindspore.Tensor(upblock).astype(mindspore.float32)

# 实例化Fusion类
upblock_channel = upblock.shape[1]
fusion = Fusion(channel_size=channel_size, upblock_channel=upblock_channel)

# 融合
x_fusion = fusion(x, x_lwir, upblock)