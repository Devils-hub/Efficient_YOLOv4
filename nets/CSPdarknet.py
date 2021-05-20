import torch
import torch.nn.functional as F
import torch.nn as nn
import math


#   MISH激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


#   卷积块
#   CONV+BATCHNORM+MISH
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#   CSPdarknet的结构块的组成部分
#   内部堆叠的残差块
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None, residual_activation=nn.Identity()):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


#   CSPdarknet的结构块
#   存在一个大残差边
#   这个大残差边绕过了很多的残差结构
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(Resblock_body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)  # 首先进行一次下采样

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)  # 残差卷积的残差边
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)  # 残差卷积
            self.blocks_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),  # 残差块的次数
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)  # 残差卷积
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)  # 残差卷积的残差边
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)  # 残差卷积

            self.blocks_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],  # 残差块的次数
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)  # 残差卷积

    def forward(self, x):
        x = self.downsample_conv(x)  # 首先进行下采样，下采样之后卷积核变为原来的一半，通道数变为原来的二倍

        x0 = self.split_conv0(x)  # 残差边

        x1 = self.split_conv1(x)  # 残差卷积
        x1 = self.blocks_conv(x1)  # 残差卷积连接残差块的次数

        x = torch.cat([x1, x0], dim=1)  # 残差边和残差块的融合
        x = self.concat_conv(x)  # 残差模块输出后进行卷积操作

        return x
# class Resblock_body(nn.Module):
#     def __init__(self, in_channels, out_channels, num_blocks, first):
#         super(Resblock_body, self).__init__()
#         self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)
#
#         if first:
#             self.res_side = BasicConv(out_channels, out_channels, 1)  # 残差卷积的残差边
#             self.res_conv = BasicConv(out_channels, out_channels, 1)  # 残差卷积
#             self.resblock_conv = nn.Sequential(
#                 Resblock(channels=out_channels, hidden_channels=out_channels//2),  # 残差块的次数
#                 BasicConv(out_channels, out_channels, 1)
#             )
#             self.concat_conv = BasicConv(out_channels*2, out_channels, 1)  # 残差卷积
#         else:
#             self.res_side = BasicConv(out_channels, out_channels//2, 1)  # 残差卷积的残差边
#             self.res_conv = BasicConv(out_channels, out_channels//2, 1)  # 残差卷积
#
#             self.resblock_conv = nn.Sequential(
#                 *[Resblock(out_channels//2) for _ in range(num_blocks)],  # 残差块的次数
#                 BasicConv(out_channels//2, out_channels//2, 1)
#             )
#             self.concat_conv = BasicConv(out_channels, out_channels, 1)  # 残差卷积
#
#     def forward(self, x):
#         x = self.downsample_conv(x)  # 首先进行下采样
#
#         x0 = self.res_side(x)  # 残差边
#
#         x1 = self.res_conv(x)  # 残差卷积
#         x1 = self.resblock_conv(x1)  # 残差卷积连接残差块的次数
#
#         x = torch.cat([x1, x0], dim=1)  # 残差边和残差块的融合
#         x = self.concat_conv(x)  # 残差卷积输出
#         return x


class CSPDarkNet(nn.Module):
    def __init__(self):
        super(CSPDarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = BasicConv(3, self.inplanes, kernel_size=3, stride=1)  # 输入通道数为3
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            Resblock_body(self.inplanes, self.feature_channels[0], 1, first=True),
            Resblock_body(self.feature_channels[0], self.feature_channels[1], 2, first=False),
            Resblock_body(self.feature_channels[1], self.feature_channels[2], 8, first=False),
            Resblock_body(self.feature_channels[2], self.feature_channels[3], 8, first=False),
            Resblock_body(self.feature_channels[3], self.feature_channels[4], 4, first=False)
        ])

        self.num_features = 1
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5


def darknet53(pretrained, **kwargs):
    model = CSPDarkNet()
    if pretrained:
        if isinstance(pretrained, str):  # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()，判断给定变量类型
            model.load_state_dict(torch.load(pretrained))  # isinstance() 会认为子类是一种父类类型，考虑继承关系
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
