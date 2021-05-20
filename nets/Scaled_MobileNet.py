import math
import torch
import torch.nn as nn
from collections import OrderedDict


class Conv(nn.Module):  # 卷积块
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky=0.1):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(negative_slope=leaky, inplace=True)
        # self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def conv_bn_relu(in_channels, out_channnels, stride, leaky=0.1):  # 标准卷积块
    x = nn.Sequential(
        nn.Conv2d(in_channels, out_channnels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channnels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    return x


def conv_dw(in_channels, out_channels, stride, leaky=0.1):  # DepthwiseConv2D
    x = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )
    return x


# def conv_dw_block(in_channels, out_channels, stride):
#     x = nn.Sequential(
#         conv_dw(in_channels, out_channels, stride),
#         conv_dw(out_channels, out_channels, stride // 2)
#     )
#     return x


class MobileNet(nn.Module):  # MobilenetV1-0.25
    def __init__(self):
        super(MobileNet, self).__init__()

        self.stage1 = nn.Sequential(  # mobilenetV1-0.25是mobilenetV1-1通道数压缩为原来1/4的网络
            conv_bn_relu(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            # conv_dw_block(16, 32, 2),
            # conv_dw_block(32, 64, 2),
        )

        self.stage2 = nn.Sequential(
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        # self.stage2 = nn.Sequential(
        #     conv_dw_block(64, 128, 2),
        #     *[conv_dw(128, 128, 1) for _ in range(4)])

        self.stage3 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            # conv_dw_block(128, 256, 2),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        return x1, x2, x3


# class MobileNet(nn.Module):  # MobilenetV1-0.25
#     def __init__(self):
#         super(MobileNet, self).__init__()
#
#         self.stage1 = nn.Sequential(  # mobilenetV1-0.25是mobilenetV1-1通道数压缩为原来1/4的网络
#             conv_bn_relu(3, 8, 2),
#             conv_dw(8, 16, 1),
#             conv_dw_block(16, 32, 2),
#             conv_dw_block(32, 64, 2))
#         self.stage2 = nn.Sequential(
#             conv_dw_block(64, 128, 2),
#             *[conv_dw(128, 128, 1) for _ in range(4)])
#         self.stage3 = conv_dw_block(128, 256, 2)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(256, 1000)
#
#     def forward(self, x):
#         x1 = self.stage1(x)
#         x2 = self.stage2(x1)
#         x3 = self.stage3(x2)
#         # out = OrderedDict()  # OrderedDict是记住键首次插入顺序的字典，如果新条目覆盖现有条目，则原始插入位置保持不变
#         #
#         # out['stage1'] = x1
#         # out['stage2'] = x2
#         # out['stage3'] = x3
#         # return out
#         x = self.avg_pool(x3)
#         x = x.view(-1, 256)
#         x = self.fc(x)
#         return x


def mobilenet(pretrained, **kwargs):
    model = MobileNet()
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


#  SPP结构，利用不同大小的池化核进行池化，池化后堆叠
class SPP(nn.Module):
    def __init__(self, pool_sizes=[1, 5, 9, 13]):
        super(SPP, self).__init__()  # 3种卷积核的最大池化  MaxPool2d（卷积核，步长，填充）
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        return torch.cat([x] + [features(x) for features in self.maxpools], 1)  # 3次特征和直接连接线的融合叠加


#   卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):  # （输入，输入/2）
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),  # conv2d(输入，输入/2，卷积核，步长)
            nn.Upsample(scale_factor=2, mode='nearest')  # Upsample(放大的倍数，上采样方法'最近邻') 最近邻是默认的
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


#   五次卷积块
class five_csp_conv(nn.Module):
    def __init__(self, filters_list, in_filters):
        super(five_csp_conv, self).__init__()
        self.conv = Conv(in_filters, filters_list[0], 1)
        self.conv4 = nn.Sequential(
            Conv(filters_list[0], filters_list[1], 3),  # conv2d(输入，输入/2，卷积核，步长)
            Conv(filters_list[1], filters_list[0], 1),
            Conv(filters_list[0], filters_list[1], 3),
            Conv(filters_list[1], filters_list[0], 1),
        )

    def forward(self, x):
        conv = self.conv(x)
        conv5 = self.conv4(conv)
        convs = torch.cat([conv, conv5], dim=1)
        return self.conv(convs)


# 最后获得yolov4的输出   一个3x3的卷积 + 一个1x1的卷积   （特征的整合 -> yolov4的检测结果）
def yolo_head(filters_list, in_filters):  # （[2*输入，通道数]，输入）
    m = nn.Sequential(
        Conv(in_filters, filters_list[0], 3),  # 卷积 + bn + Leak_Relu   nn.Conv(in_channels, out_channels, kernel_size)
        nn.Conv2d(filters_list[0], filters_list[1], 1),  # 单纯的卷积层 nn.Conv2d(输入，输入/2，卷积核，步长)
    )
    return m


#  yolo_body
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        self.backbone = mobilenet(None)  # backbone

        self.conv1 = Conv(1024, 512, 1)
        self.conv11 = Conv(512, 1024, 3)
        self.conv12 = Conv(1024, 512, 1)
        self.SPP = SPP()
        self.conv2 = Conv(2560, 512, 1)
        self.conv21 = Conv(512, 512, 3)

        self.conv22 = Conv(1024, 512, 1)

        self.upsample1 = Upsample(512, 256)  # 卷积+上采样  Upsample(输入，输入/2)  上面的输出就是这里的输入
        self.conv_for_P4 = Conv(512, 256, 1)  # （输入，输入/2，卷积核）
        self.make_five_conv1 = five_csp_conv([256, 512], 512)  # （[输出，输出*2]，输入）  上面的上采样输出就是这里的输入

        self.upsample2 = Upsample(256, 128)  # 卷积+上采样  Upsample(输入，输入/2)  上面的输出就是这里的输入
        self.conv_for_P3 = Conv(256, 128, 1)  # （输入，输入/2，卷积核）
        self.make_five_conv2 = five_csp_conv([128, 256], 256)  # （[输出，输出*2]，输入）  上面的上采样输出就是这里的输入

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter2 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)  # 第一个yolo_head  （[2*输入，通道数]，输入）

        self.down_sample1 = Conv(128, 256, 3, stride=2)  # （输入，输入/2，卷积核，步长）
        self.make_five_conv3 = five_csp_conv([256, 512], 512)  # （[输出，输出*2]，输入）
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter1 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head2 = yolo_head([512, final_out_filter1], 256)  # 第二个yolo_head  （[2*输入，通道数]，输入）

        self.down_sample2 = Conv(256, 512, 3, stride=2)  # （输入，输入/2，卷积核，步长）
        self.make_five_conv4 = five_csp_conv([512, 1024], 1024)  # （[输出，输出*2]，输入）
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter0 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head1 = yolo_head([1024, final_out_filter0], 512)  # 第三个yolo_head  （[2*输入，通道数]，输入）

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)  # x0是1024的输出，x1是512的输出，x2是256的输出

        conv1 = self.conv1(x0)
        conv11 = self.conv11(conv1)
        conv12 = self.conv12(conv11)
        SPP = self.SPP(conv12)
        conv2 = self.conv2(SPP)
        conv21 = self.conv21(conv2)
        conv21 = torch.cat([conv1, conv21], axis=1)
        P5 = self.conv22(conv21)

        P5_upsample = self.upsample1(P5)  # 对SPP模块进行一次卷积+上采样
        P4 = self.conv_for_P4(x1)  # 对512的输出进行一次卷积（卷积+bn+leak_relu）
        P4 = torch.cat([P4, P5_upsample], axis=1)  # 将512卷积后的结果与SPP上采样的结果进行融合
        P4 = self.make_five_conv1(P4)  # 再将融合后的结果进行5次卷积

        P4_upsample = self.upsample2(P4)  # 将5次卷积的结果进行上采样
        P3 = self.conv_for_P3(x2)  # 对256的输出进行卷积操作
        P3 = torch.cat([P3, P4_upsample], axis=1)  # 将256的卷积和5次卷积的结果进行融合
        # P3 = P3 + x0
        P3 = self.make_five_conv2(P3)  # 将融合后的结果进行5次卷积  接下来就直接进入第一个yolo_head
        # P3 = self.ssh1(P3)

        P3_downsample = self.down_sample1(P3)  # 上面融合后的结果进行5次卷积后，不进入第一个yolo_head先进行下采样
        P4 = torch.cat([P3_downsample, P4], axis=1)  # 将下采样的结果和512卷积融合SPP进行5次卷积后的结果进行融合
        # P4 = P4 + x1
        P4 = self.make_five_conv3(P4)  # 融合后的结果进行5次卷积  接下来就直接进入到第二个yolo_head

        P4_downsample = self.down_sample2(P4)  # 上面融合后的结果进行5次卷积后，不进入第二个yolo_head先进行下采样
        P5 = torch.cat([P4_downsample, P5], axis=1)  # 将下采样的结果和SPP模块（P5）进行融合
        # P5 = P5 + x2
        P5 = self.make_five_conv4(P5)  # 将融合后的结果进行5次卷积  接下来就直接进入到第三个yolo_head

        out2 = self.yolo_head3(P3)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到第一个yolo_head的结果
        out1 = self.yolo_head2(P4)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到第二个yolo_head的结果
        out0 = self.yolo_head1(P5)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到最下面的yolo_head的结果

        return out0, out1, out2  # 1024的预测结果，512的预测结果，256的预测结果
