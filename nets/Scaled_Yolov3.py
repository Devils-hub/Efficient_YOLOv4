import torch
import torch.nn as nn


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BaseConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#   Darknet的结构块的组成部分
#   内部堆叠的残差块
class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()

        self.block = nn.Sequential(
            BaseConv(channels, channels, 1, 1),
            BaseConv(channels, channels, 3, 1)
        )

    def forward(self, x):
        return x + self.block(x)


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.conv1 = BaseConv(3, 32, 3, 1)  # 一个卷积块 = 1层卷积
        self.conv2 = BaseConv(32, 64, 3, 2)

        self.conv3_4 = Resblock(64)  # 一个残差块 = 2层卷积
        self.conv5 = BaseConv(64, 128, 3, 2)
        self.conv6_9 = nn.Sequential(*[Resblock(128) for _ in range(2)])  # = 4层卷积
        self.conv10 = BaseConv(128, 256, 3, 2)
        self.conv11_26 = nn.Sequential(*[Resblock(256) for _ in range(8)])  # = 16层卷积
        self.conv27 = BaseConv(256, 512, 3, 2)
        self.conv28_43 = nn.Sequential(*[Resblock(512) for _ in range(8)])  # = 16层卷积
        self.conv44 = BaseConv(512, 1024, 3, 2)
        self.conv45_52 = nn.Sequential(*[Resblock(1024) for _ in range(4)])  # = 8层卷积

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3_4 = self.conv3_4(conv2)
        conv5 = self.conv5(conv3_4)
        conv6_9 = self.conv6_9(conv5)
        conv10 = self.conv10(conv6_9)
        conv11_26 = self.conv11_26(conv10)  # 52 x 52 x 256的特征层
        conv27 = self.conv27(conv11_26)
        conv28_43 = self.conv28_43(conv27)  # 26 x 26 x 512的特征层
        conv44 = self.conv44(conv28_43)
        conv45_52 = self.conv45_52(conv44)  # 13 x 13 x 1024的特征层
        return conv11_26, conv28_43, conv45_52  # YOLOv3用，所以输出了3次特征


def darknet53(pretrained, **kwargs):
    model = DarkNet()
    if pretrained:
        model.load_state_dict(torch.load(pretrained))  # 加载预训练模型
    return model


class conv2d(nn.Module):  # 一个卷积模块
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#   卷积 + 上采样
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):  # （输入，输入/2）
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),  # conv2d(输入，输入/2，卷积核，步长)
            nn.Upsample(scale_factor=2, mode='nearest')  # Upsample(放大的倍数，上采样方法'最近邻') 最近邻是默认的
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


#   五次卷积块
def make_five_conv(filters_list, in_filters):  # （[输出，输出*2]，输入）
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),  # conv2d(输入，输入/2，卷积核，步长)
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


#   最后获得yolov4的输出   一个3x3的卷积 + 一个1x1的卷积   （特征的整合 -> yolov4的检测结果）
def yolo_head(filters_list, out_filter):  # （[2*输入，通道数]，输入）
    m = nn.Sequential(
        conv2d(filters_list[0], filters_list[1], 3),  # 卷积 + bn + Leak_Relu
        nn.Conv2d(filters_list[1], out_filter, 1, 1, 0, bias=True),  # 单纯的卷积层 nn.Conv2d(输入，输入/2，卷积核，步长)
    )
    return m


class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        out_filters = [64, 128, 256, 512, 1024]
        #  last_layer0
        final_out_filter = num_anchors * (5 + num_classes)
        self.last_layer0 = make_five_conv([512, 1024], out_filters[-1])
        self.yolo_head3 = yolo_head([512, 1024], final_out_filter)

        #  embedding1
        self.conv_upsample1 = Upsample(512, 256)
        self.last_layer1 = make_five_conv([256, 512], out_filters[-2] + 256)
        self.yolo_head2 = yolo_head([256, 512], final_out_filter)

        #  embedding2
        self.conv_upsample2 = Upsample(256, 128)
        self.last_layer2 = make_five_conv([128, 256], out_filters[-3] + 128)
        self.yolo_head1 = yolo_head([128, 256], final_out_filter)

        self.conv_upsample3 = Upsample(1024, 256)
        self.conv_upsample4 = Upsample(256, 1024)

    def forward(self, x):
        x2, x1, x0 = self.backbone(x)  # x0是1024的输出，x1是512的输出，x2是256的输出

        out0 = self.last_layer0(x0)
        yolo_head3 = self.yolo_head3(out0)

        conv_upsample1 = self.conv_upsample1(out0)
        up3_2 = torch.cat([conv_upsample1, x1], 1)
        conv5_in = self.last_layer1(up3_2)
        yolo_head2 = self.yolo_head2(conv5_in)

        conv_upsample2 = self.conv_upsample2(conv5_in)
        up2 = torch.cat([conv_upsample2, x2], 1)
        conv5_in1 = self.last_layer2(up2)
        yolo_head1 = self.yolo_head1(conv5_in1)

        return yolo_head3, yolo_head2, yolo_head1
