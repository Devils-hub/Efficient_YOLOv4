import torch
import torch.nn as nn
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.nets.CSPdarknet import darknet53, Mish


class conv2d(nn.Module):  # 一个卷积模块
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        # self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


#   SPP结构，利用不同大小的池化核进行池化
#   池化后堆叠
class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()  # 3种卷积核的最大池化
        self.maxpools = nn.ModuleList([nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])
        # MaxPool2d（卷积核，步长，填充）

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]  # 从0开始不要最后一个数值
        features = torch.cat(features + [x], dim=1)  # 3次特征和直接连接线的融合叠加

        return features


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


#   三次卷积块
def make_three_conv(filters_list, in_filters):  # tree_conv([输出， 2*输出]， 输入)
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),  # conv2d(输入，输入/2，卷积核，步长)
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return m


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
def yolo_head(filters_list, in_filters):  # （[2*输入，通道数]，输入）
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 3),  # 卷积 + bn + Leak_Relu   nn.Conv(in_channels, out_channels, kernel_size)
        nn.Conv2d(filters_list[0], filters_list[1], 1),  # 单纯的卷积层 nn.Conv2d(输入，输入/2，卷积核，步长)
    )
    return m


#   yolo_body
class YoloBody(nn.Module):
    def __init__(self, num_anchors, num_classes):
        super(YoloBody, self).__init__()
        #  backbone
        self.backbone = darknet53(None)

        # make_tree_conv([输出，2*输出]， 输入)
        self.conv1 = make_three_conv([512, 1024], 1024)  # 1024的输出 + 3次卷积  1024输入512输出再进行3次卷积  （[输出，输出*2]，输入）
        self.SPP = SpatialPyramidPooling()  # SPP结构，3次最大池化和直接连接线的融合叠加
        self.conv2 = make_three_conv([512, 1024], 2048)  # 3次卷积 （[输出，2*输出]，输入）

        self.upsample1 = Upsample(512, 256)  # 卷积+上采样  Upsample(输入，输入/2)  上面的输出就是这里的输入
        self.conv_for_P4 = conv2d(512, 256, 1)  # （输入，输入/2，卷积核）
        self.make_five_conv1 = make_five_conv([256, 512], 512)  # （[输出，输出*2]，输入）  上面的上采样输出就是这里的输入

        self.upsample2 = Upsample(256, 128)  # 卷积+上采样  Upsample(输入，输入/2)  上面的输出就是这里的输入
        self.conv_for_P3 = conv2d(256, 128, 1)  # （输入，输入/2，卷积核）
        self.make_five_conv2 = make_five_conv([128, 256], 256)  # （[输出，输出*2]，输入）  上面的上采样输出就是这里的输入

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 4+1+num_classes
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter2 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head3 = yolo_head([256, final_out_filter2], 128)  # 第一个yolo_head  （[2*输入，通道数]，输入）

        self.down_sample1 = conv2d(128, 256, 3, stride=2)  # （输入，输入/2，卷积核，步长）
        self.make_five_conv3 = make_five_conv([256, 512], 512)  # （[输出，输出*2]，输入）
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter1 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head2 = yolo_head([512, final_out_filter1], 256)  # 第二个yolo_head  （[2*输入，通道数]，输入）

        self.down_sample2 = conv2d(256, 512, 3, stride=2)  # （输入，输入/2，卷积核，步长）
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)  # （[输出，输出*2]，输入）
        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        # 每个网格(特征)点的3个预选框*（5个预测信息（4个参数调整先验框获得预测框1个是判断是否包含物体）+ num_classes的值是预测框中属于(每一个)类的概率）
        final_out_filter0 = num_anchors * (5 + num_classes)  # num_classes是要检测的类的个数（我的数据集类别数是6）
        self.yolo_head1 = yolo_head([1024, final_out_filter0], 512)  # 第三个yolo_head  （[2*输入，通道数]，输入）

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x)  # x0是1024的输出，x1是512的输出，x2是256的输出

        P5 = self.conv1(x0)  # 对1024进行3次卷积
        P5 = self.SPP(P5)  # 对1024进行3次卷积后进入SPP模块
        P5 = self.conv2(P5)  # 对SPP模块进行3次卷积  SPP模块结束

        P5_upsample = self.upsample1(P5)  # 对SPP模块进行一次卷积+上采样
        P4 = self.conv_for_P4(x1)  # 对512的输出进行一次卷积（卷积+bn+leak_relu）
        P4 = torch.cat([P4, P5_upsample], axis=1)  # 将512卷积后的结果与SPP上采样的结果进行融合
        P4 = self.make_five_conv1(P4)  # 再将融合后的结果进行5次卷积

        P4_upsample = self.upsample2(P4)  # 将5次卷积的结果进行上采样
        P3 = self.conv_for_P3(x2)  # 对256的输出进行卷积操作
        P3 = torch.cat([P3, P4_upsample], axis=1)  # 将256的卷积和5次卷积的结果进行融合
        P3 = self.make_five_conv2(P3)  # 将融合后的结果进行5次卷积  接下来就直接进入第一个yolo_head

        P3_downsample = self.down_sample1(P3)  # 上面融合后的结果进行5次卷积后，不进入第一个yolo_head先进行下采样
        P4 = torch.cat([P3_downsample, P4], axis=1)  # 将下采样的结果和512卷积融合SPP进行5次卷积后的结果进行融合
        P4 = self.make_five_conv3(P4)  # 融合后的结果进行5次卷积  接下来就直接进入到第二个yolo_head

        P4_downsample = self.down_sample2(P4)  # 上面融合后的结果进行5次卷积后，不进入第二个yolo_head先进行下采样
        P5 = torch.cat([P4_downsample, P5], axis=1)  # 将下采样的结果和SPP模块（P5）进行融合
        P5 = self.make_five_conv4(P5)  # 将融合后的结果进行5次卷积  接下来就直接进入到第三个yolo_head

        out2 = self.yolo_head3(P3)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到第一个yolo_head的结果
        out1 = self.yolo_head2(P4)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到第二个yolo_head的结果
        out0 = self.yolo_head1(P5)  # 经过3x3的卷积特征融合和一个1x1的普通卷积，就得到最下面的yolo_head的结果

        return out0, out1, out2  # 1024的预测结果，512的预测结果，256的预测结果

