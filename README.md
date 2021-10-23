# Efficient_YOLOv4

实现功能
-------

* 实现YOLOv4优化后的Efficient_YOLOv4的目标检测网络
* 进行网络结构的搭建
* 对自己数据集锚框的聚类
* 对网络模型的训练和预训练
* 将PANet替换为BiFPN
* 使用Swish激活函数代替Mish激活函数
* 去除了mosaic、 CmBN、 DropBlock、label_smoothing、CutMix、SAT等优化技巧，减少计算量
* 将学习率的余弦退火衰减变为0.97的指数衰减
* 将卷积块都替换成深度可分离卷积的上半部分，简称深度卷积，大幅减少计算量
* 将密集的卷积层替换为深度卷积，并加入CSP化，在参数量大幅减少的情况下保留相应的特征
* 模型体积从245MB下降到86MB
* 相同环境下模型的速度从45ms下降到35ms，达到实时速度
* MAP从41%提升到MAP0.5为59.57%

测试运行的环境
------------

* Windows10
* Python3.7
* Pytorch1.3
* CUDA10.0

Install
-------
##### Clone and install

1.git clone [https://github.com/Devils-hub/Efficient_YOLOv4.git](https://github.com/Devils-hub/Efficient_YOLOv4.git)

2.Codes are based on Python3

Training
---------

1.用labelimg标注自己的数据集为VOC格式，也可使用现有的VOC2007+VOC2012格式的数据集

2.用kmeans_for_anchors.py进行聚类获取数据的锚框

3.在使用voc2yolo3.py将数据集分为训练集和测试集

4.用voc_annotation.py得到数据的位置和信息

5.接下来就可以用trains.py进行训练了，不过要将代码中的路径改为自己的路径，并且将我的绝对路径改为你自己的绝对路径，把上面得到的锚框和得到的标签信息加入进去，就可以进行训练了
```Python
trains.py '-anchors', default="./model_data/anchors.txt" 
          '-classes', default="./model_data/class_name.txt" 
          '-annotation', default="./my_train.txt"
```

Detect
-----

```Python
detect.py '-model_path', default="./models/Epoch22-Total_Loss18.8616-Val_Loss18.8762.pth" 
          '-classes_path', default='./model_data/voc_classes.txt', type=str, help='classes path'
          '-anchors_path', default='./model_data/voc07_12anchors.txt'
```



