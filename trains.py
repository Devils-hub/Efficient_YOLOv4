import time
import torch
import numpy as np
from torch.utils.data import DataLoader
# from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.nets.yolo4 import YoloBody
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.nets.Scaled_Darknet import YoloBody
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.nets.yolo_training import YOLOLoss
from Pytorch.photo_detection.YOLO_v4.yolov4_pytorch_train.dataloader import yolo_dataset_collate, YoloDataset
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from ranger_adabelief import RangerAdaBelief
device = torch.device("cuda" if torch.cuda.is_available else "cpu")


def args_parse():
    parser = argparse.ArgumentParser(description="训练的参数")
    parser.add_argument('-input_size', default=(416, 416), type=int, help='input image size', dest='input_size')
    # parser.add_argument('-anchors', default="./model_data/voc07_12anchors.txt", type=str, help='anchor file')
    parser.add_argument('-anchors', default="./model_data/yolo_anchors.txt", type=str, help='anchor file')
    parser.add_argument('-classes', default="./model_data/voc_classes.txt", type=str, help='classes name')
    parser.add_argument('-annotation', default="./train.txt", type=str, help='annotation file')
    parser.add_argument('-test', default="./test.txt", type=str, help="test file")
    # parser.add_argument('-model', default="./model_data/yolo4_voc_weights.pth", type=str, help='modess file')
    parser.add_argument('-model', default="./mod_097_yoloanchor/Epoch75-Total_Loss6.3006-Val_Loss14.1924.pth", type=str, help='model file')
    # parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-lr', default=0.000102, type=float, help='learning rate')
    parser.add_argument('-batch_size', default=4, type=int, help='train data batch size')
    parser.add_argument('-epochs', default=450, type=int, help='train epoch size')
    args = parser.parse_args()
    return args


def parse_lines(path):  # loads the lines
    with open(path) as file:
        lines_name = file.readlines()  # 用于读取所有行到结束符并返回列表，每行作为一个元素
    return lines_name


def get_classes(path):  # loads the classes
    class_names = parse_lines(path)
    class_names = [c.strip() for c in class_names]  # strip()方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    return class_names


def get_anchors(path):  # loads the anchors
    with open(path) as line:
        anchors = line.readline()  # 每次读取一行内容
    anchors = [float(x) for x in anchors.split(',')]  # split() 通过指定分隔符对字符串进行切片
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]  # [-1是行，根据其他的数据最后确定行的值]  [::-1, :, :]得到总数x
    # x个3行2列  3行代表3组anchors框，2列表示一组由2个数组成


def train():
    args = args_parse()
    anchors = get_anchors(args.anchors)  # get anchors
    num_classes = len(get_classes(args.classes))  # get classes
    # 绘制模型
    models = YoloBody(len(anchors[0]), num_classes)

    # 使用预训练模型，如果显卡不够大的话可以使用预训练模型来微调
    print("Load pretrained model into state dict...")
    model_list = models.state_dict()
    pretrained_dict = torch.load(args.model, map_location="cuda:0")  # Load pretrained model
    pretrained_dicts = {k: v for k, v in pretrained_dict.items() if k in model_list}

    # pretrained_dicts = {}
    # for k, v in pretrained_dict.items():
    #     # print(k)
    #     if np.shape(model_list[k]) == np.shape(v):  # # 用shape可以迅速的读取矩阵的形状
    #         # print(v)
    #         pretrained_dicts[k] = v

    model_list.update(pretrained_dicts)  # 把pretrained_dicts的键值对更新到model_list
    models.load_state_dict(model_list)
    print("Finished!")

    model = models.cuda()

    yolo_losses = []  # creat loss function
    input_shape = args.input_size
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), label_smooth=0, cuda=True))

    # val_ratio = 0.1  # 训练集验证集分配
    # lines = parse_lines(args.annotation)  # 按行读取所有的标签文件行，每一行作为一个元素
    # np.random.seed(5050)
    # np.random.shuffle(lines)
    # np.random.seed(0)
    # val_num = int(len(lines) * val_ratio)
    # train_num = len(lines) - val_num
    train_lines = parse_lines(args.annotation)
    test_lines = parse_lines(args.test)
    train_num = len(train_lines)
    test_num = len(test_lines)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=5e-4)  # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)  # 优化器
    optimizer = RangerAdaBelief(model.parameters(), lr=args.lr, eps=1e-12, betas=(0.9, 0.999))  # 优化器
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)

    batch_size = args.batch_size
    # train_dataset = YoloDataset(lines[:train_num+1], (input_shape[0], input_shape[1]), mosaic=False)  # Load dataset
    # val_dataset = YoloDataset(lines[train_num+1:], (input_shape[0], input_shape[1]), mosaic=False)
    train_dataset = YoloDataset(train_lines, (input_shape[0], input_shape[1]), mosaic=False)  # Load dataset
    val_dataset = YoloDataset(test_lines, (input_shape[0], input_shape[1]), mosaic=False)
    # 4个进程，pin_memory=True在data_loader返回之前，将tensor拷贝到固定内存(锁页内存)中，drop_last最后不够一个批次则丢掉
    #  collate_fn可合并样本列表以形成小批量的Tensor对象,如果图片损坏不能读取发生异常，则将空对象过滤掉
    train = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate)
    val = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate)

    train_epoch_size = train_num // batch_size  # 训练迭代次数 = 训练的总数据数 // 训练的批次数
    val_epoch_size = test_num // batch_size  # 验证迭代次数 = 验证的总数据数 // 验证的批次数
    # val_epoch_size = test_num // batch_size  # 验证迭代次数 = 验证的总数据数 // 验证的批次数

    for param in models.backbone.parameters():  # 冻结部分网络
        param.requires_grad = True  # 是否对backbone进行梯度更新

    writer = SummaryWriter(log_dir="./log", flush_secs=60)  # 进行训练可视化
    epochs = args.epochs
    for epoch in range(75, epochs):
        if epoch > epochs:
            break
        total_loss = 0
        val_loss = 0
        model.train()
        start = time.time()
        with tqdm(total=train_epoch_size, desc=f"Epoch {epoch + 1} / {epochs}") as pbar:  # 进行训练
            for i, (data, targets) in enumerate(train):  # i是索引，(data, targets)是train中的元素
                if i >= train_epoch_size:
                    break
                images = torch.from_numpy(data).cuda()  # 转换成张量的形式
                targets = [torch.from_numpy(ann) for ann in targets]  # 转换成张量数组的形式

                # pytorch的参数反向传播时，梯度是累积计算的不是替换，但是每个batch处理的时候并不需要其他梯度混合累积计算，因此需要梯度置零
                optimizer.zero_grad()  # 将模型的参数梯度初始化为0
                outputs = model(images)  # 前向传播计算预测值

                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)  # 计算当前损失
                    losses.append(loss_item[0])

                loss = sum(losses)  # 一次训练的损失和
                loss.backward()  # 反向传播计算梯度  梯度的反向传播
                optimizer.step()  # 经过一次梯度下降，对所有的参数进行更新

                total_loss += loss
                spend_time = time.time() - start  # 训练所用的时间

                def get_lr(optimizer):  # loads the learning rate
                    for param_group in optimizer.param_groups:  # param_groups里面保存了参数组及其对应的学习率,动量
                        return param_group['lr']

                dicts = {"step/s": spend_time, "lr": get_lr(optimizer), "total_loss": total_loss.item() / (i + 1)}
                pbar.set_postfix(dicts)  # 进度条右提示
                pbar.update(1)

                # 将loss写入tensorboard，每一步都写
                writer.add_scalar('Train_loss', loss, (epoch * train_epoch_size + i))

        model.eval()
        print("Start validation")
        start_time = time.time()
        with tqdm(total=val_epoch_size, desc=f'Epoch {epoch + 1} / {epochs}') as pbar:  # 进行验证
            for i, (val_data, targets_val) in enumerate(val):  # i是索引，(data, targets)是train中的元素
                if i >= val_epoch_size:
                    break
                with torch.no_grad():  # 用来禁止梯度进行计算
                    images_val = torch.from_numpy(val_data).cuda()  # 转换成张量的形式
                    targets_val = [torch.from_numpy(ann) for ann in targets_val]  # 转换成张量数组的形式
                    optimizer.zero_grad()  # 将模型的梯度初始化为0
                    outputs = model(images_val)  # 前向传播计算预测值

                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets_val)  # 计算当前损失
                        losses.append(loss_item[0])
                    loss = sum(losses)  # 一次验证的损失和
                    val_loss += loss

                    spend_time = time.time() - start_time  # 验证所用的时间
                    dicts = {"step/s": spend_time, "val_loss": val_loss.item() / (i + 1)}
                    pbar.set_postfix(dicts)  # 进度条右提示
                    pbar.update(1)

                    # 将loss写入tensorboard，每个迭代保存一次
                    writer.add_scalar('Val_loss', val_loss / (val_epoch_size + 1), epoch)
                    writer.close()
        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(epochs))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))

        print('Saving state, iter:', str(epoch + 1))
        torch.save(models.state_dict(), 'mod_097_yoloanchor/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % ((epoch + 1), total_loss / (train_epoch_size + 1), val_loss / (val_epoch_size + 1)))
        lr_scheduler.step()


if __name__ == "__main__":
    train()
