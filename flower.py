import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings

warnings.filterwarnings("ignore")
import random
import sys
import copy
import json
from PIL import Image
import torch
from torchvision import models

# 读取数据
data_dir = './flower_data/'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
data_transforms = {
    'train':
        transforms.Compose([
            # Compose：按顺序进行组合
            transforms.Resize([96, 96]),
            # 不管原数据的大小，规定用于训练的图片的大小，Resize根据实际来
            # 以下6行代码是数据增强的过程，数据不够时，通过数据增强，更高效的利用数据，平移，翻转，放大等方法让数据具有更多的多样性
            transforms.RandomRotation(45),  # 随机旋转，-45到45度之间随机选
            transforms.CenterCrop(64),  # 从中心开始裁剪
            # 从96*96 随机裁剪64*64也有无数种可能性
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率概率
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
            # p=0.5指的是每张图像有50%的裁剪可能性
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相
            # 参数1为亮度，参数2为对比度，参数3为饱和度，参数4为色相，不是重点，考虑极端光线条件
            transforms.RandomGrayscale(p=0.025),  # 概率转换成灰度率，3通道就是R=G=B
            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差

        ]),
    'valid':

        transforms.Compose([
            transforms.Resize([64, 64]),

            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ]),
}
batch_size = 128

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True) for x in
               ['train', 'valid']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

model_name = 'resnet'

feature_extract = True
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():

            param.requires_grad = False


model_ft = models.resnet18()
print(model_ft)


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)

    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 64
    return model_ft, input_size


model_ft, input_size = initialize_model(model_name, 9, feature_extract, use_pretrained=True)


model_ft = model_ft.to(device)

# 模型保存
filename = 'best.pt'

params_to_update = model_ft.parameters()

print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)
print(model_ft)
# 优化器设置
optimizer_ft = optim.Adam(params_to_update, lr=1e-2)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
# 学习率衰减，随着epoch的进行，结果会越来越好，降低学习率，使结果更精确
# 以固定的间隔（每10个epoch）将学习率缩小为当前值的10%。这是为了在训练过程中逐渐减小学习率，以帮助模型在训练后期更好地收敛。
criterion = nn.CrossEntropyLoss()


# 交叉熵损失函数
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, filename='best.pt'):

    since = time.time()
    # 记录acc最好的那一次
    best_acc = 0
    # 最后的epoch结果的准确率比中间epoch准确率的结果差也是有可能的
    model.to(device)
    # 训练过程中打印一堆损失和指标
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses =
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # 放到你的CPU或GPU
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # 计算损失（先进行累加）
                running_loss += loss.item() * inputs.size(0)  # 0表示batch那个维度
                running_corrects += torch.sum(preds == labels.data)  # 预测结果最大的和真实值是否一致

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 训练阶段和验证阶段都要进行前向传播，但是训练阶段还要进行参数更新，而验证阶段不需要
            # 得到最好那次的模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),  # 字典里key就是各层的名字，值就是训练好的权重
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, filename)
            if phase == 'valid':
                val_acc_history.append(epoch_acc)
                valid_losses.append(epoch_loss)

            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_losses.append(epoch_loss)

        print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])
        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,
                                                                                            num_epochs=20)
for param in model_ft.parameters():
    param.requires_grad = True


optimizer = optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 损失函数
criterion = nn.CrossEntropyLoss()
# 加载之前训练好的权重参数

checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])
model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer,
                                                                                            num_epochs=10, )
model_ft, input_size = initialize_model(model_name,9, feature_extract, use_pretrained=True)


model_ft = model_ft.to(device)

# 保存文件的名字
filename = 'best.pt'

# 加载模型
checkpoint = torch.load(filename)
best_acc = checkpoint['best_acc']
model_ft.load_state_dict(checkpoint['state_dict'])

dataiter = iter(dataloaders['valid'])
images, labels = dataiter.__next__()


model_ft.eval()

if train_on_gpu:
    output = model_ft(images.cuda())
else:
    output = model_ft(images)
_, preds_tensor = torch.max(output, 1)

preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
preds


def im_convert(tensor):
    """ 展示数据"""

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


fig = plt.figure(figsize=(20, 20))
columns = 3
rows = 3

for idx in range(columns * rows):
    ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
    plt.imshow(im_convert(images[idx]))
    predicted_name = cat_to_name.get(str(preds[idx].item()+1), "Unknown")
    actual_name = cat_to_name.get(str(labels[idx].item()+1), "Unknown")
    ax.set_title("{} ({})".format(predicted_name, actual_name),
                 color=("green" if predicted_name == actual_name else "red"))
plt.show()
