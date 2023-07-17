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
import random
import sys
import copy
import json
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# def load_annotations(ann_file):
#     """
#     读标注文件
#     :param ann_file: 标注文件的地址
#     :return: 字典类型的数据 {image ，标注}
#     """
#     data_infos = {}
#     with open(ann_file) as f:
#         samples = [x.strip().split(' ') for x in f.readlines()]
#         for filename, gt_lable in samples:
#             data_infos[filename] = np.array(gt_lable, dtype=np.int64)
#     return data_infos


# img_lable = load_annotations('train.txt')
# image_name = list(img_lable.keys())
# lable = list(img_lable.values())

train_dir = './train_filelist'
valis_dir = './val_filelist'

from torch.utils.data import Dataset, DataLoader


class FlowerDataset(Dataset):
    """
    必须有2个函数
    def 和 getitem
    """

    def __init__(self, root_dir, ann_file, transform=None):
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def load_annotations(self):
        """
        读标注文件
        :param ann_file: 标注文件的地址
        :return: 字典类型的数据 {image ，标注}
        """
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos

    def __getitem__(self, item):
        image = Image.open(self.img[item])
        label = self.label[item]
        if self.transform:
            image = self.transform(image)
        label = torch.from_numpy(np.array(label))
        return image, label

    def __len__(self):
        return len(self.img)


data_transform = {
    'train ':
        transforms.Compose(
            [
                transforms.Resize(64),
                transforms.RandomRotation(45),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值 标准差
            ]

        ),
    'valid':
        transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值 标准差
        ])
}

train_dataset = FlowerDataset(root_dir=train_dir, ann_file='train.txt', transform=data_transform['train '])
val_dataset = FlowerDataset(valis_dir, 'val.txt', data_transform['valid'])
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

# image, label = next(iter(train_loader))
# sample = image[0].squeeze()
# sample = sample.permute((1, 2, 0)).numpy()
# sample *= [0.229, 0.224, 0.225]
# sample += [0.485, 0.456, 0.406]
# plt.imshow(sample)
# plt.show()
# print('Label is: {}'.format(label[0].numpy()))

dataloaders = {'train': train_loader, 'valid': val_loader}
model_name = 'resnet'
feature_extract = True

# 是否用gpu训练

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available ,Training on cpu...')
else:
    print('CUDA  is available! Training on gpu...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18()
# 修改fc层
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 102))

# 优化器设置
optimizer_ft = optim.Adam(params=model_ft.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer_ft, 7, gamma=0.1)
criterion = nn.CrossEntropyLoss()


def train_model(model, dataloaders, criterion, optimizer, num_epoch=20, filename='best.pth'):
    since = time.time()
    best_acc = 0
    model.to(device)

    val_acc_history = []
    train_acc_history = []
    train_loss = []
    valid_loss = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())
    for epoch in range(num_epoch):
        print('Epoch{}/{}'.format(epoch, num_epoch - 1))
        print('-' * 10)

        # 训练和验证
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # 训练
            else:
                model.eval()  # 验证

            running_loss = 0
            running_corrects = 0

            # 把数据都取了遍
            for input, label in dataloaders[phase]:
                inputs = input.to(device)
                labels = label.to(device)

                # 梯度清零
                optimizer.zero_grad()
                # 只有训练的时候需要梯度更新
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

        time_elapsed = time.time() - since
        print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # 得到最好的模型
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            state = {
                'state': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, filename)

        if phase == 'valid':
            val_acc_history.append(epoch_acc)
            valid_loss.append(epoch_loss)
            scheduler.step(epoch_loss)
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss.append(epoch_loss)

    print('Optimizer learning rate : {:.7f}'.format(optimizer.param_groups[0]['lr']))
    LRs.append(optimizer.param_groups[0]['lr'])
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 训练完后用最好的一次当做模型最终的结果,等着一会测试
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_loss, train_loss, LRs


model_ft, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train_model(model_ft, dataloaders,
                                                                                            criterion, optimizer_ft,

                                                                                            filename='best.pth')
