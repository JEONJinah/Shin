import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader

import ClassNet
import os
import pandas as pd
from PIL import Image


# path_dir = 'D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/' \
#        '롯데정보통신_MegaProduct/LPD_competition/test'
# file_list = os.listdir(path_dir)
# print(len(file_list), type(file_list), file_list[0])
# print(f'{path_dir}/{file_list[0]}')

def show_test_img(image):
    """ 랜드마크와 함께 이미지 보여주기 """
    plt.imshow(image)

class OwnDataSet(Dataset):
    """ 얼굴 랜드마크 데이터셋. """

    def __init__(self, csv_file, root_dir, transform=None):

        self.num_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.num_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.num_frame.iloc[idx, 0])
        image = io.imread(img_name)
        sample = {'image': image}
        if self.transform:
            # pil_img = Image.fromarray(sample['image'])
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """ 주어진 크기로 샘플안에 있는 이미지를 재변환 합니다.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        tuple로 주어진다면 결과값은 output_size 와 동일해야하며
        int일때는 설정된 값보다 작은 이미지들의 가로와 세로는 output_size 에 적절한 비율로 변환됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img}


class CenterCrop(object):
    """ 샘플에 있는 이미지를 무작위로 자르기.

    Args:
        output_size (tuple 또는 int): 원하는 결과값의 크기입니다.
        int로 설정하시면 정사각형 형태로 자르게 됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)
        # print(top, left, new_h, new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}


class ToTensor(object):
    """ 샘플 안에 있는 n차원 배열을 Tensor로 변홥힙니다. """

    def __call__(self, sample):
        image = sample['image']

        # 색깔 축들을 바꿔치기해야하는데 그 이유는 numpy와 torch의 이미지 표현방식이 다르기 때문입니다.
        # numpy 이미지: H x W x C
        # torch 이미지: C X H X W
        image = image.transpose((2, 0, 1))  #channel, w, h
        # image = image.transpose((0, 1, 2))
        return {'image': torch.from_numpy(image)}

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    lotte_dataset = OwnDataSet(csv_file='D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/'
                                        '롯데정보통신_MegaProduct/LPD_competition/sample.csv',
                               root_dir='D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/'
                                        '롯데정보통신_MegaProduct/LPD_competition/test/',
                               transform=transforms.Compose([
                                   Rescale(256),
                                   CenterCrop(224),
                                   ToTensor()
                                ]))
    test_loader = DataLoader(lotte_dataset, batch_size=10,
                             shuffle=False, num_workers=4)
    df = pd.read_csv('D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/'
                     '롯데정보통신_MegaProduct/LPD_competition/sample.csv')

    print(len(test_loader), test_loader.batch_size)

    train_dataset = ImageFolder(root='D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/'
                                     '롯데정보통신_MegaProduct/LPD_competition/train', transform=None)
    class_names = train_dataset.classes

    net = models.vgg16_bn(pretrained=True)
    features = list(net.classifier.children())[:-1]
    features.extend([nn.Linear(4096, 2048)])
    features.extend([nn.ReLU(inplace=True)])
    features.extend([nn.Dropout(0.5)])
    features.extend([nn.Linear(2048, 1000)])
    net.classifier = nn.Sequential(*features)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # if torch.cuda.device_count() >= 1:
    #     net = nn.DataParallel(net)

    # value = 2
    # df.prediction[value] = int(494)
    # print(df.loc[0:3])
    # df.to_csv("test_modified.csv", sep=",", index=False)

    net.load_state_dict(torch.load("./VGG16_v3-lotte_0322.pth"))
    net.train(False)
    net.eval()
    column_idx = 0
    for data in test_loader:
        images = data
        images = (images['image'].to(device)).float()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        batch_idx = 0
        for idx in predicted:
            print(class_names[idx], end=" ")
            value = int((test_loader.batch_size * column_idx) + batch_idx)
            df.prediction[value] = int(class_names[idx])
            batch_idx += 1
        print()
        column_idx += 1
    df.to_csv("lotte_prediction.csv", sep=",", index=False)


    # image = Image.open("D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/"
    #                    "롯데정보통신_MegaProduct/LPD_competition/test/10.jpg")
    # area = (16, 16, 240, 240)
    # crop_img = image.crop(area)
    # crop_img.show()
    # np_array = np.array(crop_img)
    # np_array = np_array/255
    # np_array = np.transpose(np_array, (2, 0, 1))
    # np_array = np_array[np.newaxis, :, :, :]
    # # np_array = np.resize(np_array, (1, 3, 224, 224))
    # x_np = torch.from_numpy(np_array)
    # print(type(x_np))
    # net = models.vgg16_bn()
    # features = list(net.classifier.children())[:-1]
    # features.extend([nn.Linear(4096, 2048)])
    # features.extend([nn.ReLU(inplace=True)])
    # features.extend([nn.Dropout(0.5)])
    # features.extend([nn.Linear(2048, 1000)])
    # net.classifier = nn.Sequential(*features)
    # net.load_state_dict(torch.load("./VGG16_v2-lotte_0321.pth"))
    # # print(net.classifier[6].out_features) # 1000
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # net.to(device)
    # if torch.cuda.device_count() >= 1:
    #     print('\n====> Training on GPU!')
    #     net = nn.DataParallel(net)
    # net.train(False)
    # net.eval()
    # # data_iter = iter(test_loader)
    # # images = data_iter.next()
    # # inputs = images['image']
    # # inputs = inputs.float()
    # inputs = x_np.float()
    # outputs = net(inputs)
    # # imshow(torchvision.utils.make_grid(images['image']))
    # _, predicted = torch.max(outputs.data, 1)
    # print(predicted)
    # images = data_iter.next()
    # inputs = images['image']
    # inputs = inputs.float()
    # outputs = net(inputs)
    # imshow(torchvision.utils.make_grid(images['image']))
    # _, predicted = torch.max(outputs.data, 1)
    # print(predicted)
    # images = data_iter.next()
    # inputs = images['image']
    # inputs = inputs.float()
    # outputs = net(inputs)
    # imshow(torchvision.utils.make_grid(images['image']))
    # _, predicted = torch.max(outputs.data, 1)
    # print(predicted)


# fig = plt.figure()
#
# for i in range(len(lotte_dataset)):
#     sample = lotte_dataset[i]
#
#     print(i, sample['image'].shape)
#
#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     show_test_img(**sample)
#
#     if i == 3:
#         plt.show()
#         break




# infer_data = pd.read_csv(PATH)

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
# if __name__ == '__main__':
#
#     transformer = transforms.Compose(
#             [transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor()]
#     )
#
#     test_dataset = ImageFolder(root='D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/'
#                                      '롯데정보통신_MegaProduct/LPD_competition/test', transform=transformer)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2)
#
#     net = models.vgg16_bn()
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     net.to(device)
#     if torch.cuda.device_count() >= 1:
#         print('\n====> Training on GPU!')
#         net = nn.DataParallel(net)
#
#     net.load_state_dict(torch.load("./VGG16_v2-lotte_0320.pth"))
#     data_iter = iter(test_loader)
#     images, labels = data_iter.next()
#     outputs = net(images)
#     imshow(torchvision.utils.make_grid(images))
#     _, predicted = torch.max(outputs.data, 1)
#     print(predicted)

    # img = images[0].numpy()
    # img1 = images[1].numpy()
    # plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.show()
    # plt.imshow(np.transpose(img1, (1, 2, 0)))
    # plt.show()

    # for data in test_loader:
    #     images, labels = data
    #     images, labels = images.to(device), labels.to(device)
    #     outputs = net(images)
    #     _, predicted = torch.max(outputs.data, 1)
