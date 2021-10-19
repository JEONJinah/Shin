import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import copy
from PIL import Image
if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")

    TRAIN = 'train'
    data_dir = 'D:/0.2021/0.강의자료/0.AI엔지니어링/2.화_프로젝트실습/롯데정보통신_MegaProduct/LPD_competition' # D:/pythonProject'
    data_transforms = {
        TRAIN: transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally.
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
    }

    image_datasets = {
        x: ImageFolder(
            os.path.join(data_dir, x),
            transform=data_transforms[x]
        )
        for x in [TRAIN]
    }

    data_loaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=24,  #96
            shuffle=True, num_workers=4
        )
        for x in [TRAIN]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN]}
    print(dataset_sizes)

    for x in [TRAIN]:
        print("Loaded {} images under {}".format(dataset_sizes[x], x))

    # class_names = image_datasets[TRAIN].classes
    # print(image_datasets[TRAIN].classes)
    # print(image_datasets[TRAIN].classes[439], image_datasets[TRAIN].classes[122], image_datasets[TRAIN].classes[18])

    # def imshow(inp, title=None):
    #     inp = inp.numpy().transpose((1, 2, 0))
    #     # plt.figure(figsize=(10, 10))
    #     plt.axis('off')
    #     plt.imshow(inp)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)
    #
    # def show_databatch(inputs, classes):
    #     out = torchvision.utils.make_grid(inputs)
    #     imshow(out, title=[class_names[x] for x in classes])
    #
    # # Get a batch of training data
    # inputs, classes = next(iter(data_loaders[TRAIN]))
    # show_databatch(inputs, classes)


    def eval_model(vgg, criterion):
        since = time.time()
        avg_loss = 0
        avg_acc = 0
        loss_test = 0
        acc_test = 0

        test_batches = len(data_loaders[TRAIN])
        print("Evaluating model")
        print('-' * 10)

        for i, data in enumerate(data_loaders[TRAIN]):
            if i % 100 == 0:
                print("\rTest batch {}/{}".format(i, test_batches), end='', flush=True)

            vgg.train(False)
            vgg.eval()
            inputs, labels = data

            if use_gpu:
                inputs, labels = Variable(inputs.cuda(), volatile=True), Variable(labels.cuda(), volatile=True)
            else:
                inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)

            outputs = vgg(inputs)

            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # loss_test += loss.data[0]
            loss_test += loss.data
            acc_test += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        avg_loss = loss_test / dataset_sizes[TRAIN]
        avg_acc = acc_test / dataset_sizes[TRAIN]

        elapsed_time = time.time() - since
        print()
        print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Avg loss (test): {:.4f}".format(avg_loss))
        print("Avg acc (test): {:.4f}".format(avg_acc))
        print('-' * 10)

    def train_model(vgg, criterion, optimizer, scheduler, num_epochs=10):
        since = time.time()
        best_model_wts = copy.deepcopy(vgg.state_dict())
        best_acc = 0.0

        avg_loss = 0
        avg_acc = 0

        train_batches = len(data_loaders[TRAIN])

        for epoch in range(num_epochs):
            print("Epoch {}/{}".format(epoch, num_epochs))
            print('-' * 10)

            loss_train = 0
            acc_train = 0

            vgg.train(True)

            for i, data in enumerate(data_loaders[TRAIN]):
                if i % 100 == 0:
                    print("\rTraining batch {}/{}".format(i, train_batches), end='', flush=True)

                inputs, labels = data

                if use_gpu:
                    inputs, labels = inputs.to(device), labels.to(device)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = vgg(inputs)

                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                loss_train += loss.item()
                acc_train += torch.sum(preds == labels.data)

                del inputs, labels, outputs, preds
                torch.cuda.empty_cache()

            print()

            avg_loss = loss_train / dataset_sizes[TRAIN]
            avg_acc = acc_train / dataset_sizes[TRAIN]

            print()
            print("Epoch {} result: ".format(epoch))
            print("Avg loss (train): {:.4f}".format(avg_loss))
            print("Avg acc (train): {:.4f}".format(avg_acc))
            print('-' * 10)
            print()

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_model_wts = copy.deepcopy(vgg.state_dict())
                print("Model Parameter Update\n")

        elapsed_time = time.time() - since
        print()
        print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
        print("Best acc: {:.4f}".format(best_acc))

        vgg.load_state_dict(best_model_wts)
        return vgg

    vgg16 = models.vgg16_bn(pretrained=True)
    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    features = list(vgg16.classifier.children())[:-1]
    features.extend([nn.Linear(4096, 2048)])
    features.extend([nn.ReLU(inplace=True)])
    features.extend([nn.Dropout(0.5)])
    features.extend([nn.Linear(2048, 1000)])
    vgg16.classifier = nn.Sequential(*features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(vgg16)

    if use_gpu:
        vgg16.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    vgg16 = train_model(vgg16, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
    torch.save(vgg16.state_dict(), 'VGG16_v4-lotte_0322.pth')

    eval_model(vgg16, criterion)
