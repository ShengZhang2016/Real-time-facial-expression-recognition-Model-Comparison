# -*- coding: utf-8 -*-
# @Author: zhangsheng
# @Date:   2018-03-12 19:55:47
# @Last Modified by:   zhangsheng
# @Last Modified time: 2018-03-12 22:48:14

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
import time
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(42),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize()
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        transforms.CenterCrop(42),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize()
    ])
}

data_dir = '../Datasets/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['train', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

INPUT_SIZE = 42

def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out


def test_model():
    inputs, labels = next(iter(dataloaders['train']))
    print(inputs.size())
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in classes])
    model = RNN()
    if use_gpu:
        model = model.cuda()
    # print(model)
    outputs = model(inputs)
    # print(outputs)


def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    accuracies = []
    all_losses = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                inputs, labels = data

                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs.view(-1, 42, 42)), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            all_losses.append(epoch_loss)
            accuracies.append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))

        # return model
        model.load_state_dict(best_model_wts)
        torch.save(model, 'myRNN_best_model.pkl')
        torch.save(model.state_dict(), 'myRNN_model_params.pkl')

    plt.figure()
    plt.plot(all_losses)
    plt.plot(accuracies)

def testImage1(model):
    testImage = "../webcam/images1.jpeg"
    # normalize = transforms.Normalize(
    #    mean=[0.485, 0.456, 0.406],
    #    std=[0.229, 0.224, 0.225]
    # )

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(42),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])

    img_pil = Image.open(testImage)
    model = RNN()
    img_tensor = preprocess(img_pil)
    # img_tensor.unsqueeze_(0)

    img_variable = Variable(img_tensor)
    # model.eval()
    fc_out = model(img_variable)
    preds = torch.max(fc_out, 1)[1]

    labels = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
               'Sad': 4, 'Surprise': 5, 'Neutral': 6}
    # print("predict is : ", preds.data.squeeze())
    # print(labels[preds.data.squeeze()])
    print(labels[preds])

if __name__ == '__main__':
    
    model = RNN()
    if use_gpu:
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_model(model, criterion, optimizer, num_epochs=50)

    

