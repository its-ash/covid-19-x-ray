import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional
from torch import Tensor
import torch.optim as optim
from collections import namedtuple
from data_tranform import dataloader, BATCH_SIZE


class MyNet(nn.Module):

    def __get_model(self, _input, _output, _dropout=0.2):
        return nn.Sequential(
            nn.Conv2d(_input, _output, 3),
            nn.BatchNorm2d(_output),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(_dropout)
        )

    def __get_model_linear(self, _input, _output, _dropout=0.2):
        return nn.Sequential(
            nn.Linear(_input, _output),
            nn.Dropout(_dropout),
            nn.ReLU(),
        )
    def __init__(self, num_class):
        super(MyNet, self).__init__()
        self.num_class = num_class
        # 2D Layer
        self.layer1 = self.__get_model(1, 32)
        self.layer2 = self.__get_model(32, 64)
        self.layer3 = self.__get_model(64, 64)
        self.layer4 = self.__get_model(64, 128)
        self.layer5 = self.__get_model(128, 64)
        self.layer6 = self.__get_model(64, 32)

        # Linear Input
        self.layer_flat1 = self.__get_model_linear(32 * 2 * 2, 4094, _dropout=0.5)
        self.layer_flat2 = self.__get_model_linear(4094, 2048)
        self.layer_flat3 = self.__get_model_linear(2048, 1024)
        self.layer_flat4 = self.__get_model_linear(1024, 512)
        self.layer_flat5 = self.__get_model_linear(512, 1024)
        self.layer_flat6 = self.__get_model_linear(1024, 2116)

        # 2D Layer
        self.layer7 = self.__get_model(1, 128)
        self.layer8 = self.__get_model(128, 512)
        self.layer9 = self.__get_model(512, 128)

        # Linear Input
        self.layer_flat7 = self.__get_model_linear(128 * 4 * 4, 1024)
        self.layer_flat8 = self.__get_model_linear(1024, 512)

        self.final = nn.Sequential(nn.Linear(512, self.num_class), nn.Softmax())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, 32*2*2)
        x = self.layer_flat1(x)
        x = self.layer_flat2(x)
        x = self.layer_flat3(x)
        x = self.layer_flat4(x)
        x = self.layer_flat5(x)
        x = self.layer_flat6(x)
        x = x.reshape(-1,1, 46, 46)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = x.view(-1, 128 * 4 * 4 )
        x = self.layer_flat7(x)
        x = self.layer_flat8(x)
        x = self.final(x)
        return x


net = MyNet(3)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(1):
    net.train()
    running_loss = 0.0
    for batch, data in enumerate(dataloader):
        image, lable = data
        output = net(image)
        loss = criterion(output, lable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.data.size(0)
    print('Epoch: {}, Avg. Loss: {}'.format(
        epoch + 1, running_loss/len(dataloader.dataset)))

