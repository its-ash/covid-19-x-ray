import torch
import torch.nn as nn


class MyNet(nn.Module):

    def __conv_model__(self, _input, _output, _filter=3, _dropout=0.5):
        return nn.Sequential(
            nn.Conv2d(_input, _output, _filter),
            nn.BatchNorm2d(_output),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(_dropout)
        )

    def __linear__model__(self, _input, _output, _dropout=0.5):
        return nn.Sequential(
            nn.Linear(_input, _output),
            nn.Dropout(_dropout),
            nn.ReLU(),
        )

    def __init__(self, num_class):
        super(MyNet, self).__init__()
        self.num_class = num_class
        # 2D Layer
        self.layer1 = self.__conv_model__(1, 16)
        self.layer2 = self.__conv_model__(16, 32)
        self.layer3 = self.__conv_model__(32, 64)
        self.layer4 = self.__conv_model__(64, 128)
        self.layer5 = self.__conv_model__(128, 256)
        self.layer6 = self.__conv_model__(256, 512, 5)

        self.linear_size = 512 * 1 * 1

        self.layer7 = self.__linear__model__(self.linear_size, 512)
        self.layer8 = self.__linear__model__(512, 256)
        self.layer9 = self.__linear__model__(256, 128)
        self.layer10 = self.__linear__model__(128, 128)
        self.layer11 = self.__linear__model__(128, 512)
        self.final = nn.Sequential(
            nn.Linear(512, num_class), nn.Softmax(dim=1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(-1, self.linear_size)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.final(x)
        return x
