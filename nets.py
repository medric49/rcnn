from torch import nn
import torch


class DefaultDCNN(nn.Module):
    def __init__(self):
        super(DefaultDCNN, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=4, padding=0)
        self.conv12 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=4, padding=0)
        self.batchNorm11 = nn.BatchNorm2d(48)
        self.batchNorm12 = nn.BatchNorm2d(48)

        self.conv21 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        self.conv22 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        self.batchNorm21 = nn.BatchNorm2d(128)
        self.batchNorm22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)

        self.conv41 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        self.conv42 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)

        self.conv51 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv52 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)

        self.dense61 = nn.Linear(in_features=2*6*6*128, out_features=2048)
        self.dense62 = nn.Linear(in_features=2*6*6*128, out_features=2048)

        self.dense71 = nn.Linear(in_features=2*2048, out_features=2048)
        self.dense72 = nn.Linear(in_features=2*2048, out_features=2048)

        self.dense8 = nn.Linear(2*2048, out_features=1000)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x11 = self.maxpool(self.batchNorm11(self.relu(self.conv11(x))))
        x12 = self.maxpool(self.batchNorm12(self.relu(self.conv12(x))))

        x21 = self.maxpool(self.batchNorm21(self.relu(self.conv21(x11))))
        x22 = self.maxpool(self.batchNorm22(self.relu(self.conv22(x12))))

        x2 = torch.cat([x21, x22], dim=1)

        x31 = self.relu(self.conv31(x2))
        x32 = self.relu(self.conv32(x2))

        x41 = self.relu(self.conv41(x31))
        x42 = self.relu(self.conv42(x32))

        x51 = self.maxpool(self.relu(self.conv51(x41)))
        x52 = self.maxpool(self.relu(self.conv52(x42)))

        x5 = torch.cat([x51, x52], dim=1)

        x5 = self.flatten(x5)

        x61 = self.relu(self.dense61(x5))
        x62 = self.relu(self.dense62(x5))

        x6 = torch.cat([x61, x62], dim=1)

        x71 = self.relu(self.dense71(x6))
        x72 = self.relu(self.dense72(x6))

        x7 = torch.cat([x71, x72], dim=1)

        x8 = self.relu(self.dense8(x7))

        x = self.softmax(x8)

        return x





