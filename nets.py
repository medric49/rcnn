from torch import nn
import torch


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=4, padding=0)
        nn.init.zeros_(self.conv11.bias)
        self.conv12 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(11, 11), stride=4, padding=0)
        nn.init.zeros_(self.conv12.bias)
        self.batchNorm11 = nn.BatchNorm2d(48)
        self.batchNorm12 = nn.BatchNorm2d(48)

        self.conv21 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        nn.init.ones_(self.conv21.bias)
        self.conv22 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(5, 5), stride=1, padding=2)
        nn.init.ones_(self.conv22.bias)
        self.batchNorm21 = nn.BatchNorm2d(128)
        self.batchNorm22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.zeros_(self.conv31.bias)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.zeros_(self.conv32.bias)

        self.conv41 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.ones_(self.conv41.bias)
        self.conv42 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.ones_(self.conv42.bias)

        self.conv51 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.ones_(self.conv51.bias)
        self.conv52 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        nn.init.ones_(self.conv52.bias)

        self.dense61 = nn.Linear(in_features=2 * 6 * 6 * 128, out_features=2048)
        nn.init.ones_(self.dense61.bias)
        self.dense62 = nn.Linear(in_features=2 * 6 * 6 * 128, out_features=2048)
        nn.init.ones_(self.dense62.bias)

        self.dense71 = nn.Linear(in_features=2 * 2048, out_features=2048)
        nn.init.ones_(self.dense71.bias)
        self.dense72 = nn.Linear(in_features=2 * 2048, out_features=2048)
        nn.init.ones_(self.dense72.bias)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(0.5)

        def init_weights(layer):
            if type(layer) in [nn.Linear, nn.Conv2d]:
                nn.init.normal_(layer.weight, mean=0, std=0.01)

        self.apply(init_weights)

    def forward(self, x):
        x11 = self.maxpool(self.batchNorm11(torch.relu(self.conv11(x))))
        x12 = self.maxpool(self.batchNorm12(torch.relu(self.conv12(x))))

        x21 = self.maxpool(self.batchNorm21(torch.relu(self.conv21(x11))))
        x22 = self.maxpool(self.batchNorm22(torch.relu(self.conv22(x12))))

        x2 = torch.cat([x21, x22], dim=1)

        x31 = torch.relu(self.conv31(x2))
        x32 = torch.relu(self.conv32(x2))

        x41 = torch.relu(self.conv41(x31))
        x42 = torch.relu(self.conv42(x32))

        x51 = self.maxpool(torch.relu(self.conv51(x41)))
        x52 = self.maxpool(torch.relu(self.conv52(x42)))

        x5 = torch.cat([x51, x52], dim=1)

        x5 = torch.flatten(x5, start_dim=1)

        x61 = self.dropout(torch.relu(self.dense61(x5)))
        x62 = self.dropout(torch.relu(self.dense62(x5)))

        x6 = torch.cat([x61, x62], dim=1)

        x71 = self.dropout(torch.relu(self.dense71(x6)))
        x72 = self.dropout(torch.relu(self.dense72(x6)))

        x7 = torch.cat([x71, x72], dim=1)
        return x7


class DCNNClassifierNet(nn.Module):
    def __init__(self):
        super(DCNNClassifierNet, self).__init__()
        self.dense = nn.Linear(in_features=4096, out_features=1000)
        nn.init.normal_(self.dense.weight, mean=0, std=0.01)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=1)
        return x


class RCNNClassifierNet(nn.Module):
    def __init__(self):
        super(RCNNClassifierNet, self).__init__()
        self.dense = nn.Linear(in_features=4096, out_features=21)

    def forward(self, x):
        x = self.dense(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=1)
        return x


class DCNNet(nn.Module):
    def __init__(self):
        super(DCNNet, self).__init__()
        self.base = BaseNet()
        self.classifier = DCNNClassifierNet()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x


class RCNNet(nn.Module):
    def __init__(self):
        super(RCNNet, self).__init__()
        self.base = BaseNet()
        self.classifier = RCNNClassifierNet()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x
