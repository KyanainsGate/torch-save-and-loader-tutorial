import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, xavier_uniform=True):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        if xavier_uniform:
            self._xiver_initiaizer(self)
        self.relu = nn.ReLU(inplace=True)
        # Like Call by Reference, Up the memory efficiency

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs

    def _xiver_initiaizer(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:  # バイアス項がある場合
                init.constant_(m.bias, 0.0)
        pass


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cbnr_1 = conv2DBatchNormRelu(
            in_channels=1,
            out_channels=64,
            kernel_size=4,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )  # (14, 14)

        self.cbnr_2 = conv2DBatchNormRelu(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )  # (7, 7)

        self.cbnr_3 = conv2DBatchNormRelu(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            bias=False,
        )  # (4,4)
        # self.pool = nn.MaxPool2d(2, 2)  # 24x24x64 -> 12x12x64
        self.fc1 = nn.Linear(4 * 4 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.classifier = nn.Softmax(dim=1)
        self.metric = 0  # used for learning rate policy 'plateau'

    def forward(self, x):
        x = self.cbnr_1(x)
        # print(x.shape)
        x = self.cbnr_2(x)
        # print(x.shape)
        x = self.cbnr_3(x)
        # print(x.shape)
        x = x.view(-1, 4 * 4 * 64)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = F.relu(self.fc2(x))
        # print(x.shape)
        outputs = self.classifier(x)
        # print(outputs.shape)
        return outputs


class Net(nn.Module):
    def __init__(self, inpt_channel=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(inpt_channel, 32, 3)  # 28x28x32 -> 26x26x32
        self.conv2 = nn.Conv2d(32, 64, 3)  # 26x26x64 -> 24x24x64
        self.pool = nn.MaxPool2d(2, 2)  # 24x24x64 -> 12x12x64
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Linear(12 * 12 * 64, 128)
        self.dropout2 = nn.Dropout2d()
        self.fc2 = nn.Linear(128, 10)
        self.metric = 0  # used for learning rate policy 'plateau'

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 12 * 12 * 64)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    net = CNN()

    print(type(net.parameters()))

    params_to_update = []
    for name, param in net.named_parameters():
        # if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
        # else:
        param.requires_grad = False

    pass
