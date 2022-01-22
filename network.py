import torch.nn as nn
from torchsummary import summary

class conv_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        nn.init.kaiming_normal_(self.conv.weight)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channels, eps=1e-5, momentum=0.997)
        self.activate = nn.ELU()

    def forward(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.batch_norm(outputs)
        outputs = self.activate(outputs)

        return outputs

class linear_block(nn.Module):

    def __init__(self, in_features, out_features):
        super(linear_block, self).__init__()

        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        #nn.init.xavier_normal_(self.linear.weight)
        #self.activate = nn.Tanh()
        nn.init.kaiming_normal_(self.linear.weight)
        self.activate = nn.ELU()

    def forward(self, inputs):
        outputs = self.linear(inputs)
        outputs = self.activate(outputs)

        return outputs

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(conv_block(in_channels=3, out_channels=16, kernel_size=5, stride=2))
        self.layers.append(conv_block(in_channels=16, out_channels=32, kernel_size=5, stride=2))
        self.layers.append(conv_block(in_channels=32, out_channels=48, kernel_size=5, stride=2))
        self.layers.append(conv_block(in_channels=48, out_channels=64, kernel_size=5, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Dropout())
        #self.layers.append(linear_block(in_features=576, out_features=10))
        #self.layers.append(linear_block(in_features=10, out_features=1))
        self.layers.append(nn.Linear(in_features=576, out_features=10))
        self.layers.append(nn.ELU())
        self.layers.append(nn.Linear(in_features=10, out_features=1))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

class PilotNet(nn.Module):

    def __init__(self):
        super(PilotNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(in_features=1152, out_features=100))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=100, out_features=50))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=50, out_features=10))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=10, out_features=1))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

class JNet(nn.Module):

    def __init__(self):
        super(JNet, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(in_features=5376, out_features=10))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(in_features=10, out_features=1))

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

if __name__ == '__main__':
    net = Network()
    #net = PilotNet()
    #net = JNet()
    net.cuda()
    summary(net, (3, 64, 200))