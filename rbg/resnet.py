from torchvision.models.resnet import BasicBlock
import torch.nn as nn
import torch


def make_layer(block, inplanes, planes, block_num,
               stride=1, downsample=None):
    if stride != 1:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes))
    layers = [block(inplanes, planes, stride, downsample)]
    for _ in range(1, block_num):
        layers.append(block(planes, planes))
    return nn.Sequential(*layers)


class CifarResNet(nn.Module):
    def __init__(self, layer_sizes, num_class=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = make_layer(BasicBlock, 16, 16, layer_sizes[0])
        self.layer2 = make_layer(BasicBlock, 16, 32, layer_sizes[1], stride=2)
        self.layer3 = make_layer(BasicBlock, 32, 64, layer_sizes[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


model_params = {
  "resnet20": [3] * 3,
  "resnet32": [5] * 3,
  "resnet44": [7] * 3,
  "resnet56": [9] * 3,
  "resnet110": [18] * 3,
  "resnet1202": [200] * 3
}


def get_cifar_resnet(model_name, num_class=10):
    return CifarResNet(model_params[model_name], num_class)


def is_cifar_resnet(model_name):
    return model_name in model_params.keys()


if __name__ == '__main__':
    net = get_model("resnet20")
    print(net)
    x = torch.empty(10, 3, 32, 32)
    net(x)
    print(is_cifar_resnet("resnet56"))
