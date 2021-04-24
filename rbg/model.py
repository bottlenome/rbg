import torch
import torch.nn as nn
from .generalization import RandomBatchGeneralization, BatchGeneralization
from torchvision import models
from .function import onehot, do_nothing


class DoNothing(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, y):
        return self.net(x, y)


class OneHot():
    def __init__(self, num_class=10):
        self.num_class = num_class

    def __call__(self, x):
        return onehot(x, self.num_class)


class WrapNormalVGG(nn.Module):
    def __init__(self, model_name, method):
        super().__init__()
        self.net = eval(
                f"models.{model_name}(pretrained={method=='finetune'})")
        if method == "finetune":
            self.net = freeze(self.net)
        self.net.classifier[0] = nn.Linear(
                in_features=25088, out_features=4096)
        self.net.classifier[3] = nn.Linear(
                in_features=4096, out_features=4096)
        self.net.classifier[6] = nn.Linear(
                in_features=4096, out_features=10)

    def forward(self, x, _):
        return self.net(x), _


class WrapVGG(nn.Module):
    def __init__(self, model_name, generalization,
                 rate=0.1, epsilon=0.4, finetune=False,
                 analyze=DoNothing, change_output=False):
        super().__init__()
        if model_name == "vgg11_bn":
            self.insert_points = [4, 8, 11, 15, 18, 22, 25]
        elif model_name == "vgg13_bn":
            self.insert_points = [3, 7, 10, 14, 17, 21, 24, 28, 31]
        elif model_name == "vgg16_bn":
            self.insert_points = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        elif model_name == "vgg19_bn":
            self.insert_points = [3, 7, 10, 14, 17, 20, 23, 27,
                                  30, 33, 36, 40, 43, 46, 49]
        else:
            print(f"unsupported vgg model: {model_name}")
            assert(False)
        self.layers = nn.ModuleList([])
        for _ in self.insert_points:
            self.layers.append(generalization(rate, epsilon))
        self.net = eval(
                f"models.{model_name}(pretrained={finetune})")
        self.net.classifier[6] = nn.Linear(
                in_features=4096, out_features=10)
        self.change_output = change_output

    def forward(self, x, y):
        for i in range(len(self.net.features)):
            if i in self.insert_points:
                j = self.insert_points.index(i)
                x = self.layers[j](x, y)
                if self.change_output:
                    y = x[1]
                    x = x[0]
            x = self.net.features[i](x)

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x, y


class WrapNormalResNet(nn.Module):
    def __init__(self, model_name, method):
        super().__init__()
        self.net = eval(
                f"models.{model_name}(pretrained={method=='finetune'})")
        if method == "finetune":
            self.net = freeze(self.net)
        if model_name == "resnet18" or model_name == "resnet34":
            self.net.fc = nn.Linear(in_features=512, out_features=10)
        else:
            self.net.fc = nn.Linear(in_features=2048, out_features=10)

    def forward(self, x, _):
        return self.net(x), _


class WrapResNet(nn.Module):
    def __init__(self, model_name, generalization,
                 rate=0.1, epsilon=0.4, finetune=False,
                 analyze=DoNothing, change_output=False):
        super().__init__()
        self.model = eval(f"models.{model_name}(pretrained={finetune})")
        if model_name == "resnet18" or model_name == "resnet34":
            self.model.fc = nn.Linear(in_features=512, out_features=10)
        else:
            self.model.fc = nn.Linear(in_features=2048, out_features=10)
        self.for_layer1 = nn.ModuleList([])
        for _ in range(len(self.model.layer1)):
            self.for_layer1.append(analyze(generalization(rate, epsilon)))
        self.for_layer2 = nn.ModuleList([])
        for _ in range(len(self.model.layer2)):
            self.for_layer2.append(analyze(generalization(rate, epsilon)))
        self.for_layer3 = nn.ModuleList([])
        for _ in range(len(self.model.layer3)):
            self.for_layer3.append(analyze(generalization(rate, epsilon)))
        self.for_layer4 = nn.ModuleList([])
        for _ in range(len(self.model.layer4)):
            self.for_layer4.append(analyze(generalization(rate, epsilon)))
        self.change_output = change_output

    def forward(self, x, y):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i in range(len(self.for_layer1)):
            x = self.for_layer1[i](x, y)
            if self.change_output:
                y = x[1]
                x = x[0]
            x = self.model.layer1[i](x)
        for i in range(len(self.for_layer2)):
            x = self.for_layer2[i](x, y)
            if self.change_output:
                y = x[1]
                x = x[0]
            x = self.model.layer2[i](x)
        for i in range(len(self.for_layer3)):
            x = self.for_layer3[i](x, y)
            if self.change_output:
                y = x[1]
                x = x[0]
            x = self.model.layer3[i](x)
        for i in range(len(self.for_layer4)):
            x = self.for_layer4[i](x, y)
            if self.change_output:
                y = x[1]
                x = x[0]
            x = self.model.layer4[i](x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x, y


def freeze(net):
    for param in net.parameters():
        param.requres_grad = False
    return net


def get_model(model_name, method):
    preprocess = do_nothing
    if method == "scratch" or method == "finetune":
        if model_name == "ViT":
            # TODO
            net = nn.Module()
        elif model_name[:len("resnet")] == "resnet":
            net = WrapNormalResNet(model_name, method)
        elif model_name[:len("vgg")] == "vgg":
            net = WrapNormalVGG(model_name, method)
        else:
            print(f"unknown model name {model_name}")
            assert(False)
    elif method == "rbg":
        if model_name[:len("resnet")] == "resnet":
            net = WrapResNet(model_name, RandomBatchGeneralization,
                             change_output=True)
        elif model_name[:len("vgg")] == "vgg":
            net = WrapVGG(model_name, RandomBatchGeneralization,
                          change_output=True)
        elif model_name == "ViT":
            # TODO
            pass
        preprocess = OneHot(10)
    elif method == "bg":
        if model_name[:len("resnet")] == "resnet":
            net = WrapResNet(model_name, BatchGeneralization)
        elif model_name[:len("vgg")] == "vgg":
            net = WrapVGG(model_name, BatchGeneralization)
        elif model_name == "ViT":
            # TODO
            pass
            net = nn.Module()
    else:
        print(f"invalid method: {method}")
        assert(False)
    return net, preprocess
