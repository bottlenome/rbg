import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from rbg.trainer import Trainer
from rbg.function import onehot_cross
from rbg.generalization import RandomBatchGeneralization, BatchGeneralization


def correctness(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item() / len(target) * 100.0


def load_cifer10(data_max, train_kwargs, test_kwargs):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        ])
    dataset1 = datasets.CIFAR10('data', train=True, download=True,
                                transform=transform_train)
    dataset2 = datasets.CIFAR10('data', train=False,
                                transform=transform_test)
    if data_max != -1:
        dataset1 = torch.utils.data.dataset.Subset(
                dataset1, list(range(0, data_max)))
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    return train_loader, test_loader


def device_settings(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        cuda_kwargs = {'num_workers': 0,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        torch.cuda.set_device('cuda:{}'.format(args.gpu))

    device = torch.device("cuda" if use_cuda else "cpu")
    return train_kwargs, test_kwargs, device


class DoNothing(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, y):
        return self.net(x, y)


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
        if self.change_output:
            return x, y
        else:
            return x


def freeze(net):
    for param in net.parameters():
        param.requres_grad = False
    return net


def get_model(model_name, method):
    if method == "scratch" or method == "finetune":
        if model_name == "ViT":
            # TODO
            net = nn.Module()
        else:
            net = eval(f"models.{model_name}(pretrained={method=='finetune'})")
        if method == "finetune":
            net = freeze(net)
        if model_name == "resnet18" or model_name == "resnet34":
            net.fc = nn.Linear(in_features=512, out_features=10)
        elif model_name[:len("resnet")] == "resnet":
            net.fc = nn.Linear(in_features=2048, out_features=10)
        elif model_name[:len("vgg")] == "vgg":
            net.classifier[3] = nn.Linear(in_features=4096, out_features=4096)
            net.classifier[6] = nn.Linear(in_features=4096, out_features=10)
        elif model_name == "ViT":
            # TODO
            pass
        else:
            print(f"unknown model name {model_name}")
            assert(False)
    elif method == "brg":
        if model_name[:len("resnet")] == "resnet":
            net = WrapResNet(model_name, RandomBatchGeneralization,
                             change_output=True)
        elif model_name[:len("vgg")] == "vgg":
            # TODO
            pass
        elif model_name == "ViT":
            # TODO
            pass
    elif method == "bg":
        if model_name[:len("resnet")] == "resnet":
            net = WrapResNet(model_name, BatchGeneralization)
        elif model_name[:len("vgg")] == "vgg":
            # TODO
            pass
            net = nn.Module()
        elif model_name == "ViT":
            # TODO
            pass
            net = nn.Module()
    else:
        print(f"invalid method: {method}")
        assert(False)
    return net


def get_criterion(method):
    if method == "brg":
        return onehot_cross
    else:
        return nn.CrossEntropyLoss()


def main(args):
    train_kwargs, test_kwargs, device = device_settings(args)
    train_loader, test_loader = load_cifer10(args.data_max,
                                             train_kwargs,
                                             test_kwargs)
    for model_name in args.models:
        for method in args.methods:
            print(f"Model: {model_name}, Method: {method}")
            net = get_model(model_name, method)
            criterion = get_criterion(method)
            score = correctness
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            t = Trainer(net, criterion, score, optimizer,
                        method, train_loader, test_loader,
                        model_name=model_name,
                        device=device,
                        debug=args.debug)
            t.train(args.epochs, debug=args.debug)


def get_args():
    parser = argparse.ArgumentParser(description='RBG trainer')
    parser.add_argument('--batch-size', type=int,
                        default=200, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--test-batch-size', type=int,
                        default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--gpu', type=int, default=0, metavar='G',
                        help='gpu num (default: 0)')
    parser.add_argument('--data_max', type=int, default=-1, metavar='N',
                        help='train data max size')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--models', type=list,
                        nargs='+',
                        default=["resnet18", "resnet152",
                                 "vgg11_bn", "vgg19_bn",
                                 "ViT"],
                        metavar='N',
                        help='target model names')
    parser.add_argument('--methods', type=list,
                        nargs='+',
                        default=["brg", "bg",
                                 "finetune", "scratch"],
                        metavar='N',
                        help='target methods')
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    return args


if __name__ == '__main__':
    main(get_args())
