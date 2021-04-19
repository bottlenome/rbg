import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models


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


def freeze(net):
    for param in net.parameters():
        param.requres_grad = False
    return net


def get_model(model_name, method):
    if model_name == "ViT":
        # TODO
        net = nn.Module()
    else:
        net = eval(f"models.{model_name}(pretrained=True)")
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
    elif method == "scratch":
        pass
    elif method == "brg":
        pass
    elif method == "rg":
        pass

    return net


def train(net, train_kwargs, test_kwargs, device):
    pass


def main(args):
    train_kwargs, test_kwargs, device = device_settings(args)
    train_loader, test_loader = load_cifer10(args.data_max,
                                             train_kwargs,
                                             test_kwargs)
    for model_name in args.models:
        for method in args.methods:
            net = get_model(model_name, method)
            print(f"Model: {model_name}, Method: {method}")
            train(net, train_loader, test_loader, device)


def get_args():
    parser = argparse.ArgumentParser(description='RBG trainer')
    parser.add_argument('--batch-size', type=int,
                        default=200, metavar='N',
                        help='input batch size for training (default: 200)')
    parser.add_argument('--test-batch-size', type=int,
                        default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--gpu', type=int, default=1, metavar='G',
                        help='gpu num (default: 1)')
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
                        default=["vgg11_bn", "vgg19_bn",
                                 "resnet18", "resnet152", "ViT"],
                        metavar='N',
                        help='target model names')
    parser.add_argument('--methods', type=list,
                        nargs='+',
                        default=["brg", "bg",
                                 "finetune", "scratch"],
                        metavar='N',
                        help='target methods')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    return args


if __name__ == '__main__':
    main(get_args())
