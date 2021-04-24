import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from rbg.trainer import Trainer
from rbg.function import onehot_cross
from rbg.model import get_model


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


def get_criterion(method):
    if method == "rbg":
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
            net, preprocess = get_model(model_name, method)
            criterion = get_criterion(method)
            score = correctness
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
            t = Trainer(net, criterion, score, optimizer,
                        method, train_loader, test_loader,
                        preprocess_target=preprocess,
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
                        default=["ViT",
                                 "resnet18", "resnet152",
                                 "vgg11_bn", "vgg19_bn"],
                        metavar='N',
                        help='target model names')
    parser.add_argument('--methods', type=list,
                        nargs='+',
                        default=["rbg", "bg",
                                 "finetune", "scratch"],
                        metavar='N',
                        help='target methods')
    parser.add_argument("--debug", action='store_true', default=False)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    return args


if __name__ == '__main__':
    main(get_args())
