import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from rbg.trainer import Trainer
from rbg.function import onehot_cross
from rbg.model import get_model
import numpy as np


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
            for rate in args.rates:
                print(f"Model: {model_name}, Method: {method}, Rate: {rate}")
                net, preprocess = get_model(model_name, method, rate)
                criterion = get_criterion(method)
                score = correctness
                if args.optmizer == "adam":
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
                    scheduler = SchedulerDonothing()
                elif args.optmizer == "sgd":
                    optimizer = torch.optim.SGD(
                            net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=200)
                else:
                    print(f"unsupported optimizer{args.optmizer}")
                    assert(False)
                start_epoch = 0
                best_score = 0
                if args.resume is not None:
                    model = torch.load(
                            args.resume, map_location=torch.device('cpu'))
                    net.load_state_dict(model["model"])
                    optimizer.load_state_dict(model["optmizer"])
                    start_epoch = model["epoch"]
                    best_score = model["score"]
                t = Trainer(net, criterion, score, optimizer,
                            method, train_loader, test_loader,
                            preprocess_target=preprocess,
                            model_name=f"{model_name}_{rate}",
                            device=device,
                            debug=args.debug,
                            scheduler=scheduler)
                t.train(args.epochs, debug=args.debug,
                        start_epoch=start_epoch, best_score=best_score)


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
    parser.add_argument('--models',
                        nargs='+',
                        default=["ViT",
                                 "resnet18", "resnet152",
                                 "vgg11_bn", "vgg19_bn"],
                        metavar='N',
                        help='target model names')
    parser.add_argument('--methods',
                        nargs='+',
                        default=["rbg", "bg",
                                 "finetune", "scratch"],
                        metavar='N',
                        help='target methods')
    parser.add_argument('--rates', type=float,
                        nargs='+',
                        default=[0.1],
                        help='augument rate in layer')
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None,
                        help='saved model path')
    parser.add_argument("--optimizer", type=str, deault="adam",
                        help="optmizer type (adam/sgd)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    return args


if __name__ == '__main__':
    main(get_args())
