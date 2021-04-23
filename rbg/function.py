import torch


def onehot(label, num_class):
    return torch.eye(num_class)[label]


def do_nothing(x):
    return x


def onehot_cross(x, t):
    y = torch.nn.functional.log_softmax(x, dim=-1)
    loss = -torch.mean(torch.sum(y * t, dim=-1))
    if loss < 0:
        print(y[:5])
        print(t[:5])
    return loss
