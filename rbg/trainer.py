import time
import torch
import torch.nn as nn
from datetime import datetime, timezone, timedelta
from math import log10
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint_sequential
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .function import onehot, do_nothing


class EarlyStopping():
    def __init__(self, partience=7, delta_rate=0.10):
        self.partience = partience
        self.delta_rate = delta_rate
        self.counter = 0
        self.best_score = float('inf')

    def __call__(self, val_loss):
        if (self.best_score) * (1 + self.delta_rate) < val_loss:
            self.counter += 1
            if self.counter >= self.partience:
                return True
        else:
            self.best_score = min(val_loss, self.best_score)
            self.counter = 0
        return False

    def reset(self):
        self.counter = 0
        self.best_score = float('inf')


class SchedulerDonothing():
    def __init__(self, optimizer):
        pass

    def step(self, metrics=None):
        pass


class WriterDoNothing():
    def __init__(self):
        pass

    def add_scalar(self, *args):
        pass


class Trainer():
    def __init__(self, net, criterion, score, optimizer,
                 method, trainloader, testloader,
                 preprocess_target=do_nothing,
                 callbacks=[],
                 model_name=None,
                 device=torch.device("cuda"),
                 timezone=timezone(timedelta(hours=+9), 'JST'),
                 debug=False,
                 scheduler=SchedulerDonothing(None)):
        self.locale = timezone
        self.net = net
        self.criterion = criterion
        self.score = score
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.callbacks = callbacks
        for c in callbacks:
            c.reset()
        self.model_name = f"{model_name}_{method}"
        self.device = device
        self.preprocess_target = preprocess_target
        if debug:
            self.writer = WriterDoNothing()
        else:
            self.writer = SummaryWriter()
        self.debug = debug
        self.scheduler = scheduler

    def loss_bg(self, inputs, targets):
        targets = targets.to(self.device)
        return self.loss_bg_base(inputs, targets, targets)

    def loss_bg_test(self, inputs, targets):
        targets = targets.to(self.device)
        return self.loss_bg_base(inputs, targets, None)

    def loss_bg_base(self, inputs, targets, ref_targets):
        outputs = self.net(inputs.to(self.device),
                           ref_targets)
        return self.criterion(outputs, targets), outputs

    def loss_normal(self, inputs, targets):
        outputs = self.net(inputs.to(self.device))
        return self.criterion(outputs, targets.to(self.device)), outputs

    def save(self, score, epoch):
        checkpoint = {
                'epoch': epoch,
                'score': score,
                'optmizer': self.optimizer.state_dict(),
                'model': self.net.state_dict()}
        if self.model_name is None:
            torch.save(self.net.state_dict(),
                       f'models/{score:.2f}_{epoch}_{time.time()}.pth')
        else:
            torch.save(checkpoint,
                       f'models/{self.model_name}.pth')

    def train_(self, epoch):
        start = time.time()
        batch_size = self.trainloader.batch_size
        total_size = len(self.trainloader) * batch_size
        self.net = self.net.train()
        running_loss = 0.0
        for i, data in enumerate(self.trainloader, 0):
            inputs, targets = data
            self.optimizer.zero_grad()
            targets = self.preprocess_target(targets)
            outputs, targets = self.net.forward(
                    inputs.to(self.device), targets.to(self.device))
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()

            eta = ((time.time() - start) * total_size /
                   ((i + 1) * batch_size))
            print("\r{}/{} ETA {:.1f}s loss:{:.6f}".format(
                (i + 1) * batch_size,
                total_size,
                eta,
                running_loss / (i + 1)),
                end="")
            if self.debug:
                break
        train_loss = running_loss / (i + 1)
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        return train_loss

    def test_(self, epoch, best_score, train_start):
        self.net = self.net.eval()
        val_loss = 0.0
        score = 0.0
        for data in self.testloader:
            inputs, targets = data
            with torch.no_grad():
                outputs, _ = self.net.forward(inputs.to(self.device), None)
                loss = self.criterion(
                        outputs, self.preprocess_target(
                            targets).to(self.device))
                val_loss += loss.item()
                score += self.score(outputs, targets.to(self.device))
            if self.debug:
                break
        val_loss /= len(self.testloader)
        score /= len(self.testloader)

        self.writer.add_scalar('Loss/test', val_loss, epoch)
        self.writer.add_scalar('{}/test'.format(self.score.__name__),
                               score, epoch)
        if best_score < score and epoch > 20:
            self.save(score, epoch)
            best_score = score

        end_time = ((datetime.now(self.locale) - train_start) *
                    (self.epoch_size - self.start_epoch) /
                    (epoch + 1 - self.start_epoch)) + train_start
        print("\nval loss:{:.6f} score:{:.2f} end time:{}".format(
            val_loss, score, end_time))
        return best_score, score

    def optimizer_to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def train(self, epoch_size, start_epoch=0, debug=False, best_score=0):
        batch_size = self.trainloader.batch_size
        total_size = len(self.trainloader) * batch_size
        train_start = datetime.now(self.locale)
        best_score = best_score
        self.net = self.net.to(self.device)
        self.optimizer_to(self.device)
        self.start_epoch = start_epoch
        self.epoch_size = epoch_size

        for epoch in range(start_epoch, epoch_size, 1):
            print("Epoch: {}".format(epoch))
            train_loss = self.train_(epoch)
            best_score, score = self.test_(epoch, best_score, train_start)
            self.scheduler.step()

            end_flag = False
            for c in self.callbacks:
                if(c(train_loss)):
                    end_flag = True
            if end_flag:
                break
            if debug:
                break

        print('Finished Training')
        return best_score


if __name__ == '__main__':
    e = EarlyStopping()
    vals = [1.11] * 20
    vals[0] = 1.0
    vals[6] = 0.9
    print(vals)
    for val in vals:
        print(e(val))
