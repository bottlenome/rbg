import torch
import torch.nn as nn
import torch.nn.functional as F
from .function import onehot, onehot_cross


class random_batch_generalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, rate, epsilon):
        batch_size = x.shape[0]
        ref_index = torch.randint(low=0, high=batch_size - 1,
                                  size=(int(batch_size * rate), ))
        target_index = torch.randint(low=0, high=batch_size - 1,
                                     size=(int(batch_size * rate), ))
        mag = torch.empty(len(ref_index)).normal_(mean=0.0, std=epsilon)
        ctx.save_for_backward(x, ref_index, target_index, mag)
        ret = x.clone()
        ret_y = y.clone()
        for i in range(len(ref_index)):
            ret[ref_index[i]] = (x[target_index[i]] * mag[i]
                                 + x[ref_index[i]] * (1 - mag[i]))

            total = (mag[i].abs() + (1 - mag[i]).abs())
            target_p = mag[i].abs() / total
            ref_p = (1 - mag[i]).abs() / total
            ret_y[target_index[i]] += y[ref_index[i]] * target_p
            ret_y[ref_index[i]] = y[ref_index[i]] * ref_p
        return ret, ret_y

    @staticmethod
    def backward(ctx, grad_output, _):
        x, ref_index, target_index, mag = ctx.saved_tensors
        grad_input = grad_output.clone()
        for i in range(len(ref_index)):
            ref = grad_input[ref_index[i]]
            # dL/da = dL/dy * dy/da
            grad_input[ref_index[i]] = ref * (1 - mag[i])
            # dL/db = dL/dy * dy/db
            grad_input[target_index[i]] += ref * mag[i]
        return grad_input, None, None, None


class RandomBatchGeneralization(nn.Module):
    def __init__(self, rate=0.1, epsilon=0.4):
        super().__init__()
        self.epsilon = epsilon
        self.rate = rate
        self.forward_ = random_batch_generalization.apply

    def forward(self, x, y):
        if self.training:
            return self.forward_(x, y, self.rate, self.epsilon)
        else:
            return x, y


class batch_generalization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, rate, epsilon):
        batch_size = x.shape[0]
        ref_index = torch.randint(low=0, high=batch_size - 1,
                                  size=(int(batch_size * rate), ))
        target_index = torch.zeros(ref_index.shape, dtype=torch.int)
        for i in range(len(ref_index)):
            same_label = torch.where(y == y[ref_index[i]])[0]
            j = torch.randint(low=0, high=len(same_label), size=(1,))
            target_index[i] = same_label[j[0]]
        mag = torch.empty(len(ref_index)).normal_(mean=0.0, std=epsilon)
        ret = x.clone()
        for i in range(len(ref_index)):
            ret[ref_index[i]] = (x[target_index[i]] * mag[i]
                                 + x[ref_index[i]] * (1 - mag[i]))
        # ctx.save_for_backward(x, ref_index, target_index, mag)
        ctx.save_for_backward(ref_index, target_index, mag)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        # x, ref_index, target_index, mag = ctx.saved_tensors
        ref_index, target_index, mag = ctx.saved_tensors
        grad_input = grad_output.clone()
        for i in range(len(ref_index)):
            ref = grad_input[ref_index[i]]
            # dL/da = dL/dy * dy/da
            grad_input[ref_index[i]] = ref * (1 - mag[i])
            # dL/db = dL/dy * dy/db
            grad_input[target_index[i]] += ref * mag[i]
        return grad_input, None, None, None


class BatchGeneralization(nn.Module):
    def __init__(self, rate=0.1, epsilon=0.4):
        super().__init__()
        self.epsilon = epsilon
        self.rate = rate
        self.forward_ = batch_generalization.apply

    def forward(self, x, y):
        if self.training:
            return self.forward_(x, y, self.rate, self.epsilon)
        else:
            return x


if __name__ == '__main__':
    import time

    def profile(func, x, y):
        start = time.perf_counter()
        ret = func(x, y)
        end = time.perf_counter()
        print("{}, {} ms".format(str(func), (end - start) * 1000))
        return ret

    x = torch.rand((100, 3, 256, 256), requires_grad=True)
    y = torch.randint(low=0, high=9, size=(100,))

    r = RandomBatchGeneralization(rate=0.5)
    ret_x, ret_y = profile(r, x, onehot(y, 10))
    ret_x.sum().backward()
    print(ret_y[:10])
    print(ret_y.shape)
    # profile(r.cuda(), x.cuda())

    r = BatchGeneralization()
    ret = profile(r, x, y)
    ret.sum().backward()

    output = torch.rand((100, 10), requires_grad=True)
    loss = onehot_cross(output, onehot(y, 10))
    print(loss)
    loss.backward()
    loss = torch.nn.functional.cross_entropy(output, y)
    print(loss)
    loss = onehot_cross(output, ret_y)
    print(loss)
