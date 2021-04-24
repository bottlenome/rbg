import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import sys
import os
from .generalization import RandomBatchGeneralization, BatchGeneralization


class AttentionBase(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.output = nn.Linear(in_channels, out_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.depth_scale = None

    def forward(self, input, memory):
        query = self.query(input)
        # scaled dot product
        if self.depth_scale is None:
            depth = input.shape[2]
            self.depth_scale = depth ** -0.5
        query *= self.depth_scale
        key = self.key(memory)
        logit = torch.bmm(query, key.transpose(1, 2).contiguous())
        attention_weight = self.softmax(logit)
        selected_value = torch.bmm(attention_weight, self.value(memory))
        return self.output(selected_value)


class SelfAttention(AttentionBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        return super().forward(x, x)


class SourceTargetAttention(AttentionBase):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, input, memory):
        return super().forward(input, memory)


class Attention2D(SelfAttention):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

    def forward(self, x):
        width = x.shape[2]
        height = x.shape[3]
        tmp_shape = (x.shape[0], x.shape[1], width * height)
        x = x.reshape(tmp_shape)
        x = x.transpose(1, 2)
        x = super().forward(x)
        x = x.transpose(1, 2)
        x = x.reshape(x.shape[0], x.shape[1], width, height)
        return x


def calc_center(forward_func, padded, w, h, kernel_size):
    shift_w = kernel_size // 2
    shift_h = kernel_size // 2
    w = w + shift_w
    h = h + shift_h
    ret = forward_func(
            padded[:,
                   :,
                   h - shift_h: h + shift_h + 1,
                   w - shift_w: w + shift_w + 1])
    # get_center
    return ret[:, :, shift_h, shift_w]


class Attention2DLayer(Attention2D):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=(1, 1)):
        super().__init__(in_channels, out_channels)
        self.kernel_size = kernel_size
        # this is for interface crrespondence
        # self.padding_size = padding
        self.padding = nn.ZeroPad2d(kernel_size // 2)
        self.stride = stride
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = nn.functional.layer_norm(x, [x.shape[2], x.shape[3]])
        ret = torch.empty(x.shape[0], self.out_channels,
                          x.shape[2] // self.stride, x.shape[3] // self.stride)
        if x.is_cuda:
            ret = ret.cuda()
        width = ret.shape[2]
        height = ret.shape[3]
        padded = self.padding(x)
        for h in range(height):
            for w in range(0, width, self.stride):
                rw = w // self.stride
                rh = w // self.stride
                ret[:, :, rh, rw] = calc_center(
                        super().forward, padded, w, h, self.kernel_size)
        return self.dropout(ret)


def zero_padding_scale(x, stride):
    shape = (x.shape[0], x.shape[1], x.shape[2] * stride, x.shape[3] * stride)
    result = torch.empty(shape)
    if x.is_cuda:
        result = result.cuda()
    for h in range(x.shape[2]):
        for w in range(x.shape[3]):
            result[:, :, h * stride, w * stride] = x[:, :, h, w]
    return result


class TransAttention2DLayer(Attention2D):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=(1, 1)):
        super().__init__(in_channels, out_channels)
        # this is for interface crrespondence
        # self.padding_size = padding
        self.kernel_size = kernel_size
        self.padding = nn.ZeroPad2d(kernel_size // 2)
        self.scale = zero_padding_scale
        self.stride = stride
        self.out_channels = out_channels
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        x = nn.functional.layer_norm(x, [x.shape[2], x.shape[3]])
        up = self.scale(x, self.stride)
        width = up.shape[2]
        height = up.shape[3]
        padded = self.padding(up)
        ret = torch.empty(x.shape[0], self.out_channels, width, height)
        if x.is_cuda:
            ret = ret.cuda()
        for h in range(height):
            for w in range(width):
                ret[:, :, h, w] = calc_center(
                        super().forward, padded, w, h, self.kernel_size)
        return self.dropout(ret)


# think channel as feature
class ConvertToVector(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = nn.ZeroPad2d(kernel_size // 2)

    def forward(self, x):
        ret = torch.zeros(
                x.shape[0],
                x.shape[2],
                x.shape[3],
                x.shape[1],
                self.kernel_size,
                self.kernel_size)
        x = self.padding(x)
        for h in range(ret.shape[2]):
            for w in range(ret.shape[3]):
                ret[:, :, h, w, :] = x[:, :,
                                       h: h + self.kernel_size,
                                       w: w + self.kernel_size]
        return ret


class ConvFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size,
                          padding=kernel_size//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          padding=kernel_size//2),
                )

    def forward(self, x):
        return self.net(x)


class ConvAttention(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=(1, 1),
                 num_heads=1):
        super().__init__()
        self.kernel_size = (1, kernel_size, kernel_size)
        self.padding = (0, kernel_size // 2, kernel_size // 2)
        self.query = nn.Conv3d(in_channels, out_channels,
                               self.kernel_size,
                               stride=stride, padding=self.padding)
        self.key = nn.Conv3d(in_channels, out_channels,
                             self.kernel_size,
                             stride=stride, padding=self.padding)
        self.value = nn.Conv3d(in_channels, out_channels,
                               kernel_size,
                               stride=stride, padding=self.padding)
        self.num_heads = num_heads
        self.softmax = nn.Softmax(dim=-1)
        self.depth_scale = (kernel_size ** 2 // num_heads) ** -0.5

    def split(self, x, num_heads):
        return einops.rearrange(x, 'b l (n d) h w -> b n l d h w', n=num_heads)

    def combine(self, x):
        return einops.rearrange(x, 'b n l d h w -> b l (n d) h w')

    def forward(self, input, memory):
        query = self.query(input)
        key = self.key(memory)
        value = self.value(memory)

        query = self.split(query, self.num_heads)
        key = self.split(key, self.num_heads)
        value = self.split(value, self.num_heads)

        # scaled dot product
        query *= self.depth_scale
        logit = torch.einsum('bkndhw,bkmdhw->bknm', query, key)
        attention_weight = self.softmax(logit)
        selected_value = torch.einsum('bknm,bkmdhw->bkndhw',
                                      attention_weight, value)
        selected_value = self.combine(selected_value)
        return selected_value


class PositionEmbedding(nn.Module):
    def __init__(self, size, stddev=0.02):
        super().__init__()
        self.position_vector = nn.Parameter(torch.randn(size) * stddev)

    def forward(self, x):
        return x + self.position_vector


class FeedForward(nn.Module):
    def __init__(self, dim, mlpdim=3072, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, mlpdim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlpdim, dim),
                nn.Dropout(dropout),
                )

    def forward(self, x, _):
        x = self.net(x)
        return x, _


class FeedForwardRBG(nn.Module):
    def __init__(self, dim, mlpdim=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, mlpdim)
        self.gelu = nn.GELU()
        self.gen1 = RandomBatchGeneralization(rate=dropout)
        self.linear2 = nn.Linear(mlpdim, dim)
        self.gen2 = RandomBatchGeneralization(rate=dropout)

    def forward(self, x, y):
        x = self.linear1(x)
        x = self.gelu(x)
        x, y = self.gen1(x, y)
        x = self.linear2(x)
        x, y = self.gen2(x, y)
        return x, y


class FeedForwardBG(nn.Module):
    def __init__(self, dim, mlpdim=3072, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, mlpdim)
        self.gelu = nn.GELU()
        self.gen1 = BatchGeneralization(rate=dropout)
        self.linear2 = nn.Linear(mlpdim, dim)
        self.gen2 = BatchGeneralization(rate=dropout)

    def forward(self, x, y):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.gen1(x, y)
        x = self.linear2(x)
        x = self.gen2(x, y)
        return x, y


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)

        self.num_heads = num_heads
        self.depth_scale = (dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)

    def split(self, x, num_heads):
        return einops.rearrange(x, 'b l (h d) -> b h l d', h=num_heads)

    def combine(self, x):
        return einops.rearrange(x, 'b h l d -> b l (h d)')

    def forward(self, input, memory):
        query = self.query(input)
        key = self.key(memory)
        value = self.value(memory)

        query = self.split(query, self.num_heads)
        key = self.split(key, self.num_heads)
        value = self.split(value, self.num_heads)

        # scaled dot product
        query *= self.depth_scale
        logit = torch.einsum('bhnd,bhmd->bhnm', query, key)
        attention_weight = self.softmax(logit)
        selected_value = torch.einsum('bhnm,bhmd->bhnd',
                                      attention_weight, value)
        selected_value = self.combine(selected_value)
        return selected_value


class Encoder1D(nn.Module):
    def __init__(self, dim, num_heads,
                 attention=Attention, feed_forward=FeedForward):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.ff = feed_forward(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, labels):
        x = F.layer_norm(inputs, inputs.shape[2:])
        x = self.attention(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = F.layer_norm(x, x.shape[2:])
        y, labels = self.ff(y, labels)
        return x + y, labels


class Encoder1DRBG(nn.Module):
    def __init__(self, dim, num_heads,
                 attention=Attention, feed_forward=FeedForward):
        super().__init__()
        self.attention = Attention(dim, num_heads)
        self.ff = feed_forward(dim)
        self.dropout = RandomBatchGeneralization()

    def forward(self, inputs):
        x = F.layer_norm(inputs, inputs.shape[2:])
        x = self.attention(x, x)
        x = self.dropout(x)
        x = x + inputs

        y = F.layer_norm(x, x.shape[2:])
        y = self.ff(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, hidden_channels, length,
                 num_layers, num_heads,
                 position_embedding=PositionEmbedding,
                 attention=Attention,
                 feed_forward=FeedForward):
        super().__init__()
        size = (hidden_channels, length)
        self.position_embedding = position_embedding(size)
        self.dropout = nn.Dropout(0.1)
        self.encoders = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoders.append(
                    Encoder1D(hidden_channels, num_heads,
                              attention, feed_forward))

    def forward(self, x, y):
        x = self.position_embedding(x)
        x = self.dropout(x)
        x = einops.rearrange(x, 'b d n -> b n d')
        for encoder in self.encoders:
            x, y = encoder(x, y)
        x = F.layer_norm(x, x.shape[2:])
        x = einops.rearrange(x, 'b n d -> b d n')
        return x, y


class EncoderRBG(nn.Module):
    def __init__(self, hidden_channels, length,
                 num_layers, num_heads,
                 position_embedding=PositionEmbedding,
                 attention=Attention,
                 feed_forward=FeedForward):
        super().__init__()
        size = (hidden_channels, length)
        self.position_embedding = position_embedding(size)
        self.dropout = RandomBatchGeneralization()
        self.encoders = nn.ModuleList([])
        for _ in range(num_layers):
            self.encoders.append(
                    Encoder1DRBG(hidden_channels, num_heads,
                                 attention, feed_forward))

    def forward(self, x):
        x = self.position_embedding(x)
        x = self.dropout(x)
        x = einops.rearrange(x, 'b d n -> b n d')
        for encoder in self.encoders:
            x = encoder(x)
        x = F.layer_norm(x, x.shape[2:])
        x = einops.rearrange(x, 'b n d -> b d n')
        return x


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.net(x)


class Embedding2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=(1, kernel_size, kernel_size),
                    stride=stride,
                    padding=(0, kernel_size // 2, kernel_size // 2))

    def forward(self, x):
        x = einops.rearrange(x, 'b c (m h) (n w) -> b m n c h w', n=2, m=2)
        x = einops.rearrange(x, 'b m n c h w -> b c (m n) h w')
        return self.net(x)


class Embedding44(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 2,
                    kernel_size=(4, 4), stride=(4, 4)),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(4, 4), stride=(4, 4)),
                )

    def forward(self, x):
        return self.net(x)


class EmbeddingBnReLU44(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 2,
                    kernel_size=(4, 4), stride=(4, 4)),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(4, 4), stride=(4, 4)),
                )

    def forward(self, x):
        return self.net(x)


class Embedding422(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 4,
                    kernel_size=(4, 4), stride=(4, 4)),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(2, 2), stride=(2, 2)),
                )

    def forward(self, x):
        return self.net(x)


class Embedding242(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 4,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(4, 4), stride=(4, 4)),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(2, 2), stride=(2, 2)),
                )

    def forward(self, x):
        return self.net(x)


class Embedding224(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 4,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(4, 4), stride=(4, 4)),
                )

    def forward(self, x):
        return self.net(x)


class Embedding2222(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 8,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 8, out_channels // 4,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(2, 2), stride=(2, 2)),
                )

    def forward(self, x):
        return self.net(x)


class EmbeddingBnReLU2222(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 8,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(out_channels // 8),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 8, out_channels // 4,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(2, 2), stride=(2, 2)),
                )

    def forward(self, x):
        return self.net(x)


class EmbeddingBnReLU3333(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels // 8,
                    kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(out_channels // 8),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 8, out_channels // 4,
                    kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(out_channels // 8),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 4, out_channels // 2,
                    kernel_size=(3, 3), stride=(2, 2), padding=1),
                nn.BatchNorm2d(out_channels // 8),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels // 2, out_channels,
                    kernel_size=(3, 3), stride=(2, 2), padding=1),
                )

    def forward(self, x):
        return self.net(x)


class DoNothing(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 in_channels=3, out_channels=10,
                 num_layers=12, num_heads=12,
                 hidden_channels=768, patch_size=(16, 16),
                 embedding=Embedding,
                 position_embedding=PositionEmbedding,
                 attention=Attention,
                 feed_forward=FeedForward,
                 encoder=Encoder):
        super().__init__()
        self.embedding = embedding(
                in_channels, hidden_channels,
                kernel_size=patch_size, stride=patch_size)
        # 5 is defived from 32 x 32 patch 16 x 16 + 1
        length = 5
        self.encoder = encoder(hidden_channels, length, num_layers,
                               num_heads,
                               position_embedding=position_embedding,
                               attention=attention,
                               feed_forward=feed_forward)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        nn.init.zeros_(self.classifier.weight)
        self.cls = nn.Parameter(torch.zeros(hidden_channels, 1))

    def forward(self, x, y):
        x = self.embedding(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)

        # add cls token to head
        x = torch.cat([torch.stack([self.cls for _ in range(b)]), x], axis=-1)

        x, y = self.encoder(x, y)

        # get token results
        x = x[:, :, 0]

        x = self.classifier(x)

        return x, y


class ViTGen(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_layers=12, num_heads=12,
                 hidden_channels=768, patch_size=(4, 4),
                 embedding=Embedding,
                 position_embedding=PositionEmbedding,
                 attention=Attention,
                 feed_forward=FeedForward,
                 encoder=Encoder,
                 image_size=(32, 32)):
        super().__init__()
        length = int((image_size[0] / patch_size[0])
                     * (image_size[1] / patch_size[1]))
        self.embedding = embedding(
                in_channels, hidden_channels,
                kernel_size=patch_size, stride=patch_size)
        self.encoder = encoder(hidden_channels, length, num_layers,
                               num_heads,
                               position_embedding=position_embedding,
                               attention=attention,
                               feed_forward=feed_forward)
        self.reconst = nn.ConvTranspose2d(hidden_channels, in_channels,
                                          kernel_size=patch_size,
                                          stride=patch_size)

    def forward(self, x):
        x = self.embedding(x)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        x = self.encoder(x)
        x = x.reshape(b, c, h, w)
        x = self.reconst(x)
        return x


if __name__ == '__main__':
    x = torch.empty(10, 32 * 32, 3)
    y = torch.empty(10, 16 * 16, 3)
    a = AttentionBase(3, 7)
    print(a(x, y).shape)

    a = SelfAttention(3, 7)
    print(a(x).shape)

    a = SourceTargetAttention(3, 7)
    print(a(x, y).shape)

    a = Attention2D(3, 7)
    x = torch.empty(10, 3, 32, 32).cuda()
    print(a.cuda()(x).shape)

    a = Attention2DLayer(3, 7, stride=2)
    print(a.cuda()(x).shape)

    a = TransAttention2DLayer(3, 7, stride=2)
    print(a.cuda()(x).shape)

    x = torch.zeros(1, 1, 3, 3)
    for h in range(3):
        for w in range(3):
            x[0, 0, w, h] = w * 3 + h
    print(x)
    model = ConvertToVector(5)
    # (1, 1, 3, 3, 1, 5, 5)
    ret = model(x)
    print(ret.shape)
    print(ret)
    model = ConvertToVector(3)
    ret = model(x)

    x = torch.zeros(1, 5, 10, 32, 32)
    model = ConvAttention(5, 5, kernel_size=3, num_heads=2)
    # in_img : (b, depth, w, h)
    # in_img : (1, 9, 32, 32)
    # in : (b, kernel_size, depth)
    # in : (1, 3, 3 , 9)
    # out : (b, kernel_size, depth)
    # out : (1, 1, 9)
    # out_img : (b, depth, w, h)
    # out_img : (1, 9, 32, 32)
    print(model(x, x).shape)

    """
    a = torch.randn(10, 3, 32, 32, 3)
    b = torch.randn(10, 3, 32, 32, 4)
    ret = torch.zeros(10, 3, 4, 3)
    for i in range(3):
        a_r = a[:, i].reshape(10, 32 * 32, 3)
        b_r = b[:, i].reshape(10, 32 * 32, 4).transpose(1, 2).contiguous()
        ret[:, i] = torch.bmm(b_r, a_r)
    print(ret[0])
    ret = torch.einsum('bahwn,bahwm->banm', a, b)
    print(ret.shape)
    print(ret[0])

    model = VisionTransformer()
    a = torch.randn(10, 3, 32, 32)
    print(model(a).shape)
    """
    x = torch.zeros(10, 3, 32, 32)
    model = Embedding2d(3, 768, kernel_size=3, stride=1)
    print(model(x).shape)

    model = ViTGen()
    x = torch.zeros(10, 1, 32, 32)
    print(model(x).shape)

