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
                )

    def forward(self, x, _):
        x = self.net(x)
        return x, _


class FeedForwardRBG(nn.Module):
    def __init__(self, dim, mlpdim=3072, dropout=0.1,
                 rate=0.1, epsilon=0.4):
        super().__init__()
        self.gen1 = RandomBatchGeneralization(rate=rate, epsilon=epsilon)
        self.linear1 = nn.Linear(dim, mlpdim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlpdim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, y):
        x, y = self.gen1(x, y)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x, y


class FeedForwardBG(nn.Module):
    def __init__(self, dim, mlpdim=3072, dropout=0.1,
                 rate=0.1, epsilon=0.4):
        super().__init__()
        self.gen1 = BatchGeneralization(rate=rate, epsilon=epsilon)
        self.linear1 = nn.Linear(dim, mlpdim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlpdim, dim)

    def forward(self, x, y):
        x = self.gen1(x, y)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x, y


class FeedForwardFactory():
    def __init__(self, method, rate, epsilon):
        self.method = method
        self.rate = rate
        self.epsilon = epsilon

    def __call__(self, dim, mlpdim=3072, dropout=0.1):
        return self.method(dim, rate=self.rate, epsilon=self.epsilon)


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
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, inputs, labels):
        x = F.layer_norm(inputs, inputs.shape[2:])
        x = self.attention(x, x)
        x = self.dropout1(x)
        x = x + inputs

        y = F.layer_norm(x, x.shape[2:])
        y, labels = self.ff(y, labels)
        y = self.dropout2(y)
        return x + y, labels


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


class Combine(nn.Module):
    def __init__(self, net, generalization):
        super().__init__()
        self.net = net
        self.generalization = generalization

    def forward(self, x, y):
        x = self.net(x)
        x, y = self.generalization(x, y)
        return x, y


class CombineThrough(nn.Module):
    def __init__(self, net, generalization):
        super().__init__()
        self.net = net
        self.generalization = generalization

    def forward(self, x, y):
        x = self.net(x)
        x = self.generalization(x, y)
        return x, y


class EmbeddingFactory():
    def __init__(self, net, generalization, rate, epsilon,
                 change_output=False):
        self.net = net
        self.generalization = generalization
        self.rate = rate
        self.epsilon = epsilon
        self.change_output = change_output

    def __call__(self, in_channels, out_channels, kernel_size, stride):
        net = self.net(in_channels, out_channels, kernel_size, stride)
        generalization = self.generalization(self.rate, self.epsilon)
        if self.change_output:
            return Combine(net, generalization)
        else:
            return CombineThrough(net, generalization)


class Embedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.net = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride)

    def forward(self, x):
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
        x, y = self.embedding(x, y)
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)

        # add cls token to head
        x = torch.cat([torch.stack([self.cls for _ in range(b)]), x], axis=-1)

        x, y = self.encoder(x, y)

        # get token results
        x = x[:, :, 0]

        x = self.classifier(x)

        return x, y


if __name__ == '__main__':
    x = torch.empty(10, 32 * 32, 3)
    y = torch.empty(10, 16 * 16, 3)
    a = AttentionBase(3, 7)
    print(a(x, y).shape)

    a = SelfAttention(3, 7)
    print(a(x).shape)

    a = SourceTargetAttention(3, 7)
    print(a(x, y).shape)
