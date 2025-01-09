from timm.models.layers import DropPath
import numpy as np
from ecb import SeqConv3x3
from norms import *
from mmcv.cnn.bricks import ConvModule
import torch.nn.functional as F

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=9,
                 reduction_ratio=2,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):

        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, kernel_size=3, padding=1, stride=1, dilation=1):
            super().__init__()

            self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                                   stride=stride, dilation=dilation, groups=dim_in)
            self.norm_layer = nn.GroupNorm(4, dim_in)
            self.conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=(1, scale[i]), padding=(0, scale[i] // 2),groups=channels),
                                 nn.GELU(),
                                 nn.Conv2d(channels, channels, kernel_size=(scale[i], 1), padding=(scale[i] // 2, 0), groups=channels))
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class IRB(nn.Module):
    def __init__(self, in_features, hidden_features=None, reduction_ratio=2, out_features=None, ksize=3, act_layer=nn.Hardswish, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.multi_conv1 = MultiScaleDWConv(hidden_features)

        #self.local_unit = DynamicConv2d(dim=hidden_features, kernel_size=3, num_groups=2)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', hidden_features, hidden_features, -1)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', hidden_features, hidden_features, -1)

        inner_dim = max(16, hidden_features // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features),
            nn.GELU(),
           # nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, hidden_features, kernel_size=1),)
            #nn.BatchNorm2d(hidden_features), )
        self.conv = DepthWiseConv2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv(self.fc1(x))
        x = self.act(x)
        x1 = self.multi_conv1(x)
        x = self.proj(x1+self.conv1x1_lpl(x))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.reshape(B, C, -1).permute(0, 2, 1)

class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.activation = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u*attn

class PoolingAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1, 2, 3, 6,8]):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.LKA = AttentionModule(dim)
        self.conv5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.bn = BatchChannelNorm(dim)
        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        self.d_convs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim) for temp in pool_ratios])

    def forward(self, x, H, W):
        B, N, C = x.shape
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_1 = self.LKA(x_).reshape(B, C, -1).permute(0, 2, 1)
        # print(q_x.shape)

        q = self.q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        pools = []
        x_2 = self.conv3(x_)
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs):
            pool = F.adaptive_avg_pool2d(x_2, (round(H / pool_ratio), round(W / pool_ratio)))
            # print(f'x_:{x_.shape}; pool:{pool.shape}; pool_post:{l(pool).shape}')
            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # print(k.shape)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v)
        x = x.transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerscale_value=1e-4, pool_ratios=[1, 2, 3, 6]):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PoolingAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, pool_ratios=pool_ratios)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.mlp = FastLeFF(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.Hardswish, drop=drop,
                       ksize=3)
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        res = x
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        x = x + (res * self.gamma)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 64, 64).cuda()
    m = Block(64).cuda()
    o = m(x)
    print(o.shape)