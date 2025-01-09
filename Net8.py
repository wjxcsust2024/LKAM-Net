from norms import *
from timm.models.layers import DropPath, trunc_normal_, to_2tuple
import torch.nn.functional as F
from block1 import Block as p2tBlock

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class DWConv(nn.Module):
    def __init__(self,hidden_features):
        super(DWConv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features
    def forward(self,x):
        #x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x.shape[2], x.shape[3]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        #x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_in, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


class SLKA(nn.Module):
    def __init__(self, n_feats, k=21, d=3, shrink=0.5, scale=2):
        super().__init__()
        f = int(n_feats*shrink)
        self.head = nn.Conv2d(n_feats, f, 1)
        self.activation = nn.GELU()
        self.ca = ChannelAttention(f, 30)
        self.conv5 = nn.Conv2d(f, f, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, stride=1, padding=1, groups=f)
        self.conv5h = nn.Conv2d(f//2, f//2, kernel_size=5, stride=1, padding=2, groups=f//2)

        self.conv_spatial_7 = nn.Conv2d(f//2, f//2, 7, stride=1, padding=9, groups=f//2, dilation=3)


        self.bn2 = BatchChannelNorm(f)
        self.bn1 = BatchChannelNorm(f//2)
        self.tail = nn.Conv2d(f, n_feats, 1)
        self.scale = scale

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        c1 = self.head(x)

        attn = self.activation(c1)
        attn1 = self.conv3(attn)

        at_1, at_2 = torch.chunk(attn1, 2, dim=1)

        attn2 = self.activation(at_1)
        c3 = self.conv5h(attn2)

        c5 = self.activation(at_2)
        c5 = self.conv_spatial_7(c5)

        c6 = self.ca(self.conv5(attn1))

        c4 = torch.cat([c3, c5], dim=1)*F.gelu(c6)

        a_1 = self.tail(c4)
        a = F.sigmoid(a_1)
        return (x * a).reshape(B, C, -1).permute(0, 2, 1)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y1 = self.attention(x)
        y2 = self.attention(x)  #新增
        out = (x*y1)*y2    #改进
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class Embed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        # _, _, H, W = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return x


class Merge(nn.Module):   #缩小
    def __init__(self, dim, h, w):
        super(Merge, self).__init__()
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2, padding=0)
        self.h = h
        self.dim = dim
        self.w = w
        self.norm = nn.BatchNorm2d(dim*2)
        self.c_1 = nn.Sequential(nn.Conv2d(dim*2, dim*2, 3, 1, 1),
                                 nn.ReLU())

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
        x = self.norm(self.conv(x))
        x = self.c_1(x)
        return x.reshape(B, self.dim*2, -1).permute(0, 2, 1)

class Expand(nn.Module):  #放大
    def __init__(self, dim, h):
        super(Expand, self).__init__()
        self.dim = dim
        self.h = h
        self.conv = nn.ConvTranspose2d(self.dim, self.dim//2, 2, stride=2)
        self.c_1 = nn.Sequential(nn.Conv2d(dim // 2, dim // 2, 3, 1, 1),
                                 nn.ReLU())

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, self.h, self.h)
        x = self.conv(x)
        x = self.c_1(x)
        return x.reshape(B, self.dim//2, -1).permute(0, 2, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embed = Embed(512)

        self.l1 = nn.Sequential(p2tBlock(96, pool_ratios=[12,16,20,24]),
                                #p2tBlock(96, pool_ratios=[12,16,20,24]),
                                p2tBlock(96, pool_ratios=[12,16,20,24]))

        self.l2 = nn.Sequential(p2tBlock(192, pool_ratios=[6,8,10,12]),
                                p2tBlock(192, pool_ratios= [6,8,10,12]))

        self.l3 = nn.Sequential(p2tBlock(384, pool_ratios= [3,4,5,6]),
                                p2tBlock(384, pool_ratios= [3,4,5,6]))

        self.l4 = nn.Sequential(p2tBlock(768, pool_ratios= [1,2,3,4]),
                                p2tBlock(768, pool_ratios= [1,2,3,4]),
                                p2tBlock(768, pool_ratios= [1,2,3,4]),
                                p2tBlock(768, pool_ratios= [1,2,3,4]))

        self.m1 = Merge(96, 128, 128)
        self.m2 = Merge(192, 64, 64)
        self.m3 = Merge(384, 32, 32)

        self.p3 = Expand(768, 16)
        self.p2 = Expand(384, 32)
        self.p1 = Expand(192, 64)

        self.d3 = nn.Sequential(p2tBlock(384, pool_ratios= [3,4,5,6]),
                                #Block(384, 32, 4, 8, 3, 16),
                                p2tBlock(384, pool_ratios= [3,4,5,6]))

        self.d2 = nn.Sequential(p2tBlock(192, pool_ratios=[6,8,10,12]),
                                #Block(192, 64, 4, 8, 3, 16),
                                p2tBlock(192, pool_ratios=[6,8,10,12]))

        self.d1 = nn.Sequential(p2tBlock(96, pool_ratios=[12,16,20,24]),
                                #Block(96, 128, 4, 8, 3, 16),
                                p2tBlock(96, pool_ratios=[12,16,20,24]))

        self.dbm3 = SLKA(384)
        self.dbm2 = SLKA(192)
        self.dbm1 = SLKA(96)

        self.up = nn.PixelShuffle(4)
        self.seg = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embed(x)  # torch.Size([1, 16384, 96])
        x1 = self.l1(x)  # torch.Size([1, 16384, 96])
        #
        x = self.m1(x1)  # torch.Size([1, 4096, 192])
        x2 = self.l2(x)  # torch.Size([1, 4096, 192])
        #
        x = self.m2(x2)  # torch.Size([1, 1024, 384])
        x3 = self.l3(x)  # torch.Size([1, 1024, 384])
        #
        x = self.m3(x3)  # torch.Size([1, 256, 768])
        x4 = self.l4(x)  # torch.Size([1, 256, 768])
        #
        x = self.p3(x4)  # torch.Size([1, 1024, 384])
        x3_temp = self.dbm3(x3+x)
        x = self.d3(x3_temp)  # torch.Size([1, 1024, 384])
        #
        x = self.p2(x)  # torch.Size([1, 4096, 192])
        x2_temp = self.dbm2(x+x2)
        x = self.d2(x2_temp)
        #
        x = self.p1(x)  # torch.Size([1, 16384, 96])
        x1_temp = self.dbm1(x+x1)
        x = self.d1(x1_temp)  # 128x128

        x = self.up(x.permute(0, 2, 1).reshape(B, 96, 128, 128))  # torch.Size([1, 6, 512, 512])
        x = self.seg(x)
        return x


if __name__ == '__main__':
    x = torch.rand(2, 3, 512, 512).cuda()
    # y = torch.rand(1, 1024, 384).cuda()
    # dbm = DeBlock(384).cuda()
    part = Net().cuda()
    out = part(x)
    print(out.shape)
    # out = dbm(y)
    # print(out.shape)