import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import math
import os
from einops import rearrange
import numbers
from model.Sin_KAN import SPKAL
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from thop import profile

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            # 插眼：下面的CBR可以不要
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class Attention_org(nn.Module):

    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos
        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        if self.projection == 'interp' and (H != self.reduce_size or W != self.reduce_size):
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))
        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))
        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))
        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)
        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)
        out = self.to_out(out)
        out = self.proj_drop(out)

        return out, q_k_attn


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm3d(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm3d, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CViT(nn.Module):

    def __init__(self, in_ch, heads, attn_drop=0., proj_drop=0., reduce_size=16, projection='interp',
                 rel_pos=True):
        super().__init__()
        self.attn_norm1 = LayerNorm3d(in_ch, LayerNorm_type='WithBias')

        self.attn = Attention_org(in_ch, heads=heads, dim_head=in_ch // heads, attn_drop=attn_drop,
                                  proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                  rel_pos=rel_pos)

        self.attn_norm2 = LayerNorm3d(in_ch, LayerNorm_type='WithBias')
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        # Attention
        out = self.attn_norm1(x)
        out, q_k_attn = self.attn(out)
        out = out + x
        residue = out
        # MLP
        out = self.attn_norm2(out)
        out = self.relu(out)
        out = self.mlp(out)
        out += residue

        return out


class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias,
                                                                  H // self.h if H > self.h else 1, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h,
                                                                  W // self.w if W > self.w else 1, dim=1)  # HW, hw, nH

        if H % self.h != 0 or W % self.w != 0:
            relative_position_bias_expanded = relative_position_bias_expanded.permute(3, 2, 0, 1)
            relative_position_bias_expanded = F.interpolate(relative_position_bias_expanded, size=(H, W),
                                                            mode='bicubic')
            relative_position_bias_expanded = relative_position_bias_expanded.permute(2, 3, 1, 0)

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               self.num_heads).permute(2, 0,
                                                                                                       1).contiguous().unsqueeze(
            0)

        return relative_position_bias_expanded


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = SPKAL(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = SPKAL(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = SPKAL(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            # # TODO

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        # t = x.reshape(B * N, C)
        # Batch 和 Hw (N) 放在一起参与计算   C单独
        x = self.fc1(x.reshape(B * N, C))
        # x = self.fc1(x)
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        # # TODO
        # x = x.reshape(B,N,C).contiguous()
        # x = self.dwconv_4(x, H, W)

        return x


class PCM(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.SiLU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class SP_KAN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, mode='train', deepsuper=True,
                 embed_dims=[256], no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):
        super(SP_KAN, self).__init__()

        basic_width = 16
        filters = [basic_width, basic_width * 2, basic_width * 4, basic_width * 8, basic_width * 16]
        self.deepsuper = deepsuper
        self.mode = mode
        self.no_kan = no_kan
        print('Deep-Supervision:', deepsuper)

        self.maxpool = nn.MaxPool2d(2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.TransH1 = CViT(filters[0], 1)
        self.TransH2 = CViT(filters[1], 1)
        self.TransH3 = CViT(filters[2], 1)
        self.TransH4 = CViT(filters[3], 1)
        self.TransH5 = CViT(filters[4], 1)

        self.stem = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.block5 = nn.ModuleList([PCM(dim=embed_dims[0],
                                         drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=self.no_kan
                                         )])

        self.patch_embed5 = PatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[0])

        self.norm5 = norm_layer(embed_dims[0])
        self.dblock5 = nn.ModuleList([PCM(dim=embed_dims[0],
                                          drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer, no_kan=self.no_kan
                                          )])

        self.dnorm5 = norm_layer(embed_dims[0])
        self.br_conv5 = conv_block(embed_dims[0], embed_dims[0])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.tail = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        if deepsuper:
            self.gt_conv5 = nn.Sequential(nn.Conv2d(filters[4], 1, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(filters[3], 1, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(filters[2], 1, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(filters[1], 1, 1))
            self.outconv = nn.Conv2d(5 * 1, 1, 1)

    def forward(self, x):
        # x=torch.cat([x,x,x],dim=1)
        B = x.shape[0]

        e1 = self.stem(x)  # 1 16 256 256
        e1 = self.TransH1(e1)  # 1 16 256 256

        e2 = self.Conv2(self.maxpool(e1))  # 1 32 128 128
        e2 = self.TransH2(e2)  # 1 32 128 128

        e3 = self.Conv3(self.maxpool(e2))  # 1 64 64  64
        e3 = self.TransH3(e3)  # 1 64 64  64

        e4 = self.Conv4(self.maxpool(e3))  # 1 128 32 32
        e4 = self.TransH4(e4)

        e5 = self.Conv5(self.maxpool(e4))  # 1 256 16 16
        e5 = self.TransH5(e5)  # 1 256 16 16

        # *****************************************************
        #                          KAN
        # *****************************************************

        out, H, W = self.patch_embed5(e5)
        for i, blk in enumerate(self.block5):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = F.relu(F.interpolate(self.br_conv5(out), scale_factor=(2, 2), mode='bilinear'))

        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock5):
            out = blk(out, H, W)
        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        #  *********** **************

        d5 = self.Up5(out)  # 1 256 16 16
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)  # 1 128 32 32

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)  # 1 64 64 64
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.tail(d2)

        if self.deepsuper:
            gt_5 = self.gt_conv5(e5)
            gt_5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt_4 = self.gt_conv4(d5)
            gt_4 = F.interpolate(gt_4, scale_factor=8, mode='bilinear', align_corners=True)
            gt_3 = self.gt_conv3(d4)
            gt_3 = F.interpolate(gt_3, scale_factor=4, mode='bilinear', align_corners=True)
            gt_2 = self.gt_conv2(d3)
            gt_2 = F.interpolate(gt_2, scale_factor=2, mode='bilinear', align_corners=True)
            d0 = self.outconv(torch.cat((gt_2, gt_3, gt_4, gt_5, out), 1))
            if self.mode == 'train':
                return (
                    torch.sigmoid(gt_5), torch.sigmoid(gt_4), torch.sigmoid(gt_3), torch.sigmoid(gt_2),
                    torch.sigmoid(d0),
                    torch.sigmoid(out))
            else:
                return torch.sigmoid(out)
        else:
            return torch.sigmoid(out)


if __name__ == '__main__':
    DEVICE = torch.device("cuda")
    model = SP_KAN(1, 1, mode='train', deepsuper=True, no_kan=False)
    model = model.cuda()
    DATA = torch.randn(1, 1, 256, 256).to(DEVICE)
    output = model(DATA)
    print(output[0].shape)

    flops, params = profile(model, (DATA,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
