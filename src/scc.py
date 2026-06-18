import torch
from torch import nn
from torch.nn import functional as F


def channel_shuffle(x, groups: int):
    batchsize, n, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, n, groups, channels_per_group)
    x = x.permute(0, 1, 3, 2).contiguous()
    x = x.view(batchsize, n, -1)
    return x


@torch.no_grad()
def _fuse_bn2d_into_linear_pre(bn: nn.BatchNorm2d, linear: nn.Linear) -> nn.Linear:
    if linear.bias is None:
        bias = torch.zeros(linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype)
    else:
        bias = linear.bias.detach().clone()

    w = linear.weight.detach().clone()
    w_orig = w.clone()

    invstd = torch.rsqrt(bn.running_var.detach() + bn.eps)
    a = bn.weight.detach() * invstd
    d = bn.bias.detach() - bn.running_mean.detach() * a

    w = w * a.unsqueeze(0)
    bias = bias + w_orig @ d

    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight.data.copy_(w)
    fused.bias.data.copy_(bias)
    return fused


@torch.no_grad()
def _fuse_bn2d_into_linear_post(linear: nn.Linear, bn: nn.BatchNorm2d) -> nn.Linear:
    if linear.bias is None:
        bias = torch.zeros(linear.out_features, device=linear.weight.device, dtype=linear.weight.dtype)
    else:
        bias = linear.bias.detach().clone()

    w = linear.weight.detach().clone()

    invstd = torch.rsqrt(bn.running_var.detach() + bn.eps)
    a = bn.weight.detach() * invstd
    d = bn.bias.detach() - bn.running_mean.detach() * a

    w = w * a.unsqueeze(1)
    bias = bias * a + d

    fused = nn.Linear(linear.in_features, linear.out_features, bias=True)
    fused.weight.data.copy_(w)
    fused.bias.data.copy_(bias)
    return fused


@torch.no_grad()
def _fuse_bn2d_into_conv2d(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype)
    else:
        bias = conv.bias.detach().clone()

    w = conv.weight.detach().clone()

    invstd = torch.rsqrt(bn.running_var.detach() + bn.eps)
    a = bn.weight.detach() * invstd
    d = bn.bias.detach() - bn.running_mean.detach() * a

    w = w * a.reshape(-1, 1, 1, 1)
    bias = bias * a + d

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )
    fused.weight.data.copy_(w)
    fused.bias.data.copy_(bias)
    return fused


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2, attn_impl: str = "auto", linear_threshold: int = 2048):
        super(Cross_Atten_Lite_split, self).__init__()
        self.inc1 = inc1
        self.inc2 = inc2
        self.midc1 = inc1 // 4
        self.midc2 = inc2 // 4
        self.need_pad = self.midc1 < self.midc2
        self.pad_size = self.midc2 - self.midc1 if self.need_pad else 0

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout_p = 0.2
        self.scale = self.midc2 ** -0.5
        self._init_weight()

        self.deploy = False
        self.attn_impl = attn_impl
        self.linear_threshold = int(linear_threshold)

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.matmul(k.transpose(1, 2), v)
        k_sum = k.sum(dim=1)
        denom = torch.matmul(q, k_sum.unsqueeze(-1)).squeeze(-1)
        denom = denom + 1e-6
        out = torch.matmul(q, kv)
        out = out / denom.unsqueeze(-1)
        return out

    def forward(self, x, x1, x2):
        b, _, h, w = x.shape
        n = h * w
        x1_norm = self.bn_x1(x1).flatten(2).transpose(1, 2)
        x2_norm = self.bn_x2(x2).flatten(2).transpose(1, 2)

        if hasattr(self, "kq_fused") and self.kq_fused is not None:
            kq = self.kq_fused(torch.cat([x1_norm, x2_norm], dim=2))
        else:
            kq1 = self.kq1(x1_norm)
            kq2 = self.kq2(x2_norm)
            kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = kq.split(self.midc2, dim=2)

        v = self.v_conv(x.flatten(2).transpose(1, 2))
        v1, v2 = v.split(self.midc1, dim=2)

        if self.need_pad:
            v1 = F.pad(v1, (0, self.pad_size))
            v2 = F.pad(v2, (0, self.pad_size))

        use_linear = False
        if self.attn_impl == "linear":
            use_linear = True
        elif self.attn_impl == "sdpa":
            use_linear = False
        else:
            use_linear = n > self.linear_threshold

        if use_linear:
            v1 = self._linear_attention(q1, k1, v1)
            v2 = self._linear_attention(q2, k2, v2)
        else:
            q1 = q1.unsqueeze(1)
            k1 = k1.unsqueeze(1)
            v1 = v1.unsqueeze(1)
            q2 = q2.unsqueeze(1)
            k2 = k2.unsqueeze(1)
            v2 = v2.unsqueeze(1)

            dropout_p = self.dropout_p if self.training else 0.0
            v1 = F.scaled_dot_product_attention(q1, k1, v1, dropout_p=dropout_p, scale=self.scale).squeeze(1)
            v2 = F.scaled_dot_product_attention(q2, k2, v2, dropout_p=dropout_p, scale=self.scale).squeeze(1)

        if self.need_pad:
            v1 = v1[:, :, : self.midc1]
            v2 = v2[:, :, : self.midc1]

        v = torch.cat([v1, v2], dim=2)
        v = self.out_conv(v).transpose(1, 2).reshape(b, -1, h, w)
        v = self.bn_last(v) + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        self.kq1 = _fuse_bn2d_into_linear_pre(self.bn_x1, self.kq1).to(self.kq1.weight.device)
        self.kq2 = _fuse_bn2d_into_linear_pre(self.bn_x2, self.kq2).to(self.kq2.weight.device)
        self.bn_x1 = nn.Identity()
        self.bn_x2 = nn.Identity()

        mid = self.midc2
        out_dim = 4 * mid
        w = torch.zeros(out_dim, self.inc1 + self.inc2, device=self.kq1.weight.device, dtype=self.kq1.weight.dtype)
        b = torch.zeros(out_dim, device=self.kq1.weight.device, dtype=self.kq1.weight.dtype)

        for i in range(2 * mid):
            w[2 * i, : self.inc1] = self.kq1.weight.data[i]
            b[2 * i] = self.kq1.bias.data[i]
            w[2 * i + 1, self.inc1 :] = self.kq2.weight.data[i]
            b[2 * i + 1] = self.kq2.bias.data[i]

        self.kq_fused = nn.Linear(self.inc1 + self.inc2, out_dim, bias=True)
        self.kq_fused.weight.data.copy_(w)
        self.kq_fused.bias.data.copy_(b)
        self.kq_fused = self.kq_fused.to(self.kq1.weight.device)
        self.kq1 = None
        self.kq2 = None

        self.out_conv = _fuse_bn2d_into_linear_post(self.out_conv, self.bn_last).to(self.out_conv.weight.device)
        self.bn_last = nn.Identity()

        self.deploy = True


class SpatialAttention_max(nn.Module):
    def __init__(self, in_channels, reduction1=16, reduction2=8):
        super(SpatialAttention_max, self).__init__()
        self.in_channels = in_channels
        self.scale_factor = 1.0 / (in_channels * in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc_spatial = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction1, in_channels, bias=False),
        )

        self.fc_channel = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction2, in_channels, bias=False),
        )

        self._init_weight()

    def forward(self, x):
        b, c, h, w = x.size()
        y_avg = self.avg_pool(x).squeeze(-1).squeeze(-1)

        y_spatial = self.fc_spatial(y_avg).unsqueeze(-1).unsqueeze(-1)
        y_channel = self.fc_channel(y_avg).unsqueeze(-1).unsqueeze(-1).sigmoid()

        spatial_weighted = x * y_spatial
        m = (spatial_weighted.sum(dim=1, keepdim=True) * self.scale_factor).sigmoid()

        return m * x * y_channel

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


class SCC_Module(nn.Module):
    def __init__(self, inc_depth2, inc_rgb, attn_impl: str = "auto", linear_threshold: int = 2048):
        super(SCC_Module, self).__init__()
        channel = inc_rgb + inc_depth2

        self.fus_atten = SpatialAttention_max(in_channels=channel)
        self.conv_bn = nn.Sequential(
            nn.Conv2d(channel, inc_depth2, kernel_size=1, bias=False),
            nn.BatchNorm2d(inc_depth2),
        )

        self.cross_atten = Cross_Atten_Lite_split(
            inc_depth2,
            inc_rgb,
            attn_impl=attn_impl,
            linear_threshold=linear_threshold,
        )

        self.deploy = False

    def forward(self, depth_out, rgb_out):
        fus_s = self.fus_atten(torch.cat([depth_out, rgb_out], dim=1))
        fus_s = self.conv_bn(fus_s)
        fus_s = self.cross_atten(fus_s, depth_out, rgb_out)

        return fus_s

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return

        if isinstance(self.conv_bn, nn.Sequential) and len(self.conv_bn) == 2:
            conv, bn = self.conv_bn[0], self.conv_bn[1]
            fused = _fuse_bn2d_into_conv2d(conv, bn).to(conv.weight.device)
            self.conv_bn = fused

        self.cross_atten.switch_to_deploy()
        self.deploy = True
