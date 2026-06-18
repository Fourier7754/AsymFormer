import math
from dataclasses import dataclass

import torch
from torch import nn


def _make_identity_kernel(out_channels: int, in_channels: int, groups: int, kernel_size: int, device, dtype):
    assert out_channels == in_channels
    assert in_channels % groups == 0
    in_channels_per_group = in_channels // groups
    weight = torch.zeros(
        (out_channels, in_channels_per_group, kernel_size, kernel_size), device=device, dtype=dtype
    )
    center = kernel_size // 2
    for oc in range(out_channels):
        ic = oc % in_channels_per_group
        weight[oc, ic, center, center] = 1
    return weight


def _pad_kernel_to_square(kernel: torch.Tensor, target_kernel_size: int):
    if kernel is None:
        return None
    k_h, k_w = kernel.shape[-2:]
    if k_h == target_kernel_size and k_w == target_kernel_size:
        return kernel
    pad_h = target_kernel_size - k_h
    pad_w = target_kernel_size - k_w
    assert pad_h >= 0 and pad_w >= 0
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return torch.nn.functional.pad(kernel, [pad_left, pad_right, pad_top, pad_bottom])


@dataclass
class RepAsymConfig:
    use_kxk: bool = True
    use_1xk: bool = True
    use_kx1: bool = True
    use_identity: bool = True


class RepAsymConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        groups: int = 1,
        deploy: bool = False,
        config: RepAsymConfig | None = None,
    ):
        super().__init__()
        assert isinstance(kernel_size, int)
        assert kernel_size % 2 == 1, "Only odd kernel_size supported for exact fusion"
        if padding is None:
            padding = kernel_size // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deploy = deploy
        self.config = config or RepAsymConfig()

        self._deploy_cache = None
        self._deploy_cache_valid = False

        if deploy:
            self.reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            return

        self.branch_kxk = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            if self.config.use_kxk
            else None
        )

        self.branch_1xk = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(1, kernel_size),
                stride=stride,
                padding=(0, padding),
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            if self.config.use_1xk
            else None
        )

        self.branch_kx1 = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(kernel_size, 1),
                stride=stride,
                padding=(padding, 0),
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            if self.config.use_kx1
            else None
        )

        self.use_identity = (
            self.config.use_identity
            and out_channels == in_channels
            and stride == 1
            and dilation == 1
        )

        if self.branch_1xk is not None:
            nn.init.zeros_(self.branch_1xk.weight)
            nn.init.zeros_(self.branch_1xk.bias)
        if self.branch_kx1 is not None:
            nn.init.zeros_(self.branch_kx1.weight)
            nn.init.zeros_(self.branch_kx1.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._deploy_cache = None
            self._deploy_cache_valid = False
        return self

    @torch.no_grad()
    def _get_or_build_deploy_cache(self):
        if self._deploy_cache_valid and self._deploy_cache is not None:
            return self._deploy_cache
        kernel, bias = self.get_equivalent_kernel_bias()
        conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        conv.weight.data.copy_(kernel)
        conv.bias.data.copy_(bias)
        conv.to(device=kernel.device, dtype=kernel.dtype)
        self._deploy_cache = conv
        self._deploy_cache_valid = True
        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.reparam(x)

        if (not self.training) and (not torch.is_grad_enabled()):
            conv = self._get_or_build_deploy_cache()
            return conv(x)

        out = 0
        if self.branch_kxk is not None:
            out = out + self.branch_kxk(x)
        if self.branch_1xk is not None:
            out = out + self.branch_1xk(x)
        if self.branch_kx1 is not None:
            out = out + self.branch_kx1(x)
        if self.use_identity:
            out = out + x
        return out

    @torch.no_grad()
    def get_equivalent_kernel_bias(self):
        device = None
        dtype = None
        for m in [self.branch_kxk, self.branch_1xk, self.branch_kx1]:
            if m is not None:
                device = m.weight.device
                dtype = m.weight.dtype
                break
        if device is None:
            device = torch.device("cpu")
            dtype = torch.float32

        eq_kernel = torch.zeros(
            (self.out_channels, self.in_channels // self.groups, self.kernel_size, self.kernel_size),
            device=device,
            dtype=dtype,
        )
        eq_bias = torch.zeros((self.out_channels,), device=device, dtype=dtype)

        if self.branch_kxk is not None:
            eq_kernel = eq_kernel + self.branch_kxk.weight
            eq_bias = eq_bias + self.branch_kxk.bias

        if self.branch_1xk is not None:
            eq_kernel = eq_kernel + _pad_kernel_to_square(self.branch_1xk.weight, self.kernel_size)
            eq_bias = eq_bias + self.branch_1xk.bias

        if self.branch_kx1 is not None:
            eq_kernel = eq_kernel + _pad_kernel_to_square(self.branch_kx1.weight, self.kernel_size)
            eq_bias = eq_bias + self.branch_kx1.bias

        if self.use_identity:
            id_kernel = _make_identity_kernel(
                out_channels=self.out_channels,
                in_channels=self.in_channels,
                groups=self.groups,
                kernel_size=self.kernel_size,
                device=device,
                dtype=dtype,
            )
            eq_kernel = eq_kernel + id_kernel

        return eq_kernel, eq_bias

    @torch.no_grad()
    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam.to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)

        for p in self.parameters():
            p.detach_()

        for name in ["branch_kxk", "branch_1xk", "branch_kx1"]:
            if hasattr(self, name):
                self.__delattr__(name)
        self.deploy = True


class RepAsymDWConv2d(RepAsymConv2d):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        deploy: bool = False,
        config: RepAsymConfig | None = None,
    ):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            deploy=deploy,
            config=config,
        )


@torch.no_grad()
def switch_to_deploy(model: nn.Module):
    for m in model.modules():
        if m is model:
            continue
        if hasattr(m, "switch_to_deploy") and callable(getattr(m, "switch_to_deploy")):
            m.switch_to_deploy()
