from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ModelEmaParamsOnly(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.9999, device=None):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = float(decay)
        self.device = device
        self._param_keys = set(dict(self.module.named_parameters()).keys())
        if self.device is not None:
            self.module.to(device=self.device)

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay

        msd = model.state_dict()
        esd = self.module.state_dict()

        for k, ema_v in esd.items():
            model_v = msd[k]
            if self.device is not None:
                model_v = model_v.to(device=self.device)

            if torch.is_floating_point(ema_v) and torch.is_floating_point(model_v) and k in self._param_keys:
                ema_v.copy_(ema_v * d + (1.0 - d) * model_v)
            else:
                ema_v.copy_(model_v)
