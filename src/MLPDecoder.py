from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1, bias=True)

    def forward(self, x):
        return self.proj(x)


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        ############## MLP decoder on C1-C4 (Optimized) ###########
        n = c1.shape[0]
        target_h, target_w = c1.shape[2], c1.shape[3]
        target_size = (target_h, target_w)

        _c1 = self.linear_c1(c1)

        _c2 = self.linear_c2(c2)
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3)
        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=self.align_corners)

        _c4 = self.linear_c4(c4)
        _c4 = F.interpolate(_c4, size=target_size, mode='bilinear', align_corners=self.align_corners)

        # Fuse and predict
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dropout is not None:
            _c = self.dropout(_c)
        x = self.linear_pred(_c)

        return x

    def load_state_dict(self, state_dict, strict=True):
        converted = OrderedDict()
        own = self.state_dict()
        for k, v in state_dict.items():
            nk = k
            if nk.endswith('.proj.weight') and nk in own and own[nk].dim() == 4 and v.dim() == 2:
                v = v.unsqueeze(-1).unsqueeze(-1)
            converted[nk] = v
        return super().load_state_dict(converted, strict=False)

    @torch.no_grad()
    def switch_to_deploy(self):
        if isinstance(self.linear_fuse, nn.Sequential) and len(self.linear_fuse) >= 2:
            conv = self.linear_fuse[0]
            bn = self.linear_fuse[1]
            if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
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
                fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
                tail = list(self.linear_fuse.children())[2:]
                self.linear_fuse = nn.Sequential(fused, *tail)

        
