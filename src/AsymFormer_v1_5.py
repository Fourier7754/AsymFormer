from torch.nn import functional as F
try:
    from src.mix_transformer_linear import OverlapPatchEmbed, mit_b0
    from src.convnext import asym_convnext_tiny as convnext_tiny
    from src.MLPDecoder import DecoderHead
    from src.reparam_layers import switch_to_deploy as _switch_to_deploy
except ImportError:
    # Fallback for when running from within src directory or different python path
    from mix_transformer_linear import OverlapPatchEmbed, mit_b0
    from convnext import asym_convnext_tiny as convnext_tiny
    from MLPDecoder import DecoderHead
    from reparam_layers import switch_to_deploy as _switch_to_deploy

import os
import torch
from torch import nn

try:
    from src.scc import SCC_Module
except ImportError:
    from scc import SCC_Module


class down_sample_block(nn.Module):
    def __init__(self, inc_depth, inc_rgb, block_num, rgb_stem, depth_stem, rgb_layer, depth_layer, depth_norm):
        super(down_sample_block, self).__init__()
        self.block_num = block_num

        if block_num != 0:
            self.depth_stem = depth_stem
            self.rgb_stem = rgb_stem
        else:
            self.depth_stem = OverlapPatchEmbed(in_chans=1, embed_dim=inc_depth)
            self.rgb_stem = rgb_stem

        self.rgb_layer = rgb_layer
        self.depth_layer = depth_layer

        self.depth_norm = depth_norm

        if self.block_num != 0:
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb)

    def forward(self, image, depth):
        B = image.shape[0]
        image = self.rgb_stem(image)
        rgb_out = self.rgb_layer(image)

        depth_out, H, W = self.depth_stem(depth)

        for blk in self.depth_layer:
            depth_out = blk(depth_out, H, W)
        depth_out = self.depth_norm(depth_out)
        depth_out = depth_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # SCC_Ablation
        if self.block_num != 0:
            merge = self.SCC(depth_out, rgb_out)
            return rgb_out, merge
        else:
            return rgb_out, depth_out


class AsymFormer_v1_5(nn.Module):
    def __init__(
        self,
        num_classes,
        rgb_use_asym_dwconv: bool = True,
        depth_use_asym_dwconv: bool = True,
        deploy: bool = False,
        pretrained_backbone: bool = True,
    ):
        super(AsymFormer_v1_5, self).__init__()

        model1 = convnext_tiny(
            pretrained=pretrained_backbone,
            drop_path_rate=0.3,
            use_asym_dwconv=rgb_use_asym_dwconv,
            deploy_dwconv=deploy,
        )
        ft1 = model1.stages
        stem = model1.downsample_layers
        stem1 = [stem[0], stem[1], stem[2], stem[3]]
        layers1 = [
            ft1[0],
            ft1[1],
            ft1[2],
            ft1[3]]

        model2 = mit_b0(use_asym_mlp_dwconv=depth_use_asym_dwconv, deploy_mlp_dwconv=deploy)
        layers2 = [model2.block1, model2.block2, model2.block3, model2.block4]
        stem2 = [model2.patch_embed1, model2.patch_embed2, model2.patch_embed3, model2.patch_embed4]
        norm2 = [model2.norm1, model2.norm2, model2.norm3, model2.norm4]

        self.channel = [32, 64, 160, 256]
        channel_list2 = [96, 192, 384, 768]

        self.down_sample_1 = down_sample_block(inc_depth=self.channel[0], inc_rgb=channel_list2[0], block_num=0,
                                               rgb_stem=stem1[0], depth_stem=None, rgb_layer=layers1[0], depth_layer=layers2[0], depth_norm=norm2[0])

        self.down_sample_2 = down_sample_block(inc_depth=self.channel[1],
                                               inc_rgb=channel_list2[1], block_num=1,
                                               rgb_stem=stem1[1], depth_stem=stem2[1], rgb_layer=layers1[1], depth_layer=layers2[1], depth_norm=norm2[1])

        self.down_sample_3 = down_sample_block(inc_depth=self.channel[2],
                                               inc_rgb=channel_list2[2], block_num=2,
                                               rgb_stem=stem1[2], depth_stem=stem2[2], rgb_layer=layers1[2], depth_layer=layers2[2], depth_norm=norm2[2])

        self.down_sample_4 = down_sample_block(inc_depth=self.channel[3],
                                               inc_rgb=channel_list2[3], block_num=3,
                                               rgb_stem=stem1[3], depth_stem=stem2[3], rgb_layer=layers1[3], depth_layer=layers2[3], depth_norm=norm2[3])

        self.Decoder = DecoderHead(in_channels=self.channel, num_classes=num_classes, dropout_ratio=0.1,
                                   norm_layer=nn.BatchNorm2d,
                                   embed_dim=256)

    def forward(self, image, depth):
        input_shape = image.shape[-2:]

        rgb_out, depth_out1 = self.down_sample_1(image, depth)
        rgb_out, depth_out2 = self.down_sample_2(rgb_out, depth_out1)

        rgb_out, depth_out3 = self.down_sample_3(rgb_out, depth_out2)
        _, depth_out = self.down_sample_4(rgb_out, depth_out3)

        rgb_out = self.Decoder(
            [depth_out1,
             depth_out2,
             depth_out3,
             depth_out])
        rgb_out = F.interpolate(rgb_out, size=input_shape, mode='bilinear', align_corners=False)
        return rgb_out

    def load_state_dict(self, state_dict, strict=True):
        new_state_dict = {}

        for key, value in state_dict.items():
            new_key = key

            if '.SCC.conv1.' in key:
                new_key = key.replace('.SCC.conv1.', '.SCC.conv_bn.0.')
            elif '.SCC.bn.' in key:
                new_key = key.replace('.SCC.bn.', '.SCC.conv_bn.1.')

            if new_key.endswith('.dwconv.weight'):
                cand = new_key.replace('.dwconv.weight', '.dwconv.branch_kxk.weight')
                if cand in self.state_dict():
                    new_key = cand
            elif new_key.endswith('.dwconv.bias'):
                cand = new_key.replace('.dwconv.bias', '.dwconv.branch_kxk.bias')
                if cand in self.state_dict():
                    new_key = cand

            if new_key.endswith('.dwconv.dwconv.weight'):
                cand = new_key.replace('.dwconv.dwconv.weight', '.dwconv.dwconv.branch_kxk.weight')
                if cand in self.state_dict():
                    new_key = cand
            elif new_key.endswith('.dwconv.dwconv.bias'):
                cand = new_key.replace('.dwconv.dwconv.bias', '.dwconv.dwconv.branch_kxk.bias')
                if cand in self.state_dict():
                    new_key = cand

            new_state_dict[new_key] = value

        return super(AsymFormer_v1_5, self).load_state_dict(new_state_dict, strict=False)

    @torch.no_grad()
    def switch_to_deploy(self):
        self.eval()
        _switch_to_deploy(self)
