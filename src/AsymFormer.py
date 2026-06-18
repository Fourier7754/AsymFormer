from torch.nn import functional as F
try:
    from src.mix_transformer import OverlapPatchEmbed, mit_b0
    from src.convnext import asym_convnext_tiny as convnext_tiny
    from src.MLPDecoder import DecoderHead
except ImportError:
    # Fallback for when running from within src directory or different python path
    from mix_transformer import OverlapPatchEmbed, mit_b0
    from convnext import asym_convnext_tiny as convnext_tiny
    from MLPDecoder import DecoderHead

from thop import profile
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
            self.SCC = SCC_Module(inc_depth2=inc_depth, inc_rgb=inc_rgb, attn_impl="sdpa")

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


class B0_T(nn.Module):
    def __init__(self, num_classes, pretrained_backbone: bool = True):
        super(B0_T, self).__init__()

        model1 = convnext_tiny(pretrained=pretrained_backbone, drop_path_rate=0.3)
        ft1 = model1.stages
        stem = model1.downsample_layers
        stem1 = [stem[0], stem[1], stem[2], stem[3]]
        layers1 = [
            ft1[0],
            ft1[1],
            ft1[2],
            ft1[3]]

        model2 = mit_b0()
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

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for old checkpoint format.
        Converts old SCC module keys (conv1, bn) to new format (conv_bn.0, conv_bn.1).
        Handles both Linear (2D) <-> Conv2d (4D) weight conversions for proj layers.
        """
        new_state_dict = {}
        model_state = self.state_dict()
        
        for key, value in state_dict.items():
            new_key = key
            
            # Convert old SCC module keys to new format
            if '.SCC.conv1.' in key:
                new_key = key.replace('.SCC.conv1.', '.SCC.conv_bn.0.')
            elif '.SCC.bn.' in key:
                new_key = key.replace('.SCC.bn.', '.SCC.conv_bn.1.')
            
            # Handle Linear <-> Conv2d conversions
            if new_key in model_state:
                model_weight = model_state[new_key]
                # Convert 2D (Linear) to 4D (Conv2d) for Decoder MLP layers
                if new_key.endswith('.proj.weight') and value.dim() == 2 and model_weight.dim() == 4:
                    value = value.unsqueeze(-1).unsqueeze(-1)
                # Convert 4D (Conv2d) to 2D (Linear) for attention proj layers
                elif '.attn.proj.weight' in new_key and value.dim() == 4 and model_weight.dim() == 2:
                    value = value.squeeze(-1).squeeze(-1)
            
            new_state_dict[new_key] = value
        
        return super(B0_T, self).load_state_dict(new_state_dict, strict=strict)

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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-resolution', action='store_true', help='Test with increasing resolutions')
    args = parser.parse_args()

    if args.test_resolution:
        import time
        print("="*50)
        print("Testing Resolution Scalability (Original AsymFormer)...")
        print("="*50)
        
        if torch.cuda.is_available():
            device = 'cuda'
            torch.backends.cudnn.benchmark = True
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"Using device: {device}")
        
        resolutions = [
            (224, 224),
            (320, 320),
            (384, 384),
            (480, 480),
            (480, 640),
            (512, 512),
            (640, 640),
            (960, 1280),
            (1024,1024),
            (1024, 2048)
        ]

        for H, W in resolutions:
            print(f"\nTesting Resolution: {H}x{W} ...")
            try:
                # Use custom FlopsProfiler for consistent comparison
                import sys
                import os
                # Ensure we can import from utils
                # Add project root to sys.path to find 'src' and 'utils'
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                if project_root not in sys.path:
                    sys.path.append(project_root)
                
                # Import from src.utils.flops_counter because we are running from project root usually
                # But inside src/AsymFormer.py, we might need different path
                try:
                    from src.utils.flops_counter import FlopsProfilerV2
                except ImportError:
                    from utils.flops_counter import FlopsProfilerV2
                
                model = B0_T(num_classes=40)
                rgb_dummy = torch.randn(1, 3, H, W)
                depth_dummy = torch.randn(1, 1, H, W)
                
                profiler = FlopsProfilerV2(model)
                profiler.start_profile()
                _ = model(rgb_dummy, depth_dummy)
                profiler.stop_profile()
                
                macs = profiler.get_total_flops()
                params = profiler.get_total_params()
                
                print(f"    GFLOPs: {macs / 1e9:.2f}")
                print(f"    Params: {params / 1e6:.2f}M")
                del model
                del profiler
            except Exception as e:
                print(f"    Profile Error: {e}")

            # Speed Test
            try:
                model_speed = B0_T(num_classes=40)
                model_speed = model_speed.to(device)
                model_speed.eval()
                
                rgb_bench = torch.randn(1, 3, H, W).to(device)
                depth_bench = torch.randn(1, 1, H, W).to(device)
                
                # Warmup
                for _ in range(10):
                    _ = model_speed(rgb_bench, depth_bench)
                if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                elif device == 'cuda': torch.cuda.synchronize()
                
                # Bench Latency
                num_iters = 50
                start = time.time()
                with torch.no_grad():
                    for _ in range(num_iters):
                        _ = model_speed(rgb_bench, depth_bench)
                        if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                        elif device == 'cuda': torch.cuda.synchronize()
                end = time.time()
                
                avg_ms = (end - start) / num_iters * 1000
                fps = 1000 / avg_ms
                print(f"    Inference (FP32 Latency): {avg_ms:.2f} ms | FPS: {fps:.2f}")

                # Bench Throughput
                start = time.time()
                with torch.no_grad():
                    for _ in range(num_iters):
                        _ = model_speed(rgb_bench, depth_bench)
                if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                elif device == 'cuda': torch.cuda.synchronize()
                end = time.time()
                
                avg_ms_throughput = (end - start) / num_iters * 1000
                fps_throughput = 1000 / avg_ms_throughput
                print(f"    Inference (FP32 Throughput): {avg_ms_throughput:.2f} ms | FPS: {fps_throughput:.2f}")

                # Speed (FP16)
                try:
                    model_fp16 = model_speed.half()
                    rgb_bench_fp16 = rgb_bench.half()
                    depth_bench_fp16 = depth_bench.half()

                    # Warmup
                    for _ in range(10):
                        _ = model_fp16(rgb_bench_fp16, depth_bench_fp16)
                    if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                    elif device == 'cuda': torch.cuda.synchronize()
                    
                    # Bench Latency
                    start = time.time()
                    with torch.no_grad():
                        for _ in range(num_iters):
                            _ = model_fp16(rgb_bench_fp16, depth_bench_fp16)
                            if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                            elif device == 'cuda': torch.cuda.synchronize()
                    end = time.time()
                    
                    avg_ms_fp16 = (end - start) / num_iters * 1000
                    fps_fp16 = 1000 / avg_ms_fp16
                    print(f"    Inference (FP16 Latency): {avg_ms_fp16:.2f} ms | FPS: {fps_fp16:.2f}")

                    # Bench Throughput
                    start = time.time()
                    with torch.no_grad():
                        for _ in range(num_iters):
                            _ = model_fp16(rgb_bench_fp16, depth_bench_fp16)
                        if device == 'mps' and hasattr(torch.mps, 'synchronize'): torch.mps.synchronize()
                        elif device == 'cuda': torch.cuda.synchronize()
                    end = time.time()
                    
                    avg_ms_fp16_throughput = (end - start) / num_iters * 1000
                    fps_fp16_throughput = 1000 / avg_ms_fp16_throughput
                    print(f"    Inference (FP16 Throughput): {avg_ms_fp16_throughput:.2f} ms | FPS: {fps_fp16_throughput:.2f}")
                except Exception as e:
                    print(f"    FP16 Error: {e}")
                
                del model_speed
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    OOM Error: {e}")
                else:
                    print(f"    Runtime Error: {e}")
            except Exception as e:
                    print(f"    Error: {e}")


        print("="*50)
    else:
        # Original simple test
        model = B0_T(num_classes=40)
        model.eval()
        image = torch.rand(1, 3, 480, 640)
        depth = torch.rand(1, 1, 480, 640)
        
        # Use our custom profiler first for detailed output
        try:
            from utils.flops_counter import FlopsProfilerV2
            print("\nUsing Custom FlopsProfiler:")
            profiler = FlopsProfilerV2(model)
            profiler.start_profile()
            _ = model(image, depth)
            profiler.stop_profile()
            
            print(f"GFLOPs: {profiler.get_total_flops() / 1e9:.4f}")
            print(f"Params: {profiler.get_total_params() / 1e6:.4f}M")
            profiler.print_model_profile()
        except ImportError:
            print("Custom profiler not found, skipping detailed profile.")

        print("\nUsing thop:")
        macs, params = profile(model, inputs=(image, depth,))
        print(f"GFLOPs: {macs / 1e9:.4f}")
        print(f"Params: {params / 1e6:.4f}M")
