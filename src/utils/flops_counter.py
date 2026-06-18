import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# Try importing torchvision for DeformConv2d support
try:
    import torchvision
    from torchvision.ops import DeformConv2d
    has_torchvision = True
except ImportError:
    has_torchvision = False
    DeformConv2d = None

def count_conv2d(m, x, y):
    # x is input tuple, y is output
    x = x[0]
    
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size(0)
    
    # Output map size
    out_h = y.size(2)
    out_w = y.size(3)
    
    # Standard MACs (Multiply-Accumulate) formula:
    # (Cin * K * K) * H_out * W_out * Cout / Groups
    
    kernel_ops = kh * kw * (cin // m.groups)
    total_ops = kernel_ops * cout * out_h * out_w * batch_size
    
    # Bias is usually ignored in MACs or counts as +1 op per output pixel. 
    # For consistency with major libraries (thop), we focus on MACs from weights.
    
    m.total_ops = torch.FloatTensor([total_ops])

def count_deform_conv2d(m, x, y):
    # DeformConv2d logic is similar to Conv2d in terms of MACs for the dot product part.
    # The offset/mask generation are usually separate Conv layers (handled separately).
    # The bilinear interpolation adds overhead but strictly speaking, the MACs for the 
    # convolution operation itself are the same as standard Conv2d.
    x = x[0]
    
    cin = m.in_channels
    cout = m.out_channels
    kh, kw = m.kernel_size
    batch_size = x.size(0)
    
    out_h = y.size(2)
    out_w = y.size(3)
    
    kernel_ops = kh * kw * (cin // m.groups)
    total_ops = kernel_ops * cout * out_h * out_w * batch_size
    
    m.total_ops = torch.FloatTensor([total_ops])

def count_linear(m, x, y):
    x = x[0]
    # x: [batch, ..., in_features]
    
    total_ops = m.in_features * m.out_features
    
    # Multiply by number of input vectors
    num_elements = x.numel() // x.size(-1)
    
    total_ops *= num_elements
    
    m.total_ops = torch.FloatTensor([total_ops])

def count_bn(m, x, y):
    x = x[0]
    nelements = x.numel()
    # BN inference: y = (x - mean) / std * gamma + beta
    # Simplified to: y = w * x + b
    # 1 MAC per element (1 mul + 1 add)
    
    total_ops = nelements
    m.total_ops = torch.FloatTensor([total_ops])

def count_ln(m, x, y):
    x = x[0]
    nelements = x.numel()
    # Similar to BN, approx 1 MAC per element for the affine transform.
    total_ops = nelements
    m.total_ops = torch.FloatTensor([total_ops])

def count_mhsa(m, x, y):
    # MultiheadAttention
    q = x[0]
    k = x[1]
    # Handle case where value is passed as kwarg (len(x) < 3)
    # We assume value has same shape as key if not provided in positional args
    if len(x) > 2:
        v = x[2]
    else:
        v = k
    
    embed_dim = m.embed_dim
    
    # Assuming q, k, v are (L, B, E) or (B, L, E)
    # We care about total tokens.
    
    # Extract Batch and SeqLen
    if m.batch_first:
        B, Lq, _ = q.shape
        _, Lk, _ = k.shape
    else:
        Lq, B, _ = q.shape
        Lk, B, _ = k.shape
        
    # 1. Linear Projections (Q, K, V)
    # Weights are (embed_dim, embed_dim) for each.
    # MACs = N * Din * Dout
    total_ops = 0
    total_ops += (B * Lq) * embed_dim * embed_dim # Q
    total_ops += (B * Lk) * embed_dim * embed_dim * 2 # K, V
    
    # 2. Attention Scores (Q * K^T) -> (B, heads, Lq, Lk)
    # Cost: B * Lq * Lk * embed_dim
    total_ops += B * Lq * Lk * embed_dim
    
    # 3. Weighted Sum (Attn * V) -> (B, heads, Lq, head_dim)
    # Cost: B * Lq * Lk * embed_dim
    total_ops += B * Lq * Lk * embed_dim
    
    # 4. Output Projection
    # Cost: (B * Lq) * embed_dim * embed_dim
    total_ops += (B * Lq) * embed_dim * embed_dim
    
    m.total_ops = torch.FloatTensor([total_ops])

def count_relu(m, x, y):
    # Element-wise
    x = x[0]
    total_ops = x.numel()
    m.total_ops = torch.FloatTensor([total_ops])

def count_avgpool(m, x, y):
    y = y
    total_ops = y.numel()
    m.total_ops = torch.FloatTensor([total_ops])

def count_adap_avgpool(m, x, y):
    y = y
    total_ops = y.numel()
    m.total_ops = torch.FloatTensor([total_ops])

def count_upsample(m, x, y):
    y = y
    # Bilinear approx 4 ops per element
    total_ops = y.numel() * 4
    m.total_ops = torch.FloatTensor([total_ops])

def count_mask2former_head(m, x, y):
    # This hook is specifically designed for the forward pass of Mask2FormerHead
    # It attempts to capture the einsum operation: 'bnc, bchw -> bnhw'
    
    # Inputs: x = (latents, mask_features, target_size)
    # latents: [B, N, C]
    # mask_features: [B, C, H, W]
    
    latents = x[0]
    mask_features = x[1]
    
    # Check shapes
    if len(latents.shape) == 3 and len(mask_features.shape) == 4:
        B, N, C = latents.shape
        _, _, H, W = mask_features.shape
        
        # Operation: torch.einsum('bnc, bchw -> bnhw', mask_embeddings, mask_features)
        # MACs = B * N * H * W * C
        # Note: In the actual forward pass, 'mask_embeddings' comes from latents -> MLP.
        # But the dimensions are the same (B, N, C).
        
        # This operation happens (num_layers + 1) times in the head (1 initial + num_layers loop)
        # We need to account for all of them.
        
        # Assuming m.extra_layers has (num_layers - 1) layers.
        # Plus 1 initial layer.
        # Total transformer layers = 1 + len(m.extra_layers) = num_layers
        # Total predictions = num_layers + 1 (1 after initial, 1 after each extra layer)
        
        num_preds = 1 + len(m.extra_layers)
        
        single_einsum_ops = B * N * H * W * C
        total_einsum_ops = single_einsum_ops * num_preds
        
        # Add to total ops
        # Note: We should be careful not to overwrite if other hooks already ran (though unlikely for a custom high-level module)
        # But since we are hooking the container 'Mask2FormerHead', its children (Conv, Linear) are counted separately via recursion.
        # This hook adds the *missing* einsum ops to the container itself.
        
        current_ops = m.total_ops.item() if hasattr(m, 'total_ops') else 0
        m.total_ops = torch.FloatTensor([current_ops + total_einsum_ops])
    else:
        pass

# Support for element-wise operations if they are nn.Modules
# (e.g., if user uses nn.Identity for residual addition wrapper, or custom modules)
def count_elementwise(m, x, y):
    x = x[0] if isinstance(x, tuple) else x
    total_ops = x.numel()
    m.total_ops = torch.FloatTensor([total_ops])

def count_grid_sample(m, x, y):
    # Grid sample involves bilinear interpolation
    # Output size: [B, C, H_out, W_out]
    # For each output pixel, we compute coordinates (small ops) and sample 4 pixels (bilinear)
    # Approx 8-10 ops per channel per pixel
    total_ops = y.numel() * 10
    m.total_ops = torch.FloatTensor([total_ops])

def count_shuffle(m, x, y):
    # Shuffle is just memory movement, usually 0 FLOPs
    m.total_ops = torch.zeros(1)

register_hooks = {
    nn.Conv2d: count_conv2d,
    nn.Linear: count_linear,
    nn.BatchNorm2d: count_bn,
    nn.LayerNorm: count_ln,
    nn.MultiheadAttention: count_mhsa,
    nn.ReLU: count_relu,
    nn.GELU: count_relu, # approx
    nn.SiLU: count_relu, # approx
    nn.Sigmoid: count_relu, # approx
    nn.AvgPool2d: count_avgpool,
    nn.AdaptiveAvgPool2d: count_adap_avgpool,
    nn.Upsample: count_upsample,
    nn.Dropout: lambda m, x, y: m.register_buffer('total_ops', torch.zeros(1)),
    nn.Identity: lambda m, x, y: m.register_buffer('total_ops', torch.zeros(1)),
}

if has_torchvision and DeformConv2d is not None:
    register_hooks[DeformConv2d] = count_deform_conv2d

# Enhanced Hooks with Shape Tracking
def wrap_hook(fn):
    def wrapper(m, x, y):
        try:
            # Store shapes
            if isinstance(x, tuple):
                 m.input_shape = [list(xi.shape) for xi in x if isinstance(xi, torch.Tensor)]
            elif isinstance(x, torch.Tensor):
                 m.input_shape = list(x.shape)
                 
            if isinstance(y, tuple):
                 m.output_shape = [list(yi.shape) for yi in y if isinstance(yi, torch.Tensor)]
            elif isinstance(y, torch.Tensor):
                 m.output_shape = list(y.shape)
                 
            fn(m, x, y)
        except Exception as e:
            # Silently ignore errors in FLOPs counting to avoid crashing the main execution
            # print(f"FlopsProfiler Error in {type(m).__name__}: {e}")
            pass
    return wrapper

# Re-wrap hooks
for k, v in register_hooks.items():
    register_hooks[k] = wrap_hook(v)

class FlopsProfiler:
    def __init__(self, model, forced_hooks=None):
        self.model = model
        self.hooks = []
        self.custom_hooks = forced_hooks if forced_hooks else {}
        self.started = False
        
    def start_profile(self):
        if self.started:
            return
        
        # Import Mask2FormerHead dynamically to avoid circular imports if possible, or assume it's available in context
        # But to be safe, we check class name string
        
        def _register_hook(m):
            m.register_buffer('total_ops', torch.zeros(1))
            m.register_buffer('total_params', torch.zeros(1))
            m.register_buffer('trainable_params', torch.zeros(1))
            
            # Count params (Total)
            params = 0
            trainable_params = 0
            for p in m.parameters(recurse=False):
                p_count = p.numel()
                params += p_count
                if p.requires_grad:
                    trainable_params += p_count
                    
            m.total_params = torch.FloatTensor([params])
            m.trainable_params = torch.FloatTensor([trainable_params])
            
            m_type = type(m)
            m_name = m_type.__name__
            
            fn = None
            
            # Special handling for Mask2FormerHead
            if m_name == 'Mask2FormerHead':
                fn = count_mask2former_head
            elif m_type in self.custom_hooks:
                fn = self.custom_hooks[m_type]
            elif m_type in register_hooks:
                fn = register_hooks[m_type]
            
            if fn is not None:
                handler = m.register_forward_hook(fn)
                self.hooks.append(handler)
        
        self.model.apply(_register_hook)
        self.started = True
        
    def stop_profile(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.started = False
        
    def get_total_flops(self):
        total_ops = 0
        for m in self.model.modules():
            if hasattr(m, 'total_ops'):
                total_ops += m.total_ops.item()
        return total_ops
        
    def get_total_params(self, only_trainable=False):
        total_params = 0
        for m in self.model.modules():
            if hasattr(m, 'total_params'):
                if only_trainable:
                    total_params += m.trainable_params.item()
                else:
                    total_params += m.total_params.item()
        return total_params

    def print_model_profile(self, detail=True):
        total_ops = self.get_total_flops()
        total_params = self.get_total_params(only_trainable=False)
        total_trainable_params = self.get_total_params(only_trainable=True)
        
        print(f"{'Layer':<50} | {'Input Shape':<25} | {'Output Shape':<25} | {'Params (T/All)':<15} | {'MACS':<15}")
        print("-" * 140)
        
        def _print_recursive(m, name, depth=0):
            # Only print if it has ops or params
            ops = m.total_ops.item() if hasattr(m, 'total_ops') else 0
            params = m.total_params.item() if hasattr(m, 'total_params') else 0
            t_params = m.trainable_params.item() if hasattr(m, 'trainable_params') else 0
            
            if ops == 0 and params == 0:
                pass
            
            # If it is a leaf module (in our registry) or has ops
            is_registered = type(m) in register_hooks
            
            # Print if it has interesting stats
            if is_registered or (ops > 0 or params > 0):
                indent = "  " * depth
                input_shape = str(getattr(m, 'input_shape', 'N/A'))
                output_shape = str(getattr(m, 'output_shape', 'N/A'))
                
                # Truncate shapes for display
                if len(input_shape) > 25: input_shape = input_shape[:22] + "..."
                if len(output_shape) > 25: output_shape = output_shape[:22] + "..."
                
                # Display Params as "Trainable/Total" if different, else just Total
                if params != t_params:
                    params_str = f"{int(t_params)}/{int(params)}"
                else:
                    params_str = f"{int(params)}"
                
                print(f"{indent + name:<50} | {input_shape:<25} | {output_shape:<25} | {params_str:<15} | {ops:<15.0f}")
            
            for child_name, child in m.named_children():
                _print_recursive(child, child_name, depth + 1)
                
        _print_recursive(self.model, "Model")
        print("-" * 140)
        print(f"Total MACs: {total_ops / 1e9:.4f} G")
        print(f"Total Params: {total_params / 1e6:.4f} M")
        print(f"Trainable Params: {total_trainable_params / 1e6:.4f} M")

# Alias FlopsProfilerV2 to FlopsProfiler for backward compatibility
class FlopsProfilerV2(FlopsProfiler):
    pass
