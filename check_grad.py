
import torch
import torch.nn as nn
from src.AsymFormerV10S4 import AsymFormerV10

def check_gradients():
    # Setup
    device = 'cpu'
    num_classes = 40
    model = AsymFormerV10(num_classes=num_classes, embed_dim=256, num_latents=128, query_proj_dim=48)
    model.train()
    
    # Enable gradients for frozen params just to see if flow reaches them (if they weren't frozen)
    # But in the script they are frozen. Let's unfreeze them to check connectivity.
    for p in model.parameters():
        p.requires_grad = True
        
    # Dummy Input
    B = 2
    H, W = 320, 320
    rgb = torch.randn(B, 3, H, W, requires_grad=True)
    depth = torch.randn(B, 1, H, W, requires_grad=True)
    
    # Forward
    print("Forward pass...")
    out = model(rgb, depth)
    
    logits = out['pred_logits'] # [B, 48, 41]
    masks = out['pred_masks']   # [B, 48, H, W]
    
    print(f"Logits shape: {logits.shape}")
    print(f"Masks shape: {masks.shape}")
    
    # Fake Loss
    loss = logits.sum() + masks.sum()
    
    # Backward
    print("Backward pass...")
    loss.backward()
    
    # Check Gradients
    print("\nChecking gradients...")
    
    # Check Head components
    print(f"Head Class MLP grad: {model.head.class_mlp[-1].weight.grad is not None}")
    print(f"Head Query Projector grad: {model.head.query_projector.weight.grad is not None}")
    
    # Check Latents (Original)
    # Note: In real training these are frozen, but here we unfroze to check flow
    print(f"Latents S8 grad: {model.latents_s8.grad is not None}")
    
    # Check Backbone
    print(f"Backbone Stem grad: {model.backbone.stem.rgb_stem[0].weight.grad is not None}")
    
    # Check Input
    print(f"Input RGB grad: {rgb.grad is not None}")
    
    # Check for NaNs
    if torch.isnan(logits).any():
        print("Logits contain NaNs!")
    if torch.isnan(masks).any():
        print("Masks contain NaNs!")
        
    # Check magnitude of masks
    print(f"Masks mean: {masks.mean().item()}, std: {masks.std().item()}")
    print(f"Masks max: {masks.max().item()}, min: {masks.min().item()}")

if __name__ == "__main__":
    check_gradients()
