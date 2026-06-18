import torch
import torch.nn as nn
import os

def load_te_checkpoint(model, checkpoint_path, map_location='cpu'):
    """
    Load a checkpoint that might have been trained with Transformer Engine (TE)
    into a model that might be running without TE (e.g., standard nn.Conv2d).
    
    Handles:
    1. 'module.' prefix (DataParallel/DDP).
    2. Reshaping 2D weights (te.Linear) to 4D weights (nn.Conv2d 1x1).
    3. Filtering out TE-specific keys like '_extra_state'.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    model_state = model.state_dict()
    loaded_state_dict = {}
    ignored_keys = []
    reshaped_keys = []
    
    for k, v in state_dict.items():
        # 1. Strip module. prefix
        if k.startswith('module.'):
            k = k[7:]
            
        # 2. Filter extra_state
        if k.endswith('_extra_state'):
            continue
            
        if k in model_state:
            target_shape = model_state[k].shape
            if v.shape != target_shape:
                # 3. Try reshaping 2D -> 4D for 1x1 Conv
                # Check if target is 4D 1x1 Conv and source is 2D Linear
                if v.dim() == 2 and len(target_shape) == 4 and target_shape[2] == 1 and target_shape[3] == 1:
                    if v.shape[0] == target_shape[0] and v.shape[1] == target_shape[1]:
                        v = v.view(target_shape)
                        reshaped_keys.append(k)
                    else:
                        # Size mismatch even after reshape
                        print(f"Warning: Size mismatch for {k}: {v.shape} vs {target_shape} (reshaping not possible)")
                        continue
                else:
                    print(f"Warning: Size mismatch for {k}: {v.shape} vs {target_shape}")
                    continue
            loaded_state_dict[k] = v
        else:
            ignored_keys.append(k)
            
    # Load
    missing, unexpected = model.load_state_dict(loaded_state_dict, strict=False)
    
    print(f"Loaded {len(loaded_state_dict)} keys.")
    if reshaped_keys:
        print(f"Reshaped {len(reshaped_keys)} keys (TE Linear -> Conv2d).")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
        
    return missing, unexpected
