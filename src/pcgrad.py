import torch
import torch.nn as nn
import torch.optim as optim
import random

class PCGrad:
    """
    PCGrad: Projected Conflict Gradient for Multi-Task Learning.
    
    Args:
        optimizer (torch.optim.Optimizer): The base optimizer.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
    """
    def __init__(self, optimizer, scaler=None):
        self.optimizer = optimizer
        self.scaler = scaler

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def state(self):
        return self.optimizer.state

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def backward(self, objectives, **kwargs):
        """
        Compute gradients for each objective and project conflicting gradients.
        
        Args:
            objectives (list): A list of loss tensors (objectives) to optimize.
        """
        grads = []
        params = []
        
        # Identify parameters that require grad
        # We need a consistent order of parameters
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    params.append(p)

        # Compute gradients for each objective
        for i, obj in enumerate(objectives):
            self.optimizer.zero_grad()
            
            # Optimization: Only retain graph for objectives before the last one
            # This helps to free up memory used by the computation graph earlier
            retain = i < len(objectives) - 1
            
            if self.scaler is not None:
                # If scaler is provided, we assume mixed precision
                self.scaler.scale(obj).backward(retain_graph=retain)
            else:
                obj.backward(retain_graph=retain)
            
            # Flatten gradients
            grad_list = []
            for p in params:
                if p.grad is not None:
                    # Detach to ensure we don't keep graph history in the stored gradients
                    grad_list.append(p.grad.detach().flatten())
                else:
                    # Treat None as zero gradient
                    grad_list.append(torch.zeros_like(p).flatten())
            
            if grad_list:
                grads.append(torch.cat(grad_list))
            else:
                grads.append(None)

        # Clear gradients before accumulating projected ones
        self.optimizer.zero_grad()
        
        # Shuffle for random order
        indices = list(range(len(grads)))
        random.shuffle(indices)
        
        # Working copy of gradients
        working_grads = []
        for g in grads:
            if g is not None:
                working_grads.append(g.clone())
            else:
                working_grads.append(None)
        
        # Project
        # PCGrad: g_i = g_i - (g_i . g_j) * g_j / ||g_j||^2 if g_i . g_j < 0
        for i in indices:
            if working_grads[i] is None: continue
            
            for j in indices:
                if i == j or working_grads[j] is None: continue
                
                g_i = working_grads[i]
                g_j = working_grads[j]
                
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2 + 1e-8)
                    working_grads[i] = g_i
        
        # Sum projected gradients
        final_grad = None
        for g in working_grads:
            if g is not None:
                if final_grad is None:
                    final_grad = g
                else:
                    final_grad += g
            
        if final_grad is not None:
            # Unflatten and assign back to parameters
            idx = 0
            for p in params:
                num_param = p.numel()
                # We assign the projected gradient
                # If we are using AMP, this gradient is scaled.
                # scaler.step() will unscale it later.
                if p.grad is None:
                    p.grad = final_grad[idx:idx+num_param].view(p.shape).clone()
                else:
                    p.grad.copy_(final_grad[idx:idx+num_param].view(p.shape))
                idx += num_param
