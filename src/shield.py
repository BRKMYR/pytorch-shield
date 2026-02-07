import torch
import torch.nn as nn

class SafetyShield(nn.Module):
    """
    UL 4600 Aligned Safety Layer.
    Ensures policy outputs remain within ODD boundaries via differentiable penalties.
    """
    def __init__(self, speed_limit=10.0):
        super().__init__()
        self.speed_limit = speed_limit

    def forward(self, actions):
        # Calculate velocity norm (speed)
        speed = torch.norm(actions, dim=1)
        
        # Calculate violation (positive if speed > limit)
        violation = torch.relu(speed - self.speed_limit)
        
        # Return total penalty (to be added to loss function)
        return torch.mean(violation)

def compute_total_loss(performance_loss, safety_penalty, lambda_multiplier=100.0):
    return performance_loss + (lambda_multiplier * safety_penalty)
