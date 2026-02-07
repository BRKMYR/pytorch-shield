import torch
from shield import SafetyShield, compute_total_loss

# Mock setup
actions = torch.randn(10, 2) * 15.0  # Random actions, some unsafe (>10.0)
shield = SafetyShield(speed_limit=10.0)

# Calculate penalties
penalty = shield(actions)
print(f"Current ODD Violation Penalty: {penalty.item():.4f}")

# Example of backprop through the safety layer
loss = compute_total_loss(torch.tensor(0.5), penalty)
print(f"Total Assured Loss: {loss.item():.4f}")
