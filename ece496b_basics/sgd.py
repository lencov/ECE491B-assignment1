import torch
import math
from typing import Optional, Callable

# Custom SGD optimizer as described.
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                # Update using: p <- p - (lr / sqrt(t+1)) * grad
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss

# Run experiment for different learning rates.
learning_rates = [1e1, 1e2, 1e3]
num_iterations = 10

for lr in learning_rates:
    # Reinitialize weights for each run.
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    print(f"\n--- Running SGD with lr = {lr} ---")
    for t in range(num_iterations):
        opt.zero_grad()  # Reset gradients.
        loss = (weights ** 2).mean()  # Quadratic loss.
        print(f"Iteration {t}: Loss = {loss.item():.4f}")
        loss.backward()  # Compute gradients.
        opt.step()    