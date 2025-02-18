import math
import torch
from torch.optim.optimizer import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0):
        """
        Implements the AdamW optimizer.

        Args:
            params: Iterable of parameters to optimize.
            lr (float): Learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator for numerical stability.
            weight_decay (float): Weight decay (L2 penalty) coefficient.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        Returns:
            The loss evaluated after the step (if closure is provided), otherwise None.
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                    state["t"] = 0

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                # Update biased first moment estimate.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second moment estimate.
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected first and second moment estimates.
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                step_size = lr * (math.sqrt(bias_correction2) / bias_correction1)

                # Update parameters using AdamW rule.
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)
                # Decoupled weight decay.
                p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss