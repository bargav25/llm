

from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch.optim import Optimizer


class AdamW(Optimizer):

    def __init__(self, params, weight_decay=0.01, lr = 1e-3, betas = (0.9, 0.999), eps=1e-8):

        defaults = {"lr": lr, "betas": betas,
                    "weight_decay": weight_decay, 
                    "eps": eps}

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):

        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                grad = p.grad.data

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]
                # Update first and second moment estimates
                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

                alpha_t = lr * math.sqrt(1-beta2**t) / (1 - beta1**t)

                # Update parameters
                p.data.addcdiv_(m, torch.sqrt(v) + epsilon, value=-alpha_t)

                # Apply weight_decay
                p.data.add_(p.data, alpha = -lr * weight_decay)


        return loss
