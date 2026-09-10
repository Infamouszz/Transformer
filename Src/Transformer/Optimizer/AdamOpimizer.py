import math

import torch

class AdamOptimizer:
    def __init__(self, parameters, total_steps, alpha=1e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.steps = 0
        self.total_steps = total_steps

        self.m = {id(p): torch.zeros_like(p) for p in parameters}
        self.v = {id(p): torch.zeros_like(p) for p in parameters}

    def zero_grad(self, param_grads):
        for p, g in param_grads:
            if g is not None:
                g.zero_()


    def update(self, params_grads):
        self.steps += 1
        current_alpha = self.cosine_alpha_decay(self.steps, self.total_steps)

        for p, g in params_grads:
            if g is None:
                continue

            parameter_id = id(p)

            self.m[parameter_id] = self.beta1 * self.m[parameter_id] + (1 - self.beta1) * g
            self.v[parameter_id] = self.beta2 * self.v[parameter_id] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[parameter_id] / (1 - self.beta1 ** self.steps)
            v_hat = self.v[parameter_id] / (1 - self.beta2 ** self.steps)

            p -= current_alpha * m_hat / (torch.sqrt(v_hat) + self.eps)

    def clip_grad_norm_(self, params_grads, max_norm=1.0):
        total_sq_norm = 0.0

        for p, g in params_grads:
            if g is not None:
                total_sq_norm += (g ** 2).sum().item()

        total_norm = total_sq_norm ** 0.5

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p, g in params_grads:
                if g is not None:
                    g *= scale

        return total_norm

    def cosine_decay_warmup(self, step, total_steps):
        warmup_steps = total_steps * 0.1
        if step < warmup_steps:
            return self.alpha * (step / max(1, warmup_steps))

        max_alpha = self.alpha
        min_alpha = self.alpha * 0.2
        current_step = min(step, total_steps)

        target_learning_rate = min_alpha + 0.5 * (max_alpha - min_alpha) * (1 + math.cos((current_step/total_steps)*math.pi))

        return target_learning_rate
