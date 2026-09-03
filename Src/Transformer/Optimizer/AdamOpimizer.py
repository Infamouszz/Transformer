import torch

class AdamOptimizer:
    def __init__(self, parameters, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.steps = 0

        self.m = {id(p): torch.zeros_like(p) for p in parameters}
        self.v = {id(p): torch.zeros_like(p) for p in parameters}

    def zero_grad(self, param_grads):
        for p, g in param_grads:
            if g is not None:
                g = torch.zeros_like(g)


    def update(self, params_grads):
        self.steps += 1

        for p, g in params_grads:
            if g is None:
                continue

            parameter_id = id(p)

            self.m[parameter_id] = self.beta1 * self.m[parameter_id] + (1 - self.beta1) * g
            self.v[parameter_id] = self.beta2 * self.v[parameter_id] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[parameter_id] / (1 - self.beta1 ** self.steps)
            v_hat = self.v[parameter_id] / (1 - self.beta2 ** self.steps)

            p -= self.alpha * m_hat / (torch.sqrt(v_hat) + self.eps)