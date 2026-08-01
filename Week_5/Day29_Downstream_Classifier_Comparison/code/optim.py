"""
optim.py
A tiny, dependency-free Adam optimizer that works uniformly on a single
pennylane/autograd ndarray (generator's theta) or a dict of ndarrays
(discriminator's weight dict). Keeps the project from needing torch.
"""

import numpy as np
from pennylane import numpy as pnp


def _zeros_like_struct(x):
    if isinstance(x, dict):
        return {k: np.zeros_like(np.asarray(v)) for k, v in x.items()}
    return np.zeros_like(np.asarray(x))


class AdamOptimizer:
    def __init__(self, lr, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = _zeros_like_struct(params)
            self.v = _zeros_like_struct(params)

        if isinstance(params, dict):
            new_params = {}
            for k in params:
                g = np.asarray(grads[k])
                self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
                self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g ** 2)
                m_hat = self.m[k] / (1 - self.b1 ** self.t)
                v_hat = self.v[k] / (1 - self.b2 ** self.t)
                updated = np.asarray(params[k]) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
                new_params[k] = pnp.array(updated, requires_grad=True)
            return new_params
        else:
            g = np.asarray(grads)
            self.m = self.b1 * self.m + (1 - self.b1) * g
            self.v = self.b2 * self.v + (1 - self.b2) * (g ** 2)
            m_hat = self.m / (1 - self.b1 ** self.t)
            v_hat = self.v / (1 - self.b2 ** self.t)
            updated = np.asarray(params) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            return pnp.array(updated, requires_grad=True)
