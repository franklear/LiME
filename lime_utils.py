import math
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class L0Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        droprate_init: float = 0.5,
        temperature: float = 2 / 3,
        lamba: float = 1.,
        local_rep: bool = False,
        limit_a: float = -0.1,
        limit_b: float = 1.1,
        epsilon: float = 1e-6,
        loga_init_std: float = 1e-2,
        mask_without_scale: bool = False,
    ):
        """
        :param num_embeddings: For the inner embedding
        :param embedding_dim: For the inner embedding
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        :param limit_a: Lower bound of the interval (gamma in the paper) to which the distribution is stretched
        :param limit_b: Upper bound of the interval (zeta in the paper)
        :param epsilon
        :param loga_init_std
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if not mask_without_scale:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)

        self.qz_loga_emb = nn.Embedding(num_embeddings, embedding_dim)
        self.temperature = temperature
        self.droprate_init = droprate_init
        self.lamba = lamba
        self.local_rep = local_rep

        self.limit_a = limit_a
        self.limit_b = limit_b
        self.epsilon = epsilon

        self.mask_without_scale = mask_without_scale

        self.emb_used = None

        if not mask_without_scale:
            self.emb.weight.data.fill_(1)
        self.qz_loga_emb.weight.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), loga_init_std)

    def pre_step_constrain_parameters(self, min: float = math.log(1e-2), max: float = math.log(1e2)) -> None:
        self.qz_loga_emb.weight.data.clamp_(min=min, max=max)
        self.emb_used = None

    def cdf_qz(self, x: float, qz_loga: torch.Tensor) -> torch.Tensor:
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - self.limit_a) / (self.limit_b - self.limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * self.temperature - qz_loga).clamp(min=self.epsilon, max=1 - self.epsilon)

    def quantile_concrete(self, x: torch.Tensor, qz_loga: torch.Tensor) -> torch.Tensor:
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = torch.sigmoid((torch.log(x) - torch.log(1 - x) + qz_loga) / self.temperature)
        return y * (self.limit_b - self.limit_a) + self.limit_a

    def l0_reg(self, qz_loga: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Expected L0 norm under the stochastic gates"""
        if qz_loga is None:
            if self.emb_used is None:
                qz_loga = self.qz_loga_emb.weight.data
            else:
                qz_loga = self.qz_loga_emb.weight.data[self.emb_used]

        lc = 1 - self.cdf_qz(0, qz_loga)
        lc = self.lamba * lc.mean()
        return lc

    def sample_z(self, prefix_shape: Iterable, qz_loga: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            u = torch.distributions.uniform.Uniform(self.epsilon, 1 - self.epsilon) \
                      .sample([*prefix_shape, self.embedding_dim]) \
                      .to(qz_loga.device)
            z = self.quantile_concrete(u, qz_loga)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:
            pi = torch.sigmoid(qz_loga).view(*prefix_shape, self.embedding_dim)
            return F.hardtanh(pi * (self.limit_b - self.limit_a) + self.limit_a, min_val=0, max_val=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.emb_used is None:
            self.emb_used = input.new_zeros(self.num_embeddings, dtype=torch.bool)
        self.emb_used[input] = True

        if not self.mask_without_scale:
            emb = self.emb(input)
        with torch.autograd.profiler.record_function("get z"):
            qz_loga = self.qz_loga_emb(input)

            if self.local_rep or not self.training:
                prefix_shape = [input.shape[0]] + [1] * len(input.shape[1:])
            else:
                prefix_shape = input.shape

            z = self.sample_z(prefix_shape, qz_loga, sample=self.training)
            z = z.expand(*input.shape, self.embedding_dim)

        if self.mask_without_scale:
            output = z
        else:
            output = emb * z
        return output
