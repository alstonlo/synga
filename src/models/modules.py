import botorch
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from botorch.models.gp_regression import SingleTaskGP
from gauche.kernels.fingerprint_kernels import MinMaxKernel
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


def ranknet_loss(preds, target):
    assert preds.ndim == target.ndim == 1
    xs = torch.combinations(preds).T
    ys = torch.combinations(target).T
    t = 0.5 * (1 + torch.sign(ys[0] - ys[1]))
    return F.binary_cross_entropy_with_logits(xs[0] - xs[1], t)


class MLP(nn.Module):

    def __init__(self, features, width, depth, out_features=1, bn=True):
        super().__init__()

        mlp = [nn.Linear(features, width)]
        for i in range(1, depth):
            mlp.extend([
                nn.BatchNorm1d(width) if bn else nn.Identity(),
                nn.GELU(),
                nn.Linear(width, width if (i + 1 < depth) else out_features),
            ])
        self.mlp = nn.Sequential(*mlp)

    def forward(self, input):
        return self.mlp(input)


class NAM(nn.Module):

    def __init__(self, features, width, depth):
        super().__init__()

        self.scorer = MLP(features, out_features=1, width=width, depth=depth, bn=False)
        self._alpha = nn.Parameter(torch.zeros([1]))

    def alpha(self):
        return self._alpha.sigmoid()

    def score(self, inputs):
        return self.scorer(inputs.float()).squeeze(-1)

    def forward(self, inputs, batch):
        h = self.score(inputs)
        a = self.alpha()
        h_sum = pyg.nn.global_add_pool(h, batch)
        h_avg = pyg.nn.global_mean_pool(h, batch)
        y = a * h_sum + (1 - a) * h_avg
        return y


class TanimotoGP(nn.Module):

    def __init__(self, X, y, device):
        super().__init__()

        self.ttype = dict(dtype=torch.double, device=device)
        X = torch.tensor(X, **self.ttype)
        y = torch.tensor(y, **self.ttype).unsqueeze(-1)

        self.gp = SingleTaskGP(
            X, y,
            train_Yvar=torch.full_like(y, 1e-6),
            covar_module=ScaleKernel(MinMaxKernel()),
            likelihood=GaussianLikelihood(),
            outcome_transform=None,
        )

    @property
    def hparams(self):
        return {
            "mean": self.gp.mean_module.constant.item(),
            "outputscale": self.gp.covar_module.outputscale.item(),
            "noise": self.gp.likelihood.noise.item(),
        }

    def fit(self):
        self.gp.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        botorch.fit.fit_gpytorch_mll(mll)
        self.gp.eval()

    # Reference: https://github.com/AustinT/basic-mol-bo-workshop2024/blob/main/tanimoto_gpbo.py
    def manual_fit(self, hparams=None):
        if hparams is None:
            hparams = dict()
        self.gp.mean_module.constant = hparams.get("mean", 0.0)
        self.gp.covar_module.outputscale = hparams.get("outputscale", 1.0)
        self.gp.likelihood.noise = hparams.get("noise", 1e-4)
        self.gp.eval()

    def __call__(self, inputs):
        posterior = self.gp.posterior(inputs.to(**self.ttype), observation_noise=True)
        mean, stddev = posterior.mean, posterior.stddev
        return mean.squeeze(-1), stddev.squeeze(-1)

    def ucb(self, inputs, beta):
        mean, stddev = self(inputs)
        return mean + beta * stddev
