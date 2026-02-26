from typing import Literal

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional import spearman_corrcoef

from src import chem, ops, oracle
from src.models.modules import NAM, TanimotoGP, ranknet_loss


class LitNAM(pl.LightningModule):

    def __init__(
        self,
        lib: str = "chemspace",
        width: int = 64,
        depth: int = 5,
        lr: float = 5e-4,
        score_batch_size: int = 10000,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__()

        fp_bbs = chem.SynthesisLibrary.read_fingerprints(lib)
        self.register_buffer("fp_bbs", torch.from_numpy(fp_bbs), persistent=False)

        self.nam = NAM(fp_bbs.shape[-1], width=width, depth=depth)

    def forward(self, bbids, batch):
        bbs = self.fp_bbs[bbids, :]
        return self.nam(bbs, batch)

    @torch.inference_mode()
    def score_blocks(self):
        preds = []
        B = self.hparams.score_batch_size
        for i in range(0, self.fp_bbs.shape[0], B):
            bbs = self.fp_bbs[i:i + B, :].float()
            preds.append(self.nam.score(bbs))
        return torch.cat(preds, dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.nam.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        return self._step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0  # only one step
        return self._step(batch, split="val")

    def _step(self, G, split):
        pred = self(bbids=G.x, batch=G.batch)
        target = G.y

        loss = ranknet_loss(pred, target)
        if split == "val":
            corr = spearman_corrcoef(pred, target)
            self.log("corr", corr, batch_size=1)
            self.log("alpha", self.nam.alpha().item(), batch_size=1)
        return loss


class AblateLitNAM(pl.LightningModule):

    def __init__(
        self,
        lib: str = "chemspace",
        objective: str = "jnk3",
        width: int = 64,
        depth: int = 5,
        lr: float = 5e-4,
        loss: Literal["mse", "rank"] = "rank",
        infer_seed: int = 0,
        infer_filter_topk: int = -1,
        infer_expand: int = 1,
        infer_batch_size: int = 10000,
        infer_num_workers: int = 10,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__()

        fp_bbs = chem.SynthesisLibrary.read_fingerprints(lib)
        self.register_buffer("fp_bbs", torch.from_numpy(fp_bbs), persistent=False)

        self.nam = NAM(fp_bbs.shape[-1], width=width, depth=depth)
        self.tgp = None
        self.train_data = None  # set outside

    def on_fit_start(self):
        self.tgp = TanimotoGP(*self.train_data, device=self.device)
        self.tgp.fit()
        del self.train_data

    def forward(self, bbids, batch):
        bbs = self.fp_bbs[bbids, :]
        return self.nam(bbs, batch)

    def score_blocks(self):
        preds = []
        B = self.hparams.infer_batch_size
        for i in range(0, self.fp_bbs.shape[0], B):
            bbs = self.fp_bbs[i:i + B, :].float()
            preds.append(self.nam.score(bbs))
        return torch.cat(preds, dim=0)

    def score_products(self, mols):
        fps = np.stack([chem.fingerprint(m, params="ml", asnumpy=True) for m in mols], axis=0)
        fps = torch.from_numpy(fps).float().to(self.device)
        return self.tgp(fps)[0]

    def configure_optimizers(self):
        return torch.optim.Adam(self.nam.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        return self._step(batch, split="train")

    def validation_step(self, batch, batch_idx):
        assert batch_idx == 0  # only one step
        return self._step(batch, split="val")

    def test_step(self, batch, batch_idx):
        assert batch_idx == 0  # only one step
        return self._step(batch, split="test")

    def _step(self, G, split):
        hp = self.hparams
        pred_nam = self(bbids=G.x, batch=G.batch)
        target = G.y

        if hp.loss == "mse":
            loss_fn = F.mse_loss
        elif hp.loss == "rank":
            loss_fn = ranknet_loss
        else:
            raise ValueError()
        loss = loss_fn(pred_nam, target)

        B = pred_nam.shape[0]
        self.log(f"{split}/loss", loss, batch_size=B)

        if split != "train":
            pred_gp, _ = self.tgp(G.mol)
            corr_nam = spearman_corrcoef(pred_nam, target)
            corr_mol = spearman_corrcoef(pred_gp, target)
            self.log(f"{split}/corr_nam", corr_nam, batch_size=1)
            self.log(f"{split}/corr_mol", corr_mol, batch_size=1)

        if split == "val":
            self.log("params/alpha", self.nam.alpha().item(), batch_size=1)

        return loss

    def on_test_epoch_end(self):
        hp = self.hparams
        rng = np.random.default_rng(hp.infer_seed)
        num_samples = 100

        if hp.infer_filter_topk > 0:
            subset = self.score_blocks().topk(hp.infer_filter_topk).indices
            subset = subset.cpu().numpy().tolist()
            libconfig = dict(name=hp.lib, subset=subset, eps=0.1)
        else:
            libconfig = dict(name=hp.lib)

        samples = set()
        with ops.ParallelMap(hp.infer_num_workers, init=infer_init, initargs=[libconfig]) as pmap:
            while len(samples) < hp.infer_expand * num_samples:
                m = (hp.infer_expand * num_samples) - len(samples)
                for T in pmap(infer_sample, rng.spawn(m)):
                    if chem.check_mol(T.root.mol):
                        samples.add(T.product)
        samples = sorted(samples)

        if len(samples) > num_samples:
            assert hp.infer_expand > 1
            scores = self.score_products(samples)
            indices = torch.topk(scores, k=num_samples).indices.cpu()
            samples = [samples[i] for i in indices]

        oracle.init(hp.objective)
        s = np.mean([oracle.call(smi) for smi in samples])
        self.log("test/scores", s, batch_size=1)


def infer_init(config):
    global _lib
    _lib = chem.SynthesisLibrary(**config)


def infer_sample(rng):
    global _lib
    chem.silence_rdlogger()
    return _lib.sample(rng=rng)
