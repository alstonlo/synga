from typing import Literal

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import tqdm
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryPrecisionAtFixedRecall,
)

from src import chem, ops
from src.models.modules import MLP


class SimFilter(nn.Module):

    def __init__(self):
        super().__init__()

        # We give the illusion that this is trainable so PL doesn't crash
        self.zero = nn.Parameter(torch.tensor(0.0))

    def forward(self, q, bb, return_probs):
        sum_kwargs = dict(dim=-1, dtype=torch.float)
        sims = torch.minimum(q, bb).sum(**sum_kwargs) / bb.sum(**sum_kwargs)
        if return_probs:
            return sims
        return sims.logit(eps=1e-5) + (0 * self.zero)


class MLPFilter(MLP):

    def forward(self, q, bb, return_probs):
        x = torch.cat([q, bb, torch.minimum(q, bb)], dim=-1).float()
        x = super().forward(x).squeeze(-1)
        return x.sigmoid() if return_probs else x


class LitBlockFilter(pl.LightningModule):

    def __init__(
        self,
        lib: str = "chemspace",
        method: Literal["mlp", "sim"] = "mlp",
        width: int = 256,
        depth: int = 5,
        lr: float = 5e-4,
        infer_batch_size: int = 10000,
    ):
        self.save_hyperparameters(logger=False)
        super().__init__()

        fp_bbs = chem.SynthesisLibrary.read_fingerprints(lib)
        self.register_buffer("fp_bbs", torch.from_numpy(fp_bbs), persistent=False)

        # Core model
        if method == "sim":
            self.filter = SimFilter()
        elif method == "mlp":
            self.filter = MLPFilter(3 * fp_bbs.shape[-1], width=width, depth=depth)
        else:
            raise ValueError()

        # Metric computation
        self.binclass_fns = torchmetrics.MetricCollection({
            "auroc": BinaryAUROC(thresholds=1000),
            "auprc": BinaryAveragePrecision(thresholds=1000),
            "pr@r1": BinaryPrecisionAtFixedRecall(thresholds=1000, min_recall=1.0),
        }, compute_groups=[["auroc", "auprc", "pr@r1"]])
        self.metrics = ops.ListOfMetrics()

    def forward(self, q, bbids, return_probs=False):
        bb = self.fp_bbs[bbids, :]
        q, bb = torch.broadcast_tensors(q, bb)
        return self.filter(q, bb, return_probs=return_probs)

    def score_blocks(self, q):
        preds = []
        B = self.hparams.infer_batch_size
        for i in range(0, self.fp_bbs.shape[0], B):
            preds.append(self(q, slice(i, i + B), return_probs=True))  # (B)
        return torch.cat(preds, dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        q, bb_pos, bb_neg = batch  # (B D) (B) (B)
        B = q.shape[0]

        qq = torch.cat([q, q], dim=0)  # (2B D)
        bb = torch.cat([bb_pos, bb_neg], dim=0)  # (2B)
        labels = torch.full([2 * B], False, device=self.device)
        labels[:B] = True

        logits = self(qq, bb)  # (2B)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Logging
        self.log("train/loss", loss, batch_size=B)

        return loss

    def validation_step(self, batch, batch_idx):
        q, bbids = batch  # (1 D) (1 M), M << N blocks
        assert (q.ndim == bbids.ndim == 2) and (q.shape[0] == bbids.shape[0] == 1)

        preds = self.score_blocks(q)
        target = torch.zeros_like(preds, dtype=torch.int)
        target[bbids[0]] = 1

        # Metrics
        self.binclass_fns.update(preds, target)
        m = self.binclass_fns.compute()
        self.binclass_fns.reset()

        m["pr@r1"], m["threshold@r1"] = m.pop("pr@r1")
        m = {k: v.cpu().item() for k, v in m.items()}
        self.metrics.update(m)

        return None

    def on_validation_epoch_end(self) -> None:
        m = ops.fix_keys(self.metrics.mean_and_std(), pre="val/")
        self.log_dict(m)
        self.metrics.reset()


class BlockFilter:

    def __init__(
        self,
        checkpoint: str,
        threshold: float = 0.5,
        expand_sim: float = 1.0,
        eps: float = 0.1,
    ):
        self.config = dict(locals())
        self.config.pop("self")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lit = LitBlockFilter.load_from_checkpoint(checkpoint, map_location=device)
        self.lit.eval()

        self.threshold = threshold
        self.expand_sim = expand_sim
        self.eps = eps

    @torch.inference_mode()
    def __call__(self, lib, queries):
        assert self.lit.hparams.lib == lib

        top_bbs = []
        for q in tqdm.tqdm(queries, desc="Filtering Blocks"):
            x = chem.fingerprint(q, params="ml", asnumpy=True)
            x = torch.tensor(x, device=self.lit.device).float()
            bbs = torch.argwhere(self.lit.score_blocks(x) >= self.threshold)
            bbs = bbs.squeeze(-1).cpu().tolist()
            top_bbs.append(bbs)

        if self.expand_sim < 1.0:
            expand_bbs = []
            nbs = chem.SynthesisLibrary.read_neighbors(lib, minsim=self.expand_sim)
            for bbs in top_bbs:
                bbs = set(bbs).union(*(set(nbs[i]) for i in bbs))
                expand_bbs.append(sorted(bbs))
            return expand_bbs
        else:
            return top_bbs
