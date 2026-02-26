import warnings
from typing import Literal, Optional

import jsonargparse
import lightning.pytorch as pl
import polars as ps
import torch
from torch.utils.data import DataLoader, Dataset

from src import chem, io, ops
from src.models.bbfilter.lit import LitBlockFilter
from src.models.trainers import SimpleTrainer

# Suppress polars multiprocess
warnings.filterwarnings("ignore", message=r"^Using fork")


class BlockRecallDataset(Dataset):

    def __init__(self, mol2bbs):
        self.mol2bbs = mol2bbs

    def choice(self, L):
        high = L if isinstance(L, int) else len(L)
        idx = torch.randint(high, size=[1]).item()
        return idx if isinstance(L, int) else L[idx]

    def __len__(self):
        return len(self.mol2bbs)

    def __getitem__(self, idx):
        mol, bbids, mut = self.mol2bbs[idx]
        fp_mol = chem.fingerprint(mol, params="ml", asnumpy=True)
        bbids = torch.tensor(bbids).int()
        return fp_mol, bbids


class BlockPairDataset(BlockRecallDataset):

    def __init__(self, mol2bbs, nbs, pmine=0.5):
        super().__init__(mol2bbs)

        self.num_blocks = len(nbs)
        self.nbs = nbs if (pmine > 0) else None
        self.pmine = pmine

    def __getitem__(self, idx):
        mol, bbids = super().__getitem__(idx)
        pos = self.choice(bbids)
        neg = None

        # Maybe try to mine a harder negative BB
        if torch.rand(1) < self.pmine:
            hard_negs = ops.unique(self.nbs[pos], exclude=set(bbids))
            if hard_negs:
                neg = self.choice(hard_negs)
        while (neg is None) or (neg in bbids):
            neg = self.choice(self.num_blocks)

        return mol, pos, neg


class BlockFilterDataModule(pl.LightningDataModule):

    def __init__(
        self,
        lib: str,
        batch_size: int = 1024,
        num_workers: int = 8,
        mine_prob: float = 0.0,
        mine_topk: int = 100,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        df = ps.read_parquet(io.LIBS_ROOT / lib / "block_recall.parquet")
        df = df.select(["train", "mol", "bbs"])
        df = df.rows_by_key(key="train")

        nbs = chem.SynthesisLibrary.read_neighbors(lib, maxk=mine_topk)
        self.trainset = BlockPairDataset(df[True], nbs, pmine=mine_prob)
        self.valset = BlockRecallDataset(df[False])

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valset,
            batch_size=1,
            num_workers=min(4, self.num_workers),
            shuffle=True,
        )


def run():
    parser = jsonargparse.ArgumentParser()

    # Populate arguments
    parser.add_argument("--action", type=Literal["train", "val"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul_precision", type=str, default="high")
    parser.add_argument("--ckpt_path", type=Optional[str], default=None)

    parser.add_class_arguments(BlockFilterDataModule, "data")
    parser.add_class_arguments(LitBlockFilter, "model")
    parser.add_class_arguments(SimpleTrainer, "trainer")

    parser.link_arguments("model.lib", "data.lib")

    # Parse
    args = parser.parse_args()

    # Instantiate torch settings
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    # Instantiate
    init = parser.instantiate_classes(args)

    # Launch
    if args.ckpt_path is not None:
        init.model = LitBlockFilter.load_from_checkpoint(args.ckpt_path, map_location="cpu")
        args.model = dict(init.model.hparams)
    for logger in init.trainer.loggers:
        logger.log_hyperparams(args.as_dict())

    run_kwargs = dict(model=init.model, datamodule=init.data)
    if args.action == "train":
        init.trainer.fit(**run_kwargs)
        init.trainer.limit_val_batches = 1.0
        init.trainer.validate(**run_kwargs)
    elif args.action == "val":
        assert args.trainer.val_steps_per_epoch is None
        init.trainer.validate(**run_kwargs)
    else:
        raise ValueError()


if __name__ == "__main__":
    run()
