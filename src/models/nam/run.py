import warnings
from typing import Literal

import jsonargparse
import lightning.pytorch as pl
import polars as ps
import torch
import torch_geometric as pyg
from torch.utils.data import Dataset, random_split

from src import chem, io
from src.models.nam.lit import AblateLitNAM
from src.models.trainers import SimpleTrainer

# Suppress polars multiprocess
warnings.filterwarnings("ignore", message=r"^Using fork")


class NAMDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        fp_mol = chem.fingerprint(row["mol"], params="ml", asnumpy=True)
        return pyg.data.Data(
            x=torch.tensor(row["bbs"]).int(),
            y=row["score"],
            mol=torch.from_numpy(fp_mol).unsqueeze(0),
        )


class NAMDataModule(pl.LightningDataModule):

    def __init__(
        self,
        fold: int = 0,
        lib: str = "chemspace",
        objective: str = "jnk3",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 0,
    ):
        super().__init__()

        self.fold = fold
        self.objective = objective
        self.batch_size = batch_size
        self.num_workers = num_workers

        df = ps.read_parquet(io.LIBS_ROOT / lib / "block_nam.parquet")
        df = df.filter((ps.col("fold") == fold) & (ps.col("objective") == objective))
        df = df.partition_by("split", as_dict=True)

        g = torch.Generator().manual_seed(seed)
        trainval = NAMDataset(df[(0,)].rows(named=True))
        train, val = random_split(trainval, [0.9, 0.1], generator=g)
        test = NAMDataset(df[(1,)].rows(named=True))
        self.datasets = {"train": train, "val": val, "test": test}

    def train_dataloader(self):
        return self._loader(split="train")

    def val_dataloader(self):
        return self._loader(split="val")

    def test_dataloader(self):
        return self._loader(split="test")

    def _loader(self, split):
        training = (split == "train")
        dataset = self.datasets[split]
        return pyg.loader.DataLoader(
            dataset=dataset,
            batch_size=(self.batch_size if training else len(dataset)),
            num_workers=self.num_workers,
            shuffle=training,
            drop_last=training,
            persistent_workers=(self.num_workers > 0),
        )

    def train_data(self):
        X, y = zip(*[(G.mol, G.y) for G in self.datasets["train"]])
        X = torch.cat(X, dim=0).float()
        y = torch.tensor(y).float()
        return X, y


def run():
    parser = jsonargparse.ArgumentParser()

    # Populate arguments
    parser.add_argument("--action", type=Literal["train", "test"], default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--matmul_precision", type=str, default="high")

    parser.add_class_arguments(NAMDataModule, "data")
    parser.add_class_arguments(AblateLitNAM, "model")
    parser.add_class_arguments(SimpleTrainer, "trainer")

    parser.link_arguments("fold", "data.fold")
    parser.link_arguments("model.lib", "data.lib")
    parser.link_arguments("model.objective", "data.objective")
    parser.link_arguments(("seed", "fold"), "model.infer_seed", compute_fn=(lambda *x: sum(x)))

    # Parse
    args = parser.parse_args()

    # Instantiate torch settings
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(args.matmul_precision)

    # Instantiate
    init = parser.instantiate_classes(args)

    # Launch
    for logger in init.trainer.loggers:
        logger.log_hyperparams(args.as_dict())
    init.model.train_data = init.data.train_data()

    run_kwargs = dict(model=init.model, datamodule=init.data)
    if args.action == "train":
        init.trainer.fit(**run_kwargs)
        init.trainer.test(**run_kwargs)
    elif args.action == "test":
        init.trainer.test(**run_kwargs)
    else:
        raise ValueError()


if __name__ == "__main__":
    run()
