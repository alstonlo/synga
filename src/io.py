import pathlib
import uuid
from typing import Optional

from lightning.pytorch.loggers import WandbLogger as _PLWandbLogger

CODE_ROOT = pathlib.Path(__file__).parent
REPO_ROOT = CODE_ROOT.parent

DATA_ROOT = REPO_ROOT / "data"
LIBS_ROOT = DATA_ROOT / "libs"

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def random_checkpoint_dir():
    rand_dir = DATA_ROOT / "checkpoints" / str(uuid.uuid4())
    assert not rand_dir.exists()
    return str(rand_dir)


def readlines(path):
    with open(path, "r") as f:
        return f.read().splitlines()


def writelines(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# This just overrides the initializer signature for jsonargparse
class WandbLogger(_PLWandbLogger):

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        dir: pathlib.Path = LOG_DIR,
    ):
        super().__init__(project=project, name=name, entity=entity, group=group, save_dir=dir)


class NoLogger:

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_table(self, key, columns=None, data=None, dataframe=None, step=None):
        pass
