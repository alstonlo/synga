import dataclasses
import hashlib
import pathlib
import shutil
import subprocess
import tempfile

import jsonargparse
import yaml
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers


# Copied with minor modifications from:
# https://github.com/luost26/ChemProjector/blob/main/chemprojector/tools/docking.py


@dataclasses.dataclass
class QVinaOption:

    center_x: float
    center_y: float
    center_z: float
    size_x: float = 20.0
    size_y: float = 20.0
    size_z: float = 20.0
    exhaustiveness: int = 16

    @classmethod
    def from_config(cls, path: pathlib.Path):
        with open(path, "r") as f:
            com = yaml.safe_load(f)["center_of_mass"]
        return QVinaOption(center_x=com[0], center_y=com[1], center_z=com[2])


@dataclasses.dataclass
class QVinaOutput:

    receptor: str
    smiles: str
    path: pathlib.Path
    mode_id: int
    affinity: float


class QVinaDockingTask:

    def __init__(self, receptor: str, ligand: str, outroot: pathlib.Path):
        super().__init__()

        recroot = pathlib.Path(__file__).parent / "receptors" / receptor
        self.receptor_name = receptor
        self.receptor_path = recroot / "receptor.pdbqt"
        self.ligand = ligand
        self.option = QVinaOption.from_config(recroot / "description.yaml")

        fname = hashlib.sha256(ligand.encode("utf-8")).hexdigest()
        self.outpath = outroot / f"{fname}.sdf"

    def __call__(self) -> QVinaOutput:
        with tempfile.TemporaryDirectory() as workdir:
            out = self._process(pathlib.Path(workdir))
        return min(out, key=lambda x: x.affinity) if out else None

    def _process(self, workdir):
        if not self.outpath.exists():
            prep_success = self._prepare_ligand(workdir)
            if not prep_success:
                print(f"Ligand preparation failed: {self.ligand}")
                return []
            vina_out = self._run_docking(workdir)
            if vina_out.returncode != 0:
                print(f"Docking failed: {self.ligand}")
                return []
        return self._parse_qvina_outputs(self.outpath)

    def _prepare_ligand(self, workdir):
        mol = Chem.MolFromSmiles(self.ligand)
        mol = Chem.AddHs(mol, addCoords=True)
        try:
            rdDistGeom.EmbedMolecule(mol)
            rdForceFieldHelpers.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"Failed to optimize molecule: {self.ligand}", e)
            return False
        sdf_writer = Chem.SDWriter(str(workdir / "ligand.sdf"))
        sdf_writer.write(mol)
        sdf_writer.close()

        out = subprocess.run(
            ["obabel", "ligand.sdf", "-O", "ligand.pdbqt"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        return out.returncode == 0

    def _run_docking(self, workdir):
        shutil.copy(self.receptor_path, workdir / "receptor.pdbqt")

        cmd = [
            "qvina2",
            "--receptor", "receptor.pdbqt",
            "--ligand", "ligand.pdbqt",
            "--center_x", f"{self.option.center_x:.3f}",
            "--center_y", f"{self.option.center_y:.3f}",
            "--center_z", f"{self.option.center_z:.3f}",
            "--size_x", f"{self.option.size_x:.3f}",
            "--size_y", f"{self.option.size_y:.3f}",
            "--size_z", f"{self.option.size_z:.3f}",
            "--exhaustiveness", f"{self.option.exhaustiveness}",
        ]

        vina_out = subprocess.run(
            cmd,
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["obabel", "ligand_out.pdbqt", "-O", "ligand_out.sdf"],
            cwd=workdir,
            check=False,
            capture_output=True,
        )
        self.outpath.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(workdir / "ligand_out.sdf", self.outpath)

        return vina_out

    def _parse_qvina_outputs(self, docked_sdf_path):
        if not docked_sdf_path.exists():
            return []
        sdf_lines = docked_sdf_path.read_text().splitlines()
        scores = [float(line.strip().split()[2]) for line in sdf_lines if line.startswith(" VINA RESULT:")]

        return [
            QVinaOutput(
                receptor=self.receptor_name,
                smiles=self.ligand,
                path=docked_sdf_path,
                mode_id=i,
                affinity=score,
            )
            for i, score in enumerate(scores, start=1)
        ]


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()
    parser.add_class_arguments(QVinaDockingTask)
    args = parser.parse_args()

    init = parser.instantiate_classes(args)
    task = QVinaDockingTask(**init.as_dict())
    print(task())
