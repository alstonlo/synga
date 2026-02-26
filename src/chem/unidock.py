import logging
import pathlib
import tempfile

import numpy as np
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from unidock_tools.modules.docking import run_unidock
from unidock_tools.modules.ligand_prep import TopologyBuilder
from unidock_tools.modules.protein_prep import pdb2pdbqt

from src import io
from src.chem.core import rdmol

LITPCBA_RECEPTORS = [
    "ADRB2", "ALDH1", "ESR_ago", "ESR_ant", "FEN1",
    "GBA", "IDH1", "KAT2A", "MAPK1", "MTORC1",
    "OPRK1", "PKM2", "PPARG", "TP53", "VDR",
]


# Adapted from:
# https://github.com/SeonghwanSeo/RxnFlow/blob/master/src/rxnflow/tasks/utils/unidock.py


class UniDocker:

    def __init__(self, receptor: str):
        logging.getLogger().setLevel(logging.WARNING)

        if receptor == "ESR_ant":
            receptor = "ESR_antago"
        recroot = io.DATA_ROOT / "LIT-PCBA" / receptor
        if not recroot.exists():
            raise ValueError()

        # Load center from ligand file
        pbmol = next(pybel.readfile("mol2", str(recroot / "ligand.mol2")))
        coords = [atom.coords for atom in pbmol.atoms]
        self.center = tuple(np.mean(coords, axis=0).round(2).tolist())

        # Create pdbqt file
        self.receptor_name = receptor
        self.receptor_path = recroot / "protein.pdbqt"
        if not self.receptor_path.exists():
            pdb2pdbqt(recroot / "protein.pdb", self.receptor_path)

    def __call__(self, mols, pmap):
        with tempfile.TemporaryDirectory() as workdir:
            return self.docking(mols, workdir, pmap=pmap)

    def docking(self, mols, workdir, pmap):
        workdir = pathlib.Path(workdir)

        # Prepare molecules
        ligands = pmap(prepare_ligand, enumerate(mols), workdir=workdir)
        ligands = [p for p in ligands if (p is not None)]

        # Run UniDock
        outputs = run_unidock(
            receptor=self.receptor_path,
            ligands=ligands,
            output_dir=workdir,
            center_x=self.center[0],
            center_y=self.center[1],
            center_z=self.center[2],
            search_mode="balance",
            num_modes=1,
            seed=1,
            refine_step=3,
        )

        # Parse scores
        scores = [0.0] * len(mols)
        for path, s in zip(*outputs):
            i = int(path.stem.split("_")[0])
            scores[i] = s[0] if s else 0.0
        return scores


def prepare_ligand(input, workdir):
    i, smi = input
    sdf_path = workdir / f"{i}.sdf"

    param = rdDistGeom.srETKDGv3()
    param.randomSeed = 1

    # Prepare molecule
    try:
        mol = rdmol(smi)
        mol = Chem.AddHs(mol)
        rdDistGeom.EmbedMolecule(mol, param)
        if mol.GetNumConformers() == 0:
            raise ValueError("embed failed")
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        print(f"Ligand preparation failed {smi}: {e}")
        return None

    try:
        topo_builder = TopologyBuilder(mol=mol)
        topo_builder.build_molecular_graph()
        topo_builder.write_sdf_file(sdf_path, do_rigid_docking=False)
        return sdf_path
    except Exception as e:
        logging.error(f"Topology builder failed: {smi}: {e}")
        return None
