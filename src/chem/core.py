from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors


def silence_rdlogger():
    logger = RDLogger.logger()
    logger.setLevel(RDLogger.CRITICAL)


def rdmol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    assert isinstance(mol, Chem.Mol), mol
    return mol


def csmiles(mol):
    return Chem.MolToSmiles(rdmol(mol), isomericSmiles=False)


def molwt(mol):
    return rdMolDescriptors.CalcExactMolWt(rdmol(mol))


def qed(mol):
    return Descriptors.qed(rdmol(mol))


def check_mol(mol, maxwt=1000, maxatoms=1000):
    return (
        (Chem.SanitizeMol(mol, catchErrors=True) == 0)
        and (1 < mol.GetNumAtoms() <= maxatoms)
        and (molwt(mol) <= maxwt)
    )
