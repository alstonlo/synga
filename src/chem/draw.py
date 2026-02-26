import pathlib

try:
    import html2image
except ImportError:
    html2image = None
from PIL import Image

from src import io
from src.chem.core import csmiles


def draw_mol(path, mol, width=800, height=800, crop=True):
    if html2image is None:
        raise ImportError("html2image not installed")
    if not isinstance(mol, str):
        mol = csmiles(mol)

    path = pathlib.Path(path)
    with open(io.DATA_ROOT / "viz.html", "r") as f:
        html = f.read()
    for k, v in dict(smiles=mol, width=width, height=height).items():
        html = html.replace("{{" + k + "}}", str(v))
    hti = html2image.Html2Image(output_path=path.parent)
    hti.screenshot(html_str=html, save_as=path.name, size=(width, height))

    if crop:
        m = 10  # add a thin margin
        image = Image.open(path)
        x1, y1, x2, y2 = image.getbbox()
        image = image.crop((x1 - m, y1 - m, x2 + m, y2 + m))
        image.save(path)
