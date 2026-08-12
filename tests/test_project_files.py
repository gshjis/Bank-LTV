import json
from pathlib import Path


def test_eda_notebook_is_valid_and_code_cells_compile():
    notebook = json.loads(Path("src/EDA.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"EDA cell {index}", "exec")
