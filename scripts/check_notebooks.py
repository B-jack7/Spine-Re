"""Validate executable cells and their embedded sources without a GPU or data."""

import ast
import json
from pathlib import Path


def check(root):
    notebooks = [root / "notebooks/Spine_Re_Quickstart.ipynb", root / "reproduce/Spine_Re_Colab.ipynb"]
    document = json.loads(notebooks[0].read_text(encoding="utf-8"))
    assert document == json.loads(notebooks[1].read_text(encoding="utf-8")), "Notebook copies differ"
    payload = None
    helper = None
    for cell in document["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        compile(source, "<notebook>", "exec")
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
                continue
            if node.targets[0].id == "payload":
                payload = json.loads(ast.literal_eval(node.value.args[0]))
            elif node.targets[0].id == "runner_source":
                helper = ast.literal_eval(node.value)
    assert payload, "Embedded training sources missing"
    for name, source in payload.items():
        assert source == (root / "reproduce" / name).read_text(encoding="utf-8"), name
    assert helper == (root / "scripts/notebook_process.py").read_text(encoding="utf-8"), "Runner copy differs"
    print("Notebook cells compile; both copies and all embedded sources match.")


if __name__ == "__main__":
    check(Path(__file__).resolve().parents[1])
