"""Execute ordinary Python notebook cells in order; no extra Jupyter dependency."""
import contextlib
import io
import json
from pathlib import Path
import re
import sys


def main():
    here = Path(__file__).resolve().parent
    path = here / "notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    chapters = set()
    namespace = {"__name__": "__notebook__"}
    executed = 0
    for index, cell in enumerate(notebook["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            chapters.update(map(int, re.findall(r"^#+ Chapter (\d+):", source, re.MULTILINE)))
            continue
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, f"<notebook cell {index}>", "exec"), namespace)
        except Exception:
            print(f"Failed notebook cell {index}:\n{source}", file=sys.stderr)
            raise
        executed += 1
        cell["execution_count"] = executed
        cell["outputs"] = ([{"output_type": "stream", "name": "stdout",
                             "text": buffer.getvalue().splitlines(keepends=True)}]
                           if buffer.getvalue() else [])
    assert chapters == set(range(16)), f"Missing chapters: {set(range(16))-chapters}"
    assert notebook["metadata"]["companion_preserved_review_cells"] >= 67
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    output = here / "output"
    output.mkdir(exist_ok=True)
    report = {"chapters": sorted(chapters), "code_cells_executed": executed,
              "total_cells": len(notebook["cells"]), "success": True}
    (output / "notebook_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
