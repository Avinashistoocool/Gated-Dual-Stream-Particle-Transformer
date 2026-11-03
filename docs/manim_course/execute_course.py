"""Execute the exact on-screen snippets in the installed PyTorch environment."""
import contextlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "companion"))
from course_registry import load_course, code_preludes


def main():
    chapters, code = load_course()
    preludes = code_preludes()
    outputs = {}
    for number, chapter in sorted(chapters.items()):
        namespace = {}
        if number in preludes:
            exec(compile(preludes[number], f"<chapter-{number}-prelude>", "exec"), namespace)
        for beat in chapter.values():
            if beat.kind != "code":
                continue
            key = beat.steps[0]
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(code[key], f"<course-example-{key}>", "exec"), namespace)
            outputs[key] = buffer.getvalue()
            if not outputs[key].strip():
                raise ValueError(f"No observable output from example {key}")
    output = HERE / "output" / "course_executed_examples.json"
    output.parent.mkdir(exist_ok=True)
    pending = output.with_suffix(".next.json")
    pending.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    pending.replace(output)
    print(f"Executed {len(outputs)} displayed examples for {len(chapters)} chapters.")


if __name__ == "__main__":
    main()
