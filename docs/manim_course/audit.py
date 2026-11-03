"""Fail before video rendering if registered English text contains unknown words."""
import ast
import json
import re
from pathlib import Path

from spellchecker import SpellChecker
from content import CHAPTER_00, CHAPTER_01, SUMMARY_00, SUMMARY_01, TEXT_COMMON, CODE

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
TECHNICAL = {
    "elementwise": "Standard mathematical term: applied separately to matching entries.",
    "channelwise": "Standard ML term: applied separately to feature channels.",
    "pytorch": "Framework name.",
    "hadamard": "Mathematical product named after Hadamard.",
    "softmax": "Mathematical function.",
    "gds": "Project acronym.",
    "part": "Project acronym component; also an English word.",
    "dp": "Relation-width dimension label defined on screen.",
    "layernorm": "PyTorch layer name.",
    "mlp": "Multilayer perceptron acronym.",
    "mlps": "Plural of the MLP acronym.",
    "cls": "Class-token notation.",
    "equivariance": "Mathematical transformation property.",
    "hyperparameters": "Configuration parameters.",
    "collimated": "Physics term for a narrow directional distribution.",
    "kinematic": "Physics term for motion-related quantities.",
    "kinematics": "Physics term.",
    "colour": "Standard British English spelling.",
    "memorising": "Standard British English spelling.",
    "memorised": "Standard British English spelling.",
    "behaviour": "Standard British English spelling.",
    "recurrently": "Adverb describing recurrent updates.",
    "rescale": "Mathematical scaling operation.",
    "rescales": "Inflected form of rescale.",
}


def main():
    OUT.mkdir(exist_ok=True)
    spell = SpellChecker(language="en")
    spell.word_frequency.load_words(TECHNICAL)
    strings = []
    for ci, chapter in enumerate((CHAPTER_00, CHAPTER_01)):
        for key, (caption, narration, source) in chapter.items():
            if len(caption.split()) > 12:
                raise ValueError(f"More than twelve words: {caption}")
            strings.append((f"{ci:02}/{key}/screen", caption))
            strings.append((f"{ci:02}/{key}/narration", narration))
    strings.extend(("common", t) for t in TEXT_COMMON + SUMMARY_00 + SUMMARY_01)
    review_scenes = [*HERE.glob("chapter_00_*.py"), *HERE.glob("chapter_01_*.py")]
    for path in review_scenes:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args:
                if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    if node.func.attr in {"prose", "definition"}:
                        strings.append((f"{path.name}:{node.lineno}", node.args[0].value))
                    elif node.func.attr == "formula":
                        for label in re.findall(r"\\text\{([^}]+)\}", node.args[0].value):
                            strings.append((f"{path.name}:{node.lineno}/math-label", label))
    strings.append(("minimap", "Inputs Embeddings Attention Prediction"))
    failures = []
    for label, text in strings:
        tokens = [word.lower() for word in re.findall(r"[A-Za-z]+", text) if len(word) > 1]
        unknown = spell.unknown(tokens)
        if unknown:
            failures.append((label, sorted(unknown)))
    code_inventory = []
    for key, code in CODE.items():
        ast.parse(code)
        code_inventory.append((key, code))
    # Mathematics and code are checked as syntax/identifiers, not misspelled prose.
    math_inventory = []
    for path in review_scenes:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "formula" and node.args and isinstance(node.args[0], ast.Constant):
                    math_inventory.append((path.name, node.args[0].value))
    report = ["# Strings checked before rendering", "",
              "Dictionary: pyspellchecker English; reviewed technical exceptions appear below.", ""]
    for label, text in strings:
        report.extend([f"## {label}", text, ""])
    report += ["## Programmatically rendered mathematics", ""]
    report += [f"- `{file}`: `{text}`" for file, text in math_inventory]
    for key, code in code_inventory:
        report.extend([f"\n## Executable code: {key}", "```python", code, "```"])
    report += ["\n## Reviewed dictionary exceptions", ""]
    report += [f"- **{word}**: {reason}" for word, reason in sorted(TECHNICAL.items())]
    report += ["\n## Result", "PASS" if not failures else f"FAIL: {failures}"]
    (OUT / "strings_checked.md").write_text("\n".join(report), encoding="utf-8")
    (OUT / "text_audit.json").write_text(json.dumps({"strings": len(strings),
        "formula_literals": len(math_inventory), "failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"Spell-check failures: {failures}")
    print(f"PASS: {len(strings)} English strings and {len(math_inventory)} formula literals audited.")


if __name__ == "__main__":
    main()
