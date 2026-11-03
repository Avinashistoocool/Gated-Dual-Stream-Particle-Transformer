"""Audit every registered narration, caption, equation label and code block."""
import ast
import json
import re
from pathlib import Path
from spellchecker import SpellChecker
from audit import TECHNICAL
from course_registry import load_course, extra_inventory

HERE = Path(__file__).resolve().parent
EXCEPTIONS = {
    "semileptonic": "A decay involving both hadronic and leptonic products.",
    "hadronic": "Involving hadrons.",
    "mha": "Multihead attention acronym.",
    "datatype": "Type of stored numerical value.",
    "modulelist": "PyTorch registered module collection.",
    "higgs": "Higgs boson.",
    "hbb": "JetClass Higgs-to-bottom label.",
    "hcc": "JetClass Higgs-to-charm label.",
    "hgg": "JetClass Higgs-to-gluon label.",
    "hqql": "JetClass semileptonic Higgs label.",
    "tbl": "JetClass semileptonic top label.",
    "tbqq": "JetClass hadronic top label.",
    "wqq": "JetClass hadronic W label.",
    "zqq": "JetClass hadronic Z label.",
    "flavour": "British physics spelling.",
    "px": "X momentum component.",
    "py": "Y momentum component.",
    "pz": "Z momentum component.",
    "atan": "Inverse tangent function identifier.",
    "rapidities": "Plural rapidity.",
    "modulo": "Remainder operation.",
    "subjets": "Resolved structures within a jet.",
    "pre": "Prefix in pre-normalization or pre-softmax.",
    "multihead": "Attention using several heads.",
    "multiheadattention": "PyTorch module identifier.",
    "backpropagate": "Apply the chain rule backward.",
    "cuda": "GPU computing platform.",
    "lookahead": "Optimizer method name.",
    "radam": "Rectified Adam optimizer.",
    "pretrained": "Previously trained model.",
    "gibibytes": "Binary storage unit.",
    "mebibytes": "Binary storage unit.",
    "unnormalized": "Not yet normalized.",
    "preactivation": "Value before a nonlinear activation.",
    "preactivations": "Values before a nonlinear activation.",
    "arctanh": "Inverse hyperbolic tangent.",
    "arctangent": "Inverse tangent.",
    "ungated": "Without multiplicative gates.",
    "diag": "Diagonal-matrix operator.",
    "neighbour": "British spelling of neighbor.",
    "neighbours": "British spelling of neighbors.",
    "exponentials": "Exponential functions.",
    "exponentiating": "Applying an exponential.",
    "exponentiate": "Apply an exponential.",
    "dataset": "Collection of examples.",
    "datasets": "Collections of examples.",
    "nonlinearly": "In a nonlinear manner.",
    "nonlinearities": "Nonlinear activation functions.",
    "workflow": "Ordered operational process.",
    "logits": "Unnormalized classifier or attention scores.",
    "logit": "One unnormalized score.",
    "softmax": "Exponential probability normalization.",
    "gelu": "Gaussian error linear unit.",
    "adamw": "Decoupled-weight-decay optimizer.",
    "backpropagation": "Reverse-mode application of the chain rule.",
    "autograd": "Automatic differentiation subsystem.",
    "jacobian": "Matrix of partial derivatives.",
    "jacobians": "Plural of Jacobian.",
    "hyperparameter": "A configuration choice, not a trained weight.",
    "hyperparameters": "Configuration choices.",
    "pseudorapidity": "Angular coordinate used in particle physics.",
    "rapidity": "Relativistic longitudinal coordinate.",
    "hadron": "Particle made of quarks.",
    "hadrons": "Plural of hadron.",
    "muon": "Particle species.",
    "azimuth": "Angular coordinate.",
    "kinematics": "Motion-related observables.",
    "diagonal": "Matrix diagonal.",
    "dropout": "Stochastic training operation.",
    "backward": "PyTorch reverse differentiation operation.",
    "checkpointing": "Recomputation technique for activation memory.",
    "overfit": "Fit training data with limited generalization.",
    "overfitting": "Fitting training data excessively.",
    "memorization": "Exact or near-exact fitting to a small dataset.",
    "memorizes": "Fits a fixed dataset closely.",
    "memorize": "Fit a fixed dataset closely.",
    "preprocessing": "Deterministic processing before model inference.",
    "reproducible": "Can be repeated under recorded conditions.",
    "reproducibility": "Ability to repeat a result.",
    "parameterization": "Choice of trainable representation.",
    "reparameterization": "Equivalent choice of trainable coordinates.",
    "reparameterisation": "British spelling of reparameterization.",
    "nonzero": "Not equal to zero.",
    "nonnegative": "Greater than or equal to zero.",
    "unbiased": "Statistical estimator convention.",
    "invariant": "Unchanged under a stated transformation.",
    "invariance": "Transformation property.",
    "permutation": "Reordering of entries.",
    "equivariant": "Output transforms consistently with reordered input.",
    "einsum": "Einstein-summation tensor API.",
    "sdpa": "Scaled dot-product attention API.",
    "jetclass": "Dataset name.",
    "auc": "Area under the ROC curve.",
    "roc": "Receiver operating characteristic.",
    "qcd": "Quantum chromodynamics.",
    "ema": "Exponential moving average.",
    "ffn": "Feed-forward network.",
    "ffns": "Plural feed-forward networks.",
    "mlp": "Multilayer perceptron.",
    "cls": "Class token notation.",
    "cpu": "Central processing unit.",
    "gpu": "Graphics processing unit.",
    "gpus": "Graphics processing units.",
    "lufs": "Loudness units relative to full scale.",
    "api": "Application programming interface.",
    "apis": "Plural of API.",
    "underpredicted": "Predicted below the target.",
    "off": "Used in off-diagonal.",
    "logarithmic": "Involving a logarithm.",
    "scaling": "Multiplication by a scale.",
    "rescaling": "Changing a scale.",
    "normalization": "Statistical or probability normalization.",
    "normalizes": "Performs normalization.",
    "normalized": "After normalization.",
    "normalizing": "Performing normalization.",
    "hyperbolic": "Describing hyperbolic functions.",
    "boolean": "True or false datatype.",
    "bool": "Boolean datatype identifier.",
    "hadamard": "Elementwise vector product.",
    "pytorch": "Framework name.",
    "tensor": "Multidimensional numerical array.",
    "tensors": "Plural tensor.",
}


def main():
    chapters, code = load_course()
    dictionary = SpellChecker(language="en")
    dictionary.word_frequency.load_words({**TECHNICAL, **EXCEPTIONS})
    strings, formulas, failures = [], [], []
    strings.append(("shared-flow", "step"))
    extra_strings, extra_formulas = extra_inventory()
    strings.extend(extra_strings)
    for name, formula in extra_formulas:
        formulas.append((name, "custom", formula))
        strings.extend((f"{name}/custom-math", label)
                       for label in re.findall(r"\\text\{([^}]+)\}", formula))
    for number, chapter in sorted(chapters.items()):
        for key, beat in chapter.items():
            if len(beat.caption.split()) > 12 or not beat.caption.endswith((".", "?")):
                failures.append((number, key, "Invalid caption", beat.caption))
            for kind, text in (("screen", beat.caption), ("narration", beat.narration)):
                strings.append((f"{number:02}/{key}/{kind}", text))
            if beat.kind == "code":
                ast.parse(code[beat.steps[0]])
            else:
                for formula in beat.steps:
                    formulas.append((number, key, formula))
                    for label in re.findall(r"\\text\{([^}]+)\}", formula):
                        strings.append((f"{number:02}/{key}/math-label", label))
    for path in HERE.glob("chapter_*.py"):
        if path.name.startswith(("chapter_00_", "chapter_01_")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            argument = node.args[0]
            if not isinstance(argument, ast.Constant) or not isinstance(argument.value, str):
                continue
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else "")
            if name in {"formula", "MathTex"}:
                formulas.append((path.name, node.lineno, argument.value))
                strings.extend((f"{path.name}:{node.lineno}/label", label)
                               for label in re.findall(r"\\text\{([^}]+)\}", argument.value))
            elif name in {"prose", "definition", "Text"}:
                strings.append((f"{path.name}:{node.lineno}", argument.value))
    for label, text in strings:
        terms = [word.lower() for word in re.findall(r"[A-Za-z]+", text) if len(word) > 1]
        unknown = dictionary.unknown(terms)
        if unknown:
            failures.append((label, sorted(unknown)))
    output = HERE / "output"
    output.mkdir(exist_ok=True)
    (output / "course_text_audit.json").write_text(json.dumps(
        {"strings": len(strings), "formulas": len(formulas), "failures": failures}, indent=2),
        encoding="utf-8")
    inventory = ["# Course strings checked", "", "Automatic spelling and syntax audit; not perceptual review.", ""]
    for label, text in strings:
        inventory.extend([f"## {label}", text, ""])
    inventory += ["## Mathematics", ""]
    inventory += [f"- Chapter {n}, {key}: `{formula}`" for n, key, formula in formulas]
    for key, source in code.items():
        inventory.extend([f"## Code: {key}", "```python", source, "```"])
    inventory += ["## Reviewed vocabulary", ""]
    inventory += [f"- {word}: {reason}" for word, reason in sorted({**TECHNICAL, **EXCEPTIONS}.items())]
    inventory += ["## Result", "PASS" if not failures else f"FAIL: {failures}"]
    (output / "course_strings_checked.md").write_text("\n".join(inventory), encoding="utf-8")
    if failures:
        raise ValueError(f"Audit failures: {failures}")
    print(f"PASS: {len(strings)} strings, {len(formulas)} formulas, {len(chapters)} chapters.")


if __name__ == "__main__":
    main()
