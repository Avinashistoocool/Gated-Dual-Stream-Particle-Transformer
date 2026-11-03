"""Extract readable, token-preserving pages from the independent companion.

Run before the chapter-14 render. Only presentation artifacts are written.
Explicit continuations change layout, not Python tokens or implementation.
"""
import ast
import io
import json
import textwrap
import tokenize
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGIN = HERE / "companion" / "gds_part_reconstruction.py"


def pages():
    source = ORIGIN.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}

    def method(name, member):
        node = next(n for n in classes[name].body
                    if isinstance(n, ast.FunctionDef) and n.name == member)
        return textwrap.dedent(ast.get_source_segment(source, node))

    def body(name, member):
        lines = method(name, member).splitlines()[1:]
        return textwrap.dedent("\n".join(lines))

    def span(text, start, end=None):
        lines = text.splitlines()
        a = next(i for i, line in enumerate(lines) if start in line)
        z = next((i for i in range(a + 1, len(lines)) if end and end in lines[i]), len(lines))
        return textwrap.dedent("\n".join(lines[a:z]))

    aggregate = body("RelationalGate", "aggregate")
    attention = body("GatedAttention", "forward")
    constructor = body("GDSParT", "__init__")
    forward = body("GDSParT", "forward")
    raw = {
        "build_particle": (
            span(body("ParticleEmbedding", "__init__"), "self.net")
            + "\n\n" + method("ParticleEmbedding", "forward")),
        "build_pair": (
            span(body("PairwiseEmbedding", "__init__"), "self.net")
            + "\n\n" + method("PairwiseEmbedding", "forward")),
        "build_shared": span(aggregate, "if self.ctx is None:", "if states is None:"),
        "build_identity": method("RelationalGate", "make_mlp"),
        "build_conditioned": span(aggregate, "context ="),
        "build_attention": span(attention, "query, key =", "if bias is not None:"),
        "build_attention_mask": span(attention, "if bias is not None:", "# Padding masks"),
        "build_attention_mix": span(attention, "weights ="),
        "build_block": method("GDSParTBlock", "forward"),
        "build_cls": method("ClassAttentionBlock", "forward"),
        "build_model_embed": span(constructor, "self.particle_embed", "if c.n_cls_layers:"),
        "build_model": span(constructor, "if c.n_cls_layers:"),
        "build_forward": span(forward, "states = self.particle_embed", "if self.cfg.n_cls_layers:"),
        "build_readout": span(forward, "if self.cfg.n_cls_layers:"),
    }

    def token_values(text):
        ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NEWLINE,
                   tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT}
        return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(text).readline)
                if t.type not in ignored]

    result = {}
    for key, text in raw.items():
        # Comment lines have already been explained in the narration.
        original = "\n".join(line for line in text.splitlines()
                             if not line.lstrip().startswith("#"))
        lines = []
        for line in original.splitlines():
            if len(line) <= 52:
                lines.append(line)
                continue
            indent = " " * (len(line) - len(line.lstrip()) + 4)
            wrapped = textwrap.wrap(line, width=50, subsequent_indent=indent,
                                    break_long_words=False, break_on_hyphens=False,
                                    replace_whitespace=False, drop_whitespace=True)
            lines.extend(part + " \\" if i < len(wrapped)-1 else part
                         for i, part in enumerate(wrapped))
        displayed = "\n".join(lines)
        if token_values(displayed) != token_values(original):
            raise ValueError(f"Presentation reflow changed code tokens: {key}")
        result[key] = {"source": displayed, "origin": str(ORIGIN.relative_to(HERE)),
                       "token_equivalence": True,
                       "pages": ["\n".join(lines[i:i+9]) for i in range(0, len(lines), 9)]}
    return result


def main():
    records = pages()
    output = HERE / "output" / "chapter_14_source_pages.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    for key, record in records.items():
        lines = record["source"].splitlines()
        print(f"{key}: {len(lines)} lines, {max(map(len, lines))} columns; tokens unchanged")


if __name__ == "__main__":
    main()
