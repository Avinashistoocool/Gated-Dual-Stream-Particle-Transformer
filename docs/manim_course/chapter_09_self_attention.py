"""Animated, narrated attention derivations; isolated renderer for Chapters 09--12."""
import json
import re
import numpy as np
from studio import *
from course_studio import CourseScene
import chapter_09_content as lesson09


class AttentionLesson(CourseScene):
    lesson = lesson09
    chapter = lesson09.CHAPTER
    chapter_number = 9

    def formula(self, tex, position=ORIGIN, color=TEXT, size=48):
        self.inventory.append({"kind": "math", "text": tex})
        palette = {"q": "Q", "Q": "Q", "k": "K", "K": "K", "v": "V", "V": "V",
                   "g": "G", "G": "G", "X": "X", "R": "R", "M": "M"}
        def tint(match):
            token = match.group()
            if token != r"\mathcal B" and token not in palette:
                return token
            role = "B" if token == r"\mathcal B" else palette[token]
            return r"{\color[HTML]{" + COLORS[role].lstrip("#") + "}"+token+"}"
        colored = re.sub(
            r"\\mathcal B|\\(?:text|operatorname|mathrm|texttt)\{[^}]*\}"
            r"|\\(?:mathbb|mathcal)\s*\{?[A-Za-z]\}?|\\[A-Za-z]+|[qQkKvVgGXRM]",
            tint, tex)
        # A colored symbol inside a TeX group must not become Manim's {{...}} splitter.
        colored = re.sub(r"\{(?=\{)", "{ ", colored)
        if not hasattr(self, "_attention_tex_template"):
            self._attention_tex_template = TexTemplate()
            self._attention_tex_template.add_to_preamble(r"\usepackage{xcolor}")
        item = MathTex(colored, font_size=size, color=color,
                       tex_template=self._attention_tex_template)
        if item.width > 12.3:
            item.scale_to_fit_width(12.3)
        return item.move_to(position)

    def setup(self):
        super().setup()
        self.remove(self.map_marker)
        self.map_index = 4
        self.map_marker = SurroundingRectangle(self.minimap[4], color=COLORS["X"], buff=.08)
        self.add(self.map_marker)

    def flow_diagram(self, kind):
        if kind == "none":
            return VGroup()
        if kind == "heads":
            labels = [("X", "X"), ("h_1", "Q"), ("h_2", "K"), ("O", "V")]
        elif kind == "gates":
            labels = [("R", "R"), ("s_i", "R"), ("g_i^Q", "G"), ("q_i'", "Q")]
        elif kind == "bias":
            labels = [("r_{ij}", "R"), (r"\mathcal B_{ij}", "B"), ("S_{ij}", "B"), ("A_{ij}", "V")]
        else:
            labels = [("X", "X"), ("Q", "Q"), ("K", "K"), ("V", "V")]
        boxes = VGroup()
        for index, (tex, color) in enumerate(labels):
            position = LEFT * 4.5 + RIGHT * index * 3 + DOWN * 1.4
            label = self.formula(tex, position, COLORS[color], 38)
            box = RoundedRectangle(width=1.65, height=.72, corner_radius=.12,
                                   stroke_color=COLORS[color], fill_color=COLORS[color],
                                   fill_opacity=.08).move_to(position)
            boxes.add(VGroup(box, label))
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i+1].get_left(),
                               buff=.1, color=GREY_B, stroke_width=2) for i in range(3)])
        return VGroup(boxes, arrows)

    def attention_diagram(self, values):
        bars = VGroup()
        for i, value in enumerate(values):
            y = -.9 - i * .72
            label = self.formula(f"A_{i+1}", LEFT * 4.5 + UP * y, COLORS["K"], 30)
            outline = Rectangle(width=5, height=.35, stroke_color=COLORS["M"], stroke_width=1)
            outline.move_to(LEFT * .8 + UP * y)
            fill = Rectangle(width=max(.002, 5 * value), height=.35, stroke_width=0,
                             fill_color=COLORS["Q"] if i == 0 else COLORS["K"], fill_opacity=.85)
            fill.align_to(outline, LEFT).set_y(y)
            number = DecimalNumber(value, num_decimal_places=4, font_size=30)
            number.move_to(RIGHT * 3.3 + UP * y)
            bars.add(VGroup(label, outline, fill, number))
        return bars

    def matrix_diagram(self):
        matrix = VGroup()
        for i in range(3):
            row = VGroup()
            for j in range(3):
                cell = Square(.45, stroke_width=1, stroke_color=COLORS["M"],
                              fill_color=COLORS["Q"] if i == 0 else COLORS["K"],
                              fill_opacity=.2 + .15 * (j == i))
                cell.move_to(RIGHT * (j - 1) * .5 + DOWN * (.9 + i * .5))
                row.add(cell)
            matrix.add(row)
        return matrix

    def derive(self, block):
        self.wipe()
        if self.tag is not None:
            self.remove(self.tag)
            self.tag = None
        if block["shape"]:
            self.shape(block["shape"], COLORS["X"])
        definition = self.definition(block["definition"]) if block["definition"] else VGroup()
        equations = [self.formula(s, UP * .65, size=44) for s in block["steps"]]
        kind = block["diagram"]
        if kind == "attention":
            diagrams = [self.attention_diagram(v) for v in block["weights"]]
        elif kind == "matrix":
            diagrams = [self.matrix_diagram()]
        else:
            diagrams = [self.flow_diagram(kind)]

        def action(index):
            if index == 0:
                return AnimationGroup(Write(equations[0]), FadeIn(definition), FadeIn(diagrams[0]))
            animations = [TransformMatchingTex(equations[index-1], equations[index])]
            if len(diagrams) > index:
                animations.append(ReplacementTransform(diagrams[index-1], diagrams[index]))
            elif kind == "matrix":
                if block["key"] == "key_derivative":
                    selected = VGroup(*[row[1] for row in diagrams[0]])
                elif block["key"] in {"query_derivative", "softmax_derivative"}:
                    selected = diagrams[0][1]
                else:
                    selected = diagrams[0][(index-1) % 3]
                animations.append(Indicate(selected, color=COLORS["G"]))
            elif len(diagrams[0]) and kind != "attention":
                animations.append(Indicate(diagrams[0][0][(index-1) % 4]))
            return animations
        self.teach(block["key"], [lambda i=i: action(i) for i in range(len(equations))], True)

    def executed(self, block):
        self.wipe()
        if self.tag is not None:
            self.remove(self.tag)
            self.tag = None
        source = self.lesson.CODE[block["code"]]
        outputs = json.loads((HERE / "output" / "course_executed_examples.json")
                             .read_text(encoding="utf-8"))
        output_key = f"c{self.chapter_number:02}_{block['code']}"
        code = Code(code_string=source, language="python", formatter_style="monokai",
                    add_line_numbers=False, background="rectangle",
                    background_config={"fill_color": "#141F35", "stroke_color": GREY_D},
                    paragraph_config={"font": "Consolas", "font_size": 28})
        if code.width > 12.5 or code.height > 4.9:
            raise ValueError(f"Code must fit without reducing 28px type: {block['code']}")
        code.move_to(UP * .1)
        result = Text(outputs[output_key].rstrip(), font="Consolas", font_size=28,
                      color=COLORS["X"], line_spacing=.8).move_to(UP * .2)
        if result.width > 12.5 or result.height > 4.5:
            raise ValueError(f"Output too wide: {block['code']}")
        label = self.prose("Executed PyTorch output.", UP * 2.1, COLORS["V"])
        self.inventory.append({"kind": "code", "text": source, "output": outputs[output_key],
                               "font_size": 28})
        self.teach(block["key"], [
            lambda: FadeIn(code.background),
            lambda: LaggedStart(*[Write(line) for line in code.code_lines], lag_ratio=.12),
            lambda: AnimationGroup(FadeOut(code), Write(label), Write(result)),
            lambda: Circumscribe(result, color=COLORS["V"])], True)

    def construct(self):
        for block in self.lesson.BEATS:
            if block["key"] == "summary":
                self.wipe()
                lines = VGroup(*[self.prose(line, UP * (1-i)) for i, line in enumerate(self.lesson.SUMMARY)])
                self.teach("summary", [lambda: Write(lines[0]), lambda: Write(lines[1]),
                                      lambda: AnimationGroup(FadeOut(self.caption), Write(lines[2]))])
            elif block["code"]:
                self.executed(block)
            else:
                self.derive(block)
        self.finish()


class Chapter09(AttentionLesson):
    pass
