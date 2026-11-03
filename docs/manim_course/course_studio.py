"""Sentence-timed mathematical transformations for the remaining course.

The scene is a calculation workspace: equations transform, matrix entries are
selected, probability bars change numerically, and graph edges carry states.
Each chapter remains independently renderable and may override animate_beat.
"""
import json
import re
from pathlib import Path
import numpy as np
from manim import *
from studio import TeachingScene, TEXT, FONT, HERE
from content import COLORS
from course_registry import load_course


class CourseScene(TeachingScene):
    def setup(self):
        chapters, self.course_code = load_course()
        self.beats = chapters[self.chapter_number]
        self.chapter = {key: (beat.caption, beat.narration, beat.source)
                        for key, beat in self.beats.items()}
        super().setup()

    def construct(self):
        for key, beat in self.beats.items():
            self.wipe()
            self.animate_beat(key, beat)
        self.finish()

    def update_map(self, color):
        index = 0 if self.chapter_number in (2, 6, 7) else (
            2 if self.chapter_number in (4, 5, 8) else 6 if self.chapter_number >= 13 else 4)
        self.remove(self.map_marker)
        self.map_index = index
        self.map_marker = SurroundingRectangle(self.minimap[index], color=COLORS[color], buff=.08)
        self.add(self.map_marker)

    def animate_beat(self, key, beat):
        self.update_map(beat.color)
        if beat.kind == "code":
            return self.show_code(key, beat.steps[0])
        equations = [self.formula(step, size=39, color=COLORS[beat.color]) for step in beat.steps]
        if key == "summary" and len(equations) == 3:
            for index, equation in enumerate(equations):
                equation.move_to(UP * (1.5-index*1.35))
            self.teach(key, [lambda equation=equation: Write(equation) for equation in equations],
                       equation_pause=True)
            return
        self.current_equation = None
        self.previous_equations = []
        self.visual = None
        self.cells = []
        self.focus = None
        self.bars = None
        if beat.kind == "matrix" and beat.values:
            self.visual = Matrix(beat.values, element_to_mobject_config={"font_size": 30},
                                 h_buff=.9, v_buff=.65).set_color(COLORS[beat.color])
            if self.visual.width > 10:
                raise ValueError(f"Matrix too wide: {key}")
            self.visual.move_to(DOWN * 1.65)
            self.cells = list(self.visual.get_entries())
        elif beat.kind == "weights" and beat.values:
            self.bars = self.make_bars(beat.values[0], beat.color)
            self.visual = self.bars
        elif beat.kind == "flow":
            self.visual = self.make_flow(len(equations), beat.color)
        count = len(equations)

        def action(index):
            def execute():
                animations = []
                new = equations[index].move_to(UP * .8)
                if self.current_equation is None:
                    animations.append(Write(new))
                    if self.visual is not None:
                        animations.append(FadeIn(self.visual))
                else:
                    old = self.current_equation
                    # Preserve the immediately preceding calculation above the active one.
                    history = old.copy().set_opacity(.45).scale(.72).move_to(UP * 2.0)
                    if self.previous_equations:
                        animations.append(FadeOut(self.previous_equations[-1]))
                    animations.append(FadeIn(history))
                    self.previous_equations.append(history)
                    animations.append(TransformMatchingShapes(old, new))
                self.current_equation = new
                if self.cells:
                    chosen = self.cells[min(index, len(self.cells)-1)]
                    rectangle = SurroundingRectangle(chosen, buff=.11, color=COLORS["G"])
                    if self.focus is None:
                        animations.append(Create(rectangle))
                    else:
                        animations.append(ReplacementTransform(self.focus, rectangle))
                    self.focus = rectangle
                if self.bars is not None and index:
                    stage = min(index, len(beat.values)-1)
                    updated = self.make_bars(beat.values[stage], beat.color)
                    animations.append(ReplacementTransform(self.bars, updated))
                    self.bars = updated
                if beat.kind == "flow" and index and self.visual is not None:
                    nodes = self.visual[0]
                    animations.append(Indicate(nodes[min(index, len(nodes)-1)], color=COLORS["G"]))
                    if index < len(nodes):
                        dot = Dot(nodes[index-1].get_center(), color=COLORS[beat.color], radius=.08)
                        self.add(dot)
                        animations.append(Succession(
                            MoveAlongPath(dot, Line(nodes[index-1].get_center(), nodes[index].get_center())),
                            FadeOut(dot)))
                return animations
            return execute
        actions = [action(i) for i in range(count)]
        sentences = re.split(r"(?<=[.!?])\s+", beat.narration)
        # If math has more stages than sentences, group final stages sequentially.
        if count > len(sentences):
            last = actions[len(sentences)-1:]
            def remaining_actions():
                result = []
                for callback in last:
                    result.append(AnimationGroup(*callback()))
                return Succession(*result)
            actions = actions[:len(sentences)-1] + [remaining_actions]
        self.teach(key, actions, equation_pause=True)

    def make_bars(self, values, color):
        numbers = np.asarray(values, dtype=float)
        maximum = max(float(abs(numbers).max()), 1.)
        baseline = Line(LEFT*4+DOWN*2.7, RIGHT*4+DOWN*2.7, color=GREY_B)
        group = VGroup(baseline)
        spacing = min(2, 7/max(1, len(numbers)))
        for i, value in enumerate(numbers):
            height = max(.035, float(abs(value))/maximum*1.65)
            bar = Rectangle(width=.7, height=height, fill_opacity=.8,
                            color=COLORS[color]).move_to(
                np.array([(i-(len(numbers)-1)/2)*spacing, -2.7+height/2, 0]))
            label = DecimalNumber(float(value), num_decimal_places=4, font_size=25,
                                  color=TEXT).next_to(bar, UP, buff=.12)
            index = MathTex(str(i+1), font_size=26).next_to(bar, DOWN, buff=.1)
            group.add(bar, label, index)
        return group

    def make_flow(self, count, color):
        count = max(2, min(count, 5))
        nodes = VGroup(*[
            VGroup(RoundedRectangle(width=1.6, height=.85, corner_radius=.12,
                                    color=COLORS[color]),
                   MathTex(r"\text{step }" + str(i+1), font_size=30, color=COLORS[color]))
            for i in range(count)]).arrange(RIGHT, buff=.7).move_to(DOWN*1.7)
        edges = VGroup(*[Arrow(nodes[i].get_right(), nodes[i+1].get_left(),
                              buff=.08, color=GREY_B) for i in range(count-1)])
        return VGroup(nodes, edges)

    def show_code(self, key, code_key):
        source = self.course_code[code_key]
        outputs = json.loads((HERE / "output" / "course_executed_examples.json").read_text(encoding="utf-8"))
        code = Code(code_string=source, language="python", formatter_style="monokai",
                    add_line_numbers=False, background="rectangle",
                    background_config={"fill_color": "#141F35", "stroke_color": GREY_D},
                    paragraph_config={"font": "Consolas", "font_size": 28})
        if code.width > 12.5 or code.height > 4.65:
            raise ValueError(f"Code does not fit at readable font size: {code_key}, {code.width}, {code.height}")
        code.move_to(UP*.45)
        output = Text(outputs[code_key].rstrip(), font="Consolas", font_size=28,
                      color=COLORS["X"], line_spacing=.8).next_to(code, DOWN, buff=.35)
        if output.width > 12.7 or output.get_bottom()[1] < -3.2:
            raise ValueError(f"Code output does not fit: {code_key}")
        self.inventory.append({"kind": "code", "text": source, "output": outputs[code_key]})
        self.teach(key, [
            lambda: FadeIn(code.background),
            lambda: LaggedStart(*[Write(line) for line in code.code_lines], lag_ratio=.12),
            lambda: Write(output),
            lambda: Circumscribe(output, color=COLORS["X"]),
        ], equation_pause=True)
