"""Chapter 4: shared affine layers, nonlinearity and experimental discipline."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter04(CourseScene):
    chapter_number = 4

    def animate_beat(self, key, beat):
        if key != "affine":
            return super().animate_beat(key, beat)
        weight = Matrix([[1, 3], [-2, 4]],
                        element_to_mobject_config={"font_size": 38}).move_to(LEFT*3.3+UP*.4)
        inputs = Matrix([[2], [-1]], element_to_mobject_config={"font_size": 38})
        inputs.set_color(COLORS["X"]).move_to(LEFT*.5+UP*.4)
        bias = Matrix([[1], [-1]], element_to_mobject_config={"font_size": 38})
        bias.move_to(RIGHT*2.3+UP*.4)
        labels = VGroup(self.formula("W", size=34).next_to(weight, UP, buff=.3),
                        self.formula("x", size=34, color=COLORS["X"]).next_to(inputs, UP, buff=.3),
                        self.formula("b", size=34).next_to(bias, UP, buff=.3))
        plus = self.formula("+", RIGHT*.9+UP*.4, size=42)
        first = self.formula(r"h_1=1(2)+3(-1)+1=0", DOWN*1.35, size=38)
        second = self.formula(r"h_2=-2(2)+4(-1)-1=-9", DOWN*2.2, size=38)
        result = self.formula(r"h=Wx+b=(0,-9)", DOWN*1.7, size=42)
        row0 = SurroundingRectangle(weight.get_rows()[0], color=COLORS["Q"], buff=.15)
        row1 = SurroundingRectangle(weight.get_rows()[1], color=COLORS["Q"], buff=.15)
        self.teach(key, [
            lambda: [FadeIn(weight), FadeIn(inputs), FadeIn(bias), Write(labels), Write(plus)],
            lambda: [Create(row0), Indicate(bias.get_entries()[0], color=COLORS["Q"]), Write(first)],
            lambda: [ReplacementTransform(row0, row1),
                     Indicate(bias.get_entries()[1], color=COLORS["Q"]), Write(second)],
            lambda: [FadeOut(first), FadeOut(second), Write(result)],
        ], equation_pause=True)
