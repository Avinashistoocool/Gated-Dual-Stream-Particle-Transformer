"""Jets, all seventeen observed features, and explicit padding masks."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter06(CourseScene):
    chapter_number = 6

    def animate_beat(self, key, beat):
        self.shape(r"F:(B,N,17)\quad m:(B,N)", COLORS["X"])
        if key == "jet":
            dots = VGroup(*[Dot([x, y, 0], color=COLORS["X"], radius=.09)
                           for x, y in [(-5, 1), (-4, -1), (-3, .7), (-2, -1.5),
                                        (-1, 1.4), (0, -.7), (1, .6), (2, -1.3)]])
            destinations = [(-1.4, .4), (-1, -.4), (-.7, .8), (-.4, -.7),
                            (0, .4), (.3, -.3), (.7, .5), (1, -.5)]
            outline = Ellipse(width=4.2, height=2.8, color=COLORS["X"])
            label = self.formula(r"\{p_i\}_{i=1}^{n}\longrightarrow y_{\rm jet}", DOWN*2, size=42)
            self.teach(key, [
                lambda: LaggedStart(*[FadeIn(dot) for dot in dots], lag_ratio=.08),
                lambda: AnimationGroup(*[dot.animate.move_to([x, y, 0])
                                         for dot, (x, y) in zip(dots, destinations)]),
                lambda: Create(outline),
                lambda: Write(label)])
            return
        if key == "angles_numeric":
            axes = Axes(x_range=[0, .6, .1], y_range=[0, .5, .1],
                        x_length=4, y_length=10/3, tips=False).shift(LEFT*1.2+DOWN*.25)
            a, b, c = axes.c2p(0, 0), axes.c2p(.4, 0), axes.c2p(.4, .3)
            horizontal = Line(a, b, color=COLORS["X"])
            vertical = Line(b, c, color=COLORS["R"])
            diagonal = Line(a, c, color=COLORS["G"])
            labels = VGroup(self.formula(r"\Delta\phi=0.4", size=30).next_to(horizontal, DOWN),
                            self.formula(r"\Delta\eta=0.3", size=30).next_to(vertical, RIGHT))
            equation = self.formula(r"\sqrt{0.3^2+0.4^2}=0.5", UP*2, size=40)
            answer = self.formula(r"f_4=4(0.5-0.2)=1.2", UP*2, size=40)
            self.teach(key, [lambda: Create(axes),
                            lambda: AnimationGroup(Create(horizontal), Create(vertical), Write(labels)),
                            lambda: AnimationGroup(Create(diagonal), Write(equation)),
                            lambda: TransformMatchingTex(equation, answer)], True)
            return
        if key == "summary":
            lines = VGroup(*[self.formula(step, size=40) for step in beat.steps]).arrange(DOWN, buff=.65)
            self.teach(key, [None, *[lambda line=line: Write(line) for line in lines]])
            return
        super().animate_beat(key, beat)
