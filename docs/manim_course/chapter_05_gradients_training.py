"""Chapter 5: stepwise backpropagation and a checked learning update."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter05(CourseScene):
    chapter_number = 5

    def animate_beat(self, key, beat):
        if key != "update":
            return super().animate_beat(key, beat)
        line = NumberLine(x_range=[0, 6, 1], length=10, include_numbers=True,
                          font_size=30).move_to(DOWN*.8)
        prediction = Dot(line.n2p(2), color=COLORS["X"], radius=.13)
        target = Dot(line.n2p(5), color=COLORS["G"], radius=.13)
        predicted_label = self.formula(r"\widehat y=2", size=34).next_to(prediction, UP, buff=.4)
        target_label = self.formula(r"y=5", size=34, color=COLORS["G"]).next_to(target, UP, buff=.4)
        weight = self.formula(r"w'=1-0.1(-6)=1.6", UP*1.9, size=38)
        bias = self.formula(r"b'=0-0.1(-3)=0.3", UP*.9, size=38)
        updated_label = self.formula(r"\widehat y'=3.5", size=34).next_to(line.n2p(3.5), UP, buff=.4)
        loss = self.formula(r"\mathcal L:4.5\longrightarrow1.125", DOWN*2.35, size=38)
        self.teach(key, [
            lambda: [Create(line), FadeIn(prediction), FadeIn(target),
                     Write(predicted_label), Write(target_label), Write(weight)],
            lambda: Write(bias),
            lambda: [prediction.animate.move_to(line.n2p(3.5)),
                     TransformMatchingShapes(predicted_label, updated_label)],
            lambda: Write(loss),
        ], equation_pause=True)
