"""Worked normalization, shared embeddings, and two-stream reuse."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter08(CourseScene):
    chapter_number = 8

    def animate_beat(self, key, beat):
        self.shape(r"X:(B,N,128)\quad R:(B,N,N,64)", COLORS["X"])
        if key in {"dual", "reuse"}:
            nodes = VGroup(*[RoundedRectangle(width=2.0, height=.9, color=COLORS["X"])
                             for _ in range(3)]).arrange(RIGHT, buff=1.1).shift(UP*.3)
            states = VGroup(*[self.formula(rf"X^{{({i})}}", node.get_center(), COLORS["X"], 42)
                              for i, node in enumerate(nodes)])
            edges = VGroup(*[Arrow(nodes[i].get_right(), nodes[i+1].get_left(),
                                  buff=.1, color=COLORS["X"]) for i in range(2)])
            relation = self.formula(r"R:(B,N,N,64)", DOWN*1.7, COLORS["R"], 42)
            feeds = VGroup(*[Arrow(relation.get_top(), node.get_bottom(), buff=.15,
                                  color=COLORS["R"]) for node in nodes])
            shape = self.formula(r"X:(B,N,128)", UP*1.9, COLORS["X"], 40)
            conclusion = self.formula(r"(X,R)\ne(Q,K)" if key == "dual" else
                                      r"R\text{ is computed once per forward pass}", DOWN*2.6, size=34)
            self.teach(key, [
                lambda: AnimationGroup(Create(nodes), Write(states), Write(shape)),
                lambda: AnimationGroup(Write(relation), Create(feeds)),
                lambda: AnimationGroup(Create(edges), Write(conclusion)),
                lambda: LaggedStart(*[Indicate(state, color=COLORS["X"]) for state in states],
                                     lag_ratio=.3),
                lambda: Circumscribe(relation, color=COLORS["R"])], True)
            return
        if key == "summary":
            lines = VGroup(*[self.formula(step, size=38) for step in beat.steps]).arrange(DOWN, buff=.65)
            lines[1].set_color(COLORS["R"])
            self.teach(key, [None, *[lambda line=line: Write(line) for line in lines]])
            return
        super().animate_beat(key, beat)
