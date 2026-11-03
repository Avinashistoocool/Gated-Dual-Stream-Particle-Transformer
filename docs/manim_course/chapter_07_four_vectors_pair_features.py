"""Four-vector geometry and the complete symmetric pair calculation."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter07(CourseScene):
    chapter_number = 7

    def animate_beat(self, key, beat):
        self.shape(r"v:(B,N,4)\quad U:(B,N,N,4)", COLORS["R"])
        if key == "pt":
            axes = Axes(x_range=[0, 5, 1], y_range=[0, 5, 1],
                        x_length=3.6, y_length=3.6, tips=False).shift(LEFT*2+DOWN*.2)
            origin, elbow, tip = axes.c2p(0, 0), axes.c2p(3, 0), axes.c2p(3, 4)
            x = Line(origin, elbow, color=COLORS["X"])
            y = Line(elbow, tip, color=COLORS["R"])
            vector = Arrow(origin, tip, buff=0, color=COLORS["G"])
            labels = VGroup(self.formula("p_x=3", size=30).next_to(x, DOWN),
                            self.formula("p_y=4", size=30).next_to(y, RIGHT))
            first = self.formula(r"p_T^2=3^2+4^2", RIGHT*3+UP*.5, size=40)
            final = self.formula(r"p_T=\sqrt{25}=5", RIGHT*3+DOWN*.5, size=40)
            self.teach(key, [lambda: Create(axes),
                            lambda: AnimationGroup(Create(x), Create(y), Write(labels)),
                            lambda: AnimationGroup(GrowArrow(vector), Write(first)),
                            lambda: Write(final)], True)
            return
        if key == "wrap":
            circle = Circle(radius=1.6, color=GRAY).shift(LEFT*2)
            origin = circle.get_center()
            ray1 = Line(origin, origin+1.6*np.array([np.cos(3.1), np.sin(3.1), 0]),
                        color=COLORS["X"])
            ray2 = Line(origin, origin+1.6*np.array([np.cos(-3.1), np.sin(-3.1), 0]),
                        color=COLORS["R"])
            long_arc = Arc(radius=1.2, start_angle=-3.1, angle=6.2,
                           arc_center=origin, color=COLORS["G"])
            short_arc = Arc(radius=1.2, start_angle=3.1, angle=2*PI-6.2,
                            arc_center=origin, color=COLORS["B"])
            naive = self.formula(r"\Delta\phi_{\rm naive}=6.2", RIGHT*2.7+UP*.6, size=38)
            wrapped = self.formula(r"\delta\phi=-0.083185", RIGHT*2.7+DOWN*.6, size=38)
            self.teach(key, [lambda: AnimationGroup(Create(circle), Create(ray1), Create(ray2)),
                            lambda: AnimationGroup(Create(long_arc), Write(naive)),
                            lambda: AnimationGroup(ReplacementTransform(long_arc, short_arc), Write(wrapped)),
                            lambda: Circumscribe(wrapped, color=COLORS["B"])], True)
            return
        if key == "summary":
            lines = VGroup(*[self.formula(step, size=38, color=COLORS["R"])
                            for step in beat.steps]).arrange(DOWN, buff=.65)
            self.teach(key, [None, *[lambda line=line: Write(line) for line in lines]])
            return
        super().animate_beat(key, beat)
