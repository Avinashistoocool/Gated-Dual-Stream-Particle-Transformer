"""Review chapter 0: sources are report pages 6-12 and Appendix I."""
from studio import *
from content import CHAPTER_00, SUMMARY_00


class Chapter00(TeachingScene):
    chapter = CHAPTER_00
    chapter_number = 0

    def construct(self):
        # A fixed geometric arrangement illustrates a set; it is not measured data.
        points = [(-1.3,.5),(-.7,1),(-.4,.1),(.2,.8),(.8,.3),(1.2,-.4),(.1,-.8),(-.8,-.6)]
        dots = VGroup(*[Dot([x,y,0], radius=.095, color=COLORS["X"]) for x,y in points])
        outline = Ellipse(width=4.2, height=3.0, color=GREY_B)
        self.teach("recap", [lambda: LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=.12),
                             lambda: Create(outline)])
        self.teach("objective", [lambda: Indicate(dots, color=COLORS["X"]),
                                 lambda: Circumscribe(outline)])
        ten = self.formula(r"\underbrace{(\square,\ldots,\square)}_{\text{ten output scores}}", RIGHT*4, size=34)
        self.teach("jet", [lambda: Indicate(dots[0]), lambda: Indicate(outline),
                           lambda: dots.animate.shift(LEFT*2),
                           lambda: Write(ten)])

        self.wipe()
        matrix = Matrix([[r"\square",r"\cdots",r"\square"],
                         [r"\square",r"\cdots",r"\square"],
                         [r"\vdots",r"\ddots",r"\vdots"]], element_to_mobject_config={"font_size":36})
        matrix.set_color(COLORS["X"]).move_to(LEFT*2)
        feature_caption = self.prose("Each row describes one particle.", DOWN*2.35)
        self.teach("features", [lambda: Write(matrix), lambda: Circumscribe(matrix.get_rows()[0]),
                                lambda: Write(feature_caption), lambda: Indicate(matrix.get_columns()[2])])
        self.wipe()
        a = Dot(LEFT*3, color=COLORS["X"], radius=.13)
        b = Dot(RIGHT*.5+UP, color=COLORS["X"], radius=.13)
        edge = Line(a.get_center(), b.get_center(), color=COLORS["R"])
        pair = self.formula(r"(\square,\square,\square,\square)", DOWN*1.4, COLORS["R"], 40)
        self.teach("pairs", [lambda: FadeIn(a,b), lambda: Create(edge),
                             lambda: Write(pair), lambda: Indicate(pair)])

        self.wipe()
        dims = [
            self.prose("B counts jets in a batch.", UP*1.8),
            self.prose("N counts padded particle slots.", UP*.7),
            self.prose("D counts particle channels.", DOWN*.4),
            self.prose("Dp counts relation channels.", DOWN*1.5),
        ]
        # Only one prose definition is present at a time.
        self.teach("dimensions", [
            lambda: Write(dims[0]),
            lambda: ReplacementTransform(dims[0],dims[1]),
            lambda: ReplacementTransform(dims[1],dims[2]),
            lambda: ReplacementTransform(dims[2],dims[3]),
        ])
        self.wipe()
        x = self.formula(r"X:(B,N,128)", LEFT*3+UP*.7, COLORS["X"])
        r = self.formula(r"R:(B,N,N,64)", RIGHT*3+UP*.7, COLORS["R"])
        dx = self.prose("X represents particle states.", DOWN*1.0, COLORS["X"])
        dr = self.prose("R represents pair relations.", DOWN*1.0, COLORS["R"])
        self.teach("streams", [lambda: AnimationGroup(Write(x),Write(dx)),
                               lambda: AnimationGroup(Write(r),ReplacementTransform(dx,dr)),
                               lambda: Circumscribe(r), lambda: Circumscribe(x)])
        self.shape(r"X:(B,N,128)\quad R:(B,N,N,64)")
        self.wipe()
        xs = VGroup(*[self.formula("X", [(-4+4*i),.8,0],COLORS["X"]) for i in range(3)])
        relations = self.formula("R", DOWN*1.6, COLORS["R"],64)
        arrows = VGroup(*[Arrow(xs[i].get_right(),xs[i+1].get_left(),color=COLORS["X"]) for i in range(2)])
        links = VGroup(*[Arrow(relations.get_top(),s.get_bottom(),color=COLORS["R"],buff=.2) for s in xs])
        self.teach("reuse", [lambda: Write(xs),lambda: Create(arrows),
                             lambda: AnimationGroup(Write(relations),Create(links)),
                             lambda: Indicate(relations, color=COLORS["R"])])
        self.wipe()
        rel = self.formula("R",LEFT*4,COLORS["R"],64)
        gate = self.formula("G",RIGHT+UP,COLORS["G"])
        bias = self.formula(r"\mathsf{B}",RIGHT+DOWN,COLORS["B"])
        gate_def = self.prose("G represents channel gates.",DOWN*2.35,COLORS["G"])
        bias_def = self.prose("B represents additive pair bias.",DOWN*2.35,COLORS["B"])
        l1=Arrow(rel.get_right(),gate.get_left(),color=COLORS["G"])
        l2=Arrow(rel.get_right(),bias.get_left(),color=COLORS["B"])
        self.teach("hybrid", [
            lambda: Write(rel),
            lambda: AnimationGroup(Create(l1),Write(gate),Write(gate_def)),
            lambda: AnimationGroup(Create(l2),Write(bias),ReplacementTransform(gate_def,bias_def)),
            lambda: Circumscribe(VGroup(gate,bias),color=COLORS["R"]),
        ])
        self.wipe()
        # This is a comparison diagram, not a results table.
        comparisons = [
            self.prose("Original ParT is the published architecture.",UP*1.5),
            self.prose("The local baseline disables gates but retains bias.",ORIGIN),
            self.prose("Reported gates are shared; default gates are conditioned.",DOWN*1.5),
        ]
        self.teach("comparison", [
            lambda: Write(comparisons[0]),lambda: ReplacementTransform(comparisons[0],comparisons[1]),
            lambda: ReplacementTransform(comparisons[1],comparisons[2]),lambda: Indicate(comparisons[2])])
        self.wipe()
        route = self.formula(r"\text{Algebra}\to\text{Tensors}\to\text{Attention}\to\text{GDS}",ORIGIN,size=42)
        self.teach("learning",[lambda: Write(route),lambda: Indicate(self.minimap)])
        self.wipe()
        x = self.formula(r"X:(B,N,D)",LEFT*3,COLORS["X"],36)
        r = self.formula(r"R:(B,N,N,D_p)",RIGHT*3,COLORS["R"],36)
        pair_box = SurroundingRectangle(r,color=COLORS["R"])
        self.teach("exercise1",[lambda: Write(VGroup(x,r)),None,None,
                                lambda: Create(pair_box)],equation_pause=True)
        self.wipe()
        reused = self.formula(r"R\ \longrightarrow\ \boxed{1}\ ,\ \boxed{2}\ ,\ \ldots",ORIGIN,COLORS["R"])
        changing = self.formula(r"X\to X\to X",DOWN*1.4,COLORS["X"])
        self.teach("exercise2",[lambda: Write(reused),lambda: Write(changing),
                                lambda: Circumscribe(reused),lambda: Indicate(changing)],equation_pause=True)
        self.wipe()
        summary = VGroup(*[self.prose(line,UP*(1-i)) for i,line in enumerate(SUMMARY_00)])
        # The three-line summary replaces the heading to respect the prose limit.
        self.teach("summary",[lambda: Write(summary[0]),lambda: Write(summary[1]),
                              lambda: AnimationGroup(FadeOut(self.caption),Write(summary[2]))])
        self.finish()
