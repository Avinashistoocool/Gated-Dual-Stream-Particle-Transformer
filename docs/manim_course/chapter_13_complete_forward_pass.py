"""Chapter 13: assemble, calculate, and test the complete forward pass.

This file also provides the private visual vocabulary for chapters 14--15.
No shared studio state is modified.
"""
import json
from pathlib import Path

from studio import *
from course_studio import CourseScene
from content_13_15 import CHAPTER_13, SUMMARY_13


class FinalTeachingScene(CourseScene):
    """Sentence-locked transformations, graph traversal, and executed code."""

    def setup(self):
        super().setup()
        self.map_index = 6
        self.remove(self.map_marker)
        self.map_marker = SurroundingRectangle(self.minimap[6], color=COLORS["X"], buff=.08)
        self.add(self.map_marker)

    def teach(self, key, actions=(), equation_pause=False):
        print(f"Chapter {self.chapter_number:02} | {key} | {self.renderer.time:.2f}s", flush=True)
        index = 6
        if self.chapter_number in (13, 14):
            if key in {"recap", "objective", "inputs"}:
                index = 0
            elif key in {"embed", "particle", "pair", "build_particle", "build_pair"}:
                index = 2
            elif key in {
                "reuse", "normalise", "gates", "attention", "residual",
                "residual_example", "repeat", "equivariance", "build_shared",
                "build_identity", "shared", "build_conditioned", "conditioned",
                "bias", "build_attention", "build_attention_mask",
                "build_attention_mix", "build_block", "block",
            }:
                index = 4
        if index != self.map_index:
            self.remove(self.map_marker)
            self.map_index = index
            self.map_marker = SurroundingRectangle(self.minimap[index], color=COLORS["X"], buff=.08)
            self.add(self.map_marker)
        return super().teach(key, actions, equation_pause)

    def equations(self, key, stages, color=TEXT, note=None):
        self.wipe()
        colors = color if isinstance(color, (list, tuple)) else [color] * len(stages)
        frames = [self.formula(tex, UP * .2, stage_color, 48)
                  for tex, stage_color in zip(stages, colors)]
        actions = [lambda: Write(frames[0])]
        for before, after in zip(frames, frames[1:]):
            actions.append(lambda a=before, b=after: TransformMatchingTex(a, b))
        if note:
            label = self.prose(note, DOWN * 2.35)
            actions.append(lambda: Write(label))
        actions.append(lambda: Circumscribe(frames[-1], color=colors[-1]))
        self.teach(key, actions, equation_pause=True)

    def diagram(self, key, labels, colors=None, branches=False):
        self.wipe()
        colors = colors or [COLORS["X"]] * len(labels)
        nodes = VGroup()
        for label, color in zip(labels, colors):
            box = RoundedRectangle(width=2.35, height=1.0, corner_radius=.12,
                                   stroke_color=color, fill_color=color, fill_opacity=.12)
            text = self.formula(label, color=color, size=34)
            if text.width > 2.1:
                text.scale_to_fit_width(2.1)
            nodes.add(VGroup(box, text))
        nodes.arrange(RIGHT, buff=.65).move_to(UP * .3)
        edges = VGroup(*[Arrow(a.get_right(), b.get_left(), buff=.08, color=colors[i])
                        for i, (a, b) in enumerate(zip(nodes, nodes[1:]))])
        token = Dot(nodes[0].get_center(), color=COLORS["V"], radius=.1)
        actions = [lambda: LaggedStart(*[FadeIn(n) for n in nodes], lag_ratio=.2),
                   lambda: LaggedStart(*[GrowArrow(a) for a in edges], lag_ratio=.2),
                   lambda: Succession(FadeIn(token), *[
                       token.animate.move_to(node.get_center()) for node in nodes[1:]])]
        if branches:
            relation = self.formula("R", DOWN * 1.65, COLORS["R"], 48)
            arrows = VGroup(*[Arrow(relation.get_top(), node.get_bottom(), buff=.15,
                                   color=COLORS["R"]) for node in nodes[1:]])
            actions.append(lambda: AnimationGroup(Write(relation), Create(arrows)))
        else:
            actions.append(lambda: Circumscribe(nodes[-1], color=colors[-1]))
        self.teach(key, actions)

    def code_panel(self, key, example):
        self.wipe()
        outputs = json.loads((HERE / "output" / "course_executed_examples.json").read_text(encoding="utf-8"))
        source, result = self.course_code[example], outputs[example].rstrip()
        code = Code(code_string=source, language="python", formatter_style="monokai",
                    add_line_numbers=False, background="rectangle",
                    background_config={"fill_color": "#141F35", "stroke_color": GREY_D},
                    paragraph_config={"font": "Consolas", "font_size": 28})
        if code.width > 12.4 or code.height > 4.7:
            raise ValueError(f"Code does not fit at 28 pixels: {example}, {code.width}, {code.height}")
        code.move_to(UP * .7)
        output = Text(result, font="Consolas", font_size=28, color=COLORS["X"],
                      line_spacing=.8).move_to(DOWN * 2.5)
        if output.width > 12.5 or output.height > 1.5:
            raise ValueError(f"Output overflow: {example}")
        self.inventory.append({"kind": "code", "text": source, "output": result,
                               "executed_artifact": "course_executed_examples.json", "example": example})
        self.teach(key, [lambda: FadeIn(code.background),
                        lambda: LaggedStart(*[Write(line) for line in code.code_lines], lag_ratio=.12),
                        lambda: Write(output),
                        lambda: Circumscribe(output, color=COLORS["X"])])

    def summary_screen(self, key, lines):
        self.wipe()
        items = [self.prose(line, UP * (.6 - i * 1.2)) for i, line in enumerate(lines[1:])]
        self.teach(key, [lambda: Write(items[0]), lambda: Write(items[1])])


class Chapter13(FinalTeachingScene):
    chapter = CHAPTER_13
    chapter_number = 13

    def construct(self):
        self.diagram("recap", ["X", "R", r"\ell"], [COLORS["X"], COLORS["R"], TEXT])
        self.diagram("objective", ["(F,U,m)", r"X^{(L)}", "c", r"\ell"])
        self.equations("inputs", [
            r"F\in\mathbb R^{B\times N\times17},\quad U\in\mathbb R^{B\times N\times N\times4}",
            r"D=128,\quad D_p=64,\quad H=8,\quad d=D/H=16",
            r"m\in\{0,1\}^{B\times N},\quad 1=\text{real}",
        ])
        self.equations("embed", [
            r"F:(B,N,17)\longrightarrow X^{(0)}:(B,N,128)",
            r"U:(B,N,N,4)\longrightarrow R:(B,N,N,64)",
            r"X^{(0)}=E_X(F),\qquad R=E_R(U)",
        ], [COLORS["X"], COLORS["R"], TEXT])
        self.diagram("reuse", [r"X^{(0)}", r"X^{(1)}", r"X^{(2)}", r"X^{(L)}"], branches=True)
        self.equations("normalise", [
            r"H^{(\ell)}=\mathrm{LN}^{(\ell)}_1(X^{(\ell)})",
            r"H^{(\ell)},X^{(\ell)}\in\mathbb R^{B\times N\times D}",
            r"\widetilde X^{(\ell)}=X^{(\ell)}+\mathrm{Attention}_{\ell}(H^{(\ell)},\ldots)",
        ], COLORS["X"])
        self.equations("gates", [
            r"\text{shared:}\quad (G_Q^{(\ell)},G_K^{(\ell)})=(G_Q,G_K)",
            r"\text{conditioned:}\quad (G_Q^{(\ell)},G_K^{(\ell)})=\mathrm{Gate}_{\ell}(R,H^{(\ell)},m)",
            r"R^{(\ell)}=R,\qquad G^{(\ell+1)}\ \text{may differ}",
        ], COLORS["G"])
        self.attention_path()
        self.equations("residual", [
            r"\widetilde X=X+\mathrm{Attention}(\mathrm{LN}_1(X),G_Q,G_K,\mathcal B,M)",
            r"X_{\rm next}=\widetilde X+\mathrm{FFN}(\mathrm{LN}_2(\widetilde X))",
            r"X\ \longrightarrow\ \widetilde X\ \longrightarrow\ X_{\rm next}",
        ], COLORS["X"])
        self.equations("residual_example", [
            r"X=(2,-1),\quad A=(0.5,1),\quad F=(-0.5,2)",
            r"\widetilde X=(2,-1)+(0.5,1)=(2.5,0)",
            r"X_{\rm next}=(2.5,0)+(-0.5,2)=(2,2)",
        ], COLORS["X"])
        self.diagram("repeat", [r"X^{(0)}", r"X^{(1)}", r"\cdots", r"X^{(8)}"], branches=True)
        self.cls_graph()
        self.pooling_example()
        self.equations("pool_update", [
            r"\widetilde c=c+\mathrm{MHA}(Z_{\rm CLS},Z,Z)",
            r"c_{\rm next}=\widetilde c+\mathrm{FFN}(\mathrm{LN}_2(\widetilde c))",
            r"c^{(0)}\to c^{(1)}\to c^{(2)},\qquad X^{(8)}\ \text{fixed}",
        ])
        self.equations("mask", [
            r"m=(1,1,0)\qquad \text{keep}",
            r"m_{\rm CLS}=(1,1,1,0)",
            r"\text{key padding mask}=\neg m_{\rm CLS}=(0,0,0,1)",
        ], COLORS["M"])
        self.equations("classifier", [
            r"z=\mathrm{LN}_{\rm final}(c^{(2)})\in\mathbb R^{B\times128}",
            r"\ell=W_{\rm out}\mathrm{Drop}(\mathrm{GELU}(W_{\rm hid}z+b_{\rm hid}))+b_{\rm out}",
            r"\ell:(B,10),\qquad p=\mathrm{softmax}(\ell)",
        ])
        self.equations("mean", [
            r"z_i=\mathrm{LN}_{\rm final}(X_i^{(6)})",
            r"z_{\rm jet}=\frac{\sum_i m_i z_i}{\max(1,\sum_i m_i)}",
            r"(B,N,D)\longrightarrow(B,D)\longrightarrow(B,10)",
        ], COLORS["X"])
        self.permutation()
        self.equations("invariance", [
            r"f_{\rm particles}(PX,PRP^\top,Pm)=P f_{\rm particles}(X,R,m)",
            r"f_{\rm jet}(PX,PRP^\top,Pm)=f_{\rm jet}(X,R,m)",
            r"\sum_j a_{\pi(j)}v_{\pi(j)}=\sum_j a_jv_j",
        ])
        self.code_panel("code", "c13_arithmetic")
        self.equations("exercise1", [
            r"B=2,\quad N=5,\quad D=128,\quad C=10",
            r"X^{(8)}:(2,5,128)",
            r"c:(2,1,128)\quad\longrightarrow\quad\ell:(2,10)",
        ])
        self.equations("exercise2", [
            r"X'=PX,\quad m'=Pm,\quad R'=PR\quad \times",
            r"R'_{ij}=R_{\pi(i),\pi(j)}",
            r"R'=PRP^\top\quad\checkmark",
        ], COLORS["R"])
        self.equations("exercise3", [
            r"m_{\rm CLS}=(1,1,1,0)\quad\Rightarrow\quad\mathrm{ignore}=(0,0,0,1)",
            r"c=(1,1),\quad \mathrm{message}=(2,1)",
            r"\widetilde c=(1,1)+(2,1)=(3,2)",
        ])
        self.summary_screen("summary", SUMMARY_13)
        self.finish()

    def attention_path(self):
        self.wipe()
        query = MathTex(r"Q'", "=", "Q", r"\odot", "G_Q", font_size=45).move_to(LEFT * 3 + UP)
        query[0].set_color(COLORS["Q"])
        query[2].set_color(COLORS["Q"])
        query[4].set_color(COLORS["G"])
        key = MathTex(r"K'", "=", "K", r"\odot", "G_K", font_size=45).move_to(RIGHT * 3 + UP)
        key[0].set_color(COLORS["K"])
        key[2].set_color(COLORS["K"])
        key[4].set_color(COLORS["G"])
        scores = MathTex("S", "=", "Q'", r"K'^\top", "/4+", r"\mathcal B", "+", "M",
                         font_size=50).move_to(DOWN * .2)
        for index, name in ((2, "Q"), (3, "K"), (5, "B"), (7, "M")):
            scores[index].set_color(COLORS[name])
        self.inventory.extend({"kind": "math", "text": text} for text in
                              (r"Q'=Q\odot G_Q", r"K'=K\odot G_K", r"S=Q'K'^\top/4+\mathcal B+M"))
        result = self.formula(r"A=\mathrm{softmax}_j(S),\qquad O=AV", DOWN * 1.7, COLORS["V"], 45)
        self.teach("attention", [lambda: AnimationGroup(Write(query), Write(key)),
                                 lambda: TransformFromCopy(VGroup(query, key), scores),
                                 lambda: Write(result),
                                 lambda: Circumscribe(result, color=COLORS["V"])])

    def cls_graph(self):
        self.wipe()
        particles = VGroup(*[Square(.8, color=COLORS["X"], fill_opacity=.2)
                            for _ in range(3)]).arrange(RIGHT, buff=.7).move_to(DOWN * .5)
        cls = Square(.9, color=COLORS["Q"], fill_opacity=.2).move_to(UP * 1.35)
        label = self.formula("c", cls.get_center(), COLORS["Q"], 40)
        arrows = VGroup(*[Arrow(p.get_top(), cls.get_bottom(), buff=.1, color=COLORS["V"])
                         for p in particles])
        selfedge = CurvedArrow(cls.get_right(), cls.get_top(), angle=PI * 1.5, color=COLORS["V"])
        shape = self.formula(r"Q:(B,H,1,d),\quad K,V:(B,H,N+1,d)", DOWN * 2.2, size=37)
        self.teach("cls", [lambda: AnimationGroup(FadeIn(cls), Write(label)),
                           lambda: FadeIn(particles),
                           lambda: AnimationGroup(Create(arrows), Create(selfedge)),
                           lambda: Write(shape)])

    def pooling_example(self):
        self.wipe()
        values = [self.formula(tex, LEFT * (4 - i * 4) + UP * .7, COLORS["V"], 45)
                  for i, tex in enumerate([r"\binom{0}{2}", r"\binom{2}{0}", r"\binom{4}{2}"])]
        weighted = [self.formula(tex, value.get_center(), COLORS["V"], 45)
                    for tex, value in zip([r"\binom{0}{0.5}", r"\binom{1}{0}", r"\binom{1}{0.5}"], values)]
        weights = VGroup(*[self.formula(tex, value.get_center() + UP * 1.1, size=35)
                           for tex, value in zip([r"\tfrac14", r"\tfrac12", r"\tfrac14"], values)])
        total = self.formula(r"\binom{0}{0.5}+\binom{1}{0}+\binom{1}{0.5}=\binom{2}{1}",
                             DOWN * 1.5, COLORS["V"], 44)
        self.teach("pool_example", [
            lambda: LaggedStart(*[Write(v) for v in values], lag_ratio=.2),
            lambda: Write(weights),
            lambda: AnimationGroup(*[TransformMatchingTex(a, b) for a, b in zip(values, weighted)]),
            lambda: TransformFromCopy(VGroup(*weighted), total)])

    def permutation(self):
        self.wipe()
        cells = VGroup(*[Square(.65, color=COLORS["R"], fill_opacity=.12)
                        for _ in range(9)]).arrange_in_grid(3, 3, buff=.08).move_to(LEFT * 2.6)
        entries = VGroup(*[self.formula(f"r_{{{i+1}{j+1}}}", cells[3*i+j].get_center(),
                                       COLORS["R"], 26) for i in range(3) for j in range(3)])
        rule = self.formula(r"(1,2,3)\longrightarrow(3,1,2)", RIGHT * 2.5 + UP, size=35)
        equation = self.formula(r"S'=PSP^\top", RIGHT * 2.5 + DOWN * .7, COLORS["B"], 45)
        perm = [2, 0, 1]
        destinations = {(i, j): cells[3*perm.index(i)+perm.index(j)].get_center()
                        for i in range(3) for j in range(3)}
        self.teach("equivariance", [
            lambda: AnimationGroup(Create(cells), Write(entries)),
            lambda: Write(rule),
            lambda: AnimationGroup(*[entries[3*i+j].animate.move_to(destinations[i, j])
                                      for i in range(3) for j in range(3)]),
            lambda: Write(equation)])
