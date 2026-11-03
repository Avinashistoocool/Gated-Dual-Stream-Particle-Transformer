"""Chapter 15: training evidence, worked metrics, and every section 17 caveat."""
from chapter_13_complete_forward_pass import *
from content_13_15 import CHAPTER_15, SUMMARY_15


class Chapter15(FinalTeachingScene):
    chapter = CHAPTER_15
    chapter_number = 15

    def construct(self):
        self.diagram("recap", [r"\text{code}", r"\text{checks}", r"\text{evidence}"])
        self.diagram("objective", [r"\text{train}", r"\text{measure}", r"\text{compare}", r"\text{qualify}"])
        self.diagram("workflow", [r"10{\rm M}", r"\text{validation}", r"\text{select}", r"\text{test}"])
        self.equations("loss", [
            r"t_k=(1-\varepsilon)\mathbf1[k=y]+\varepsilon/10",
            r"\varepsilon=0.1:\quad t_y=0.91,\quad t_{k\ne y}=0.01",
            r"0.91+9(0.01)=1,\qquad \mathcal L=-\sum_k t_k\log p_k",
        ])
        self.diagram("step", [r"\nabla\mathcal L", r"\text{clip}", r"\text{AdamW}", r"\text{EMA}"])
        self.equations("adamw", [
            r"g=2,\quad \widehat m=2,\quad\widehat v=4,\quad\theta=1",
            r"\theta_{\rm next}=(1-\eta\lambda)\theta-\eta\frac{\widehat m}{\sqrt{\widehat v}+\epsilon}",
            r"(1-0.001\cdot0.01)\cdot1-0.001\frac{2}{2}\approx0.99899",
        ])
        self.equations("microbatch", [
            r"B=5,\quad B_1=3,\quad B_2=2,\quad\mathcal L_1=2,\quad\mathcal L_2=5",
            r"\mathcal L=\tfrac35(2)+\tfrac25(5)=3.2",
            r"\tfrac12(2+5)=3.5\quad\times",
        ])
        self.equations("recipe", [
            r"B=512,\quad E=10,\quad \eta_{\max}=10^{-3}",
            r"\lambda=0.01,\quad \beta=(0.9,0.999),\quad C=1",
            r"\text{warmup}=9766,\quad T=195320,\quad\varepsilon_{\rm LS}=0.1",
        ])
        self.ema_graph()
        self.equations("memory", [
            r"\text{activation checkpointing}:\quad\text{recompute during backward}",
            r"\text{disk checkpoint}:\quad(\theta,\bar\theta,\text{optimizer},\text{schedule},t)",
            r"\text{pair memory}=O(BN^2D_p)",
        ])
        self.accuracy_grid()
        self.auc_grid()
        self.equations("macro", [
            r"\binom{10}{2}=\frac{10(9)}2=45",
            r"A_{jk}=\tfrac12[\mathrm{AUC}(j\mid k)+\mathrm{AUC}(k\mid j)]",
            r"\mathrm{AUC}_{\rm macro}^{\rm ovo}=\frac1{45}\sum_{j<k} A_{jk}",
        ])
        self.equations("rejection", [
            r"\epsilon_B=\frac{10}{100000}=10^{-4}",
            r"\mathrm{Rej}=1/\epsilon_B=10000",
            r"\frac{\sigma}{n_{\rm survivors}}\approx\frac1{\sqrt{10}}\approx32\%",
        ])
        self.equations("working_point", [
            r"\mathcal D_S=\frac{p_S}{p_S+p_{\rm QCD}+10^{-12}}",
            r"\epsilon_S=50\%\quad\text{for most signals}",
            r"H\to\ell\nu qq':99\%,\qquad t\to b\ell\nu:99.5\%",
        ])
        self.equations("uncertainty", [
            r"\mathrm{SE}(\mathrm{Acc})\approx\sqrt{\frac{a(1-a)}n}",
            r"\sqrt{\frac{0.8484(1-0.8484)}{1000000}}\approx0.00036",
            r"\text{test counting error}\ne\text{training seed variation}",
        ])
        self.code_panel("code", "c15_metrics")
        self.identities()
        self.results_bars()
        self.equations("uncontrolled", [
            r"\text{shared:}\quad 10{\rm M}\times10=100{\rm M}\ \text{jets seen}",
            r"\text{published ParT:}\quad 10^6\times512=512{\rm M}\ \text{jets seen}",
            r"\text{test: }1{\rm M}\ne20{\rm M},\quad\text{AdamW}\ne\text{Lookahead(RAdam)}",
        ])
        self.equations("validation", [
            r"\text{official pretrained ParT}\longrightarrow\text{local pipeline}",
            r"\mathrm{Acc}=0.8612,\quad \mathrm{AUC}=0.98771,\quad n=200000",
            r"\text{published:}\quad0.861,\quad0.9877,\quad n=20{\rm M}",
        ])
        self.diagram("v1", [r"\text{v1}", r"\Delta\text{model}", r"\Delta\text{training}", r"\text{v2}"])
        self.equations("ablation", [
            r"\text{shared+bias}:0.696588\quad\text{bias only}:0.694242",
            r"\Delta=0.002346=0.2346\ \text{percentage points}",
            r"300000\ \text{jets},\quad2\ \text{epochs},\quad1\ \text{seed}",
        ])
        self.equations("ablation_scope", [
            r"\text{conditioned+bias}:0.693330",
            r"\text{conditioned, no bias}:0.622268",
            r"\text{single seed}\quad\not\Rightarrow\quad\text{measured uncertainty}",
        ])
        self.knockout_bars()
        self.equations("interaction", [
            r"\text{key removed}:0.8302\quad\text{query removed}:0.7232",
            r"\text{both removed}:0.7521",
            r"\Delta_{\rm both}\ne\Delta_Q+\Delta_K",
        ], COLORS["G"])
        self.equations("correlation", [
            r"\operatorname{mean}_c(g_{ic})\quad\ne\quad A_{ij}",
            r"S_{ij}\supset\sum_c q_{ic}k_{jc}g^Q_{ic}g^K_{jc}",
            r"O_i=\sum_j A_{ij}v_j\quad\longrightarrow\quad\text{later layers}",
        ], COLORS["G"])
        self.diagram("limits_evidence", [r"1:\text{ seed}", r"2:\text{ budget}", r"3:\text{ control}"])
        self.diagram("limits_claims", [r"4:\text{ hybrid}", r"5:\text{ survivors}", r"6:\text{ exploratory}"])
        self.diagram("limits_structure", [r"7:R\text{ fixed}", r"8:N^2", r"9:\text{ provenance}"],
                     [COLORS["R"], COLORS["M"], COLORS["B"]])
        self.equations("memory_example", [
            r"128^2\cdot64=1048576\ \text{entries per jet}",
            r"1048576\cdot4=4194304\ \text{bytes}=4\ \mathrm{MiB}",
            r"512\cdot4\ \mathrm{MiB}=2\ \mathrm{GiB}\quad(R\ \text{alone})",
        ], COLORS["R"])
        self.diagram("next_control", [r"\text{full budget}", r"\ge3\text{ seeds}", r"\text{matched}", r"20{\rm M}"])
        self.diagram("next_models", [r"\text{gate }V", r"\text{update }R", r"\text{pair gates}", r"\text{MoE}"],
                     [COLORS["V"], COLORS["R"], COLORS["G"], COLORS["X"]])
        self.equations("exercise1", [
            r"0.696588-0.694242=0.002346",
            r"100(0.002346)=0.2346\ \text{percentage points}",
            r"\text{reproducible superiority?}\quad\text{Not established}",
        ])
        self.equations("exercise2", [
            r"\text{fixed weights:}\quad0.8482\to0.7521",
            r"\text{reliance in fitted model}\quad\checkmark",
            r"\text{retrained control necessarily worse}\quad\times",
        ], COLORS["G"])
        self.equations("exercise3", [
            r"n_B=100000,\quad \mathrm{Rej}=10000",
            r"n_{\rm survivors}\approx n_B/\mathrm{Rej}=10",
            r"\text{relative uncertainty}\approx1/\sqrt{10}\approx32\%",
        ])
        self.summary_screen("summary", SUMMARY_15)
        self.finish()

    def accuracy_grid(self):
        self.wipe()
        true = self.formula(r"y=(0,1,1,1)", UP, size=50)
        predicted = self.formula(r"\hat y=(0,2,1,1)", ORIGIN, size=50)
        correct = self.formula(r"(1,0,1,1)", DOWN, COLORS["Q"], 50)
        result = self.formula(r"\mathrm{Acc}=(1+0+1+1)/4=0.75", DOWN * 2.2, size=45)
        self.teach("accuracy", [lambda: Write(true), lambda: Write(predicted),
                                lambda: TransformFromCopy(VGroup(true, predicted), correct),
                                lambda: Write(result)])

    def auc_grid(self):
        self.wipe()
        scores = self.formula(r"s=(0.9,0.6),\quad b=(0.8,0.2)", UP * 1.7, size=45)
        grid = Matrix([[1, 1], [0, 1]], element_to_mobject_config={"font_size": 42}).move_to(ORIGIN)
        result = self.formula(r"\mathrm{AUC}=\frac{1+1+0+1}{4}=0.75", DOWN * 1.9, size=46)
        self.teach("auc", [lambda: Write(scores), lambda: Create(grid.get_brackets()),
                           lambda: LaggedStart(*[Write(x) for x in grid.get_entries()], lag_ratio=.3),
                           lambda: Write(result)])

    def ema_graph(self):
        self.wipe()
        axes = Axes(x_range=[0, 1200, 400], y_range=[0, 1, .5], x_length=8,
                    y_length=3, tips=False).move_to(DOWN * .3)
        curve = axes.plot(lambda t: .999**t, x_range=[0, 1171], color=COLORS["R"])
        equation = self.formula(r"w_{\rm early}=\rho^{1171}=0.999^{1171}\approx0.31",
                                UP * 2, size=43)
        point = Dot(axes.c2p(1171, .999**1171), color=COLORS["G"])
        label = self.formula(r"t:\text{optimizer steps}", DOWN * 2.5, size=32)
        self.teach("ema", [lambda: AnimationGroup(Create(axes), Write(label)),
                           lambda: Create(curve), lambda: Write(equation),
                           lambda: AnimationGroup(FadeIn(point), Circumscribe(equation))])

    def identities(self):
        self.wipe()
        rows = [
            (r"\text{published ParT}", r"\text{paper architecture and recipe}", TEXT),
            (r"\text{local bias proxy}", r"G_Q=G_K=1,\quad\mathcal B\ne0", COLORS["B"]),
            (r"\text{reported shared hybrid}", r"G(R),\quad\mathcal B(R)", COLORS["G"]),
            (r"\text{default conditioned}", r"G_\ell(R,H^{(\ell)}),\quad\mathcal B(R)", COLORS["R"]),
        ]
        objects = []
        for i, (name, rule, color) in enumerate(rows):
            y = 1.9 - i * 1.2
            objects.append(VGroup(self.formula(name, LEFT * 3.2 + UP * y, color, 30),
                                  self.formula(rule, RIGHT * 2.4 + UP * y, color, 30)))
        self.teach("identities", [lambda obj=obj: Write(obj) for obj in objects])

    def results_bars(self):
        self.wipe()
        axes = Axes(x_range=[0, 1, .2], y_range=[0, 3, 1], x_length=7,
                    y_length=3.6, tips=False).move_to(RIGHT * .6 + DOWN * .2)
        rows = [("v1", .8272, .9824, COLORS["R"]),
                (r"\text{shared v2}", .8484, .9853, COLORS["X"]),
                (r"\text{published ParT}", .850, .9860, COLORS["B"])]
        groups = []
        for i, (name, acc, auc, color) in enumerate(rows):
            y = 2.5 - i
            start = axes.c2p(0, y)
            bar = Line(start, axes.c2p(acc, y), stroke_width=22, color=color)
            label = self.formula(name, start + LEFT * 1.6, color, 27)
            value = self.formula(fr"{acc:.4f}\quad {auc:.4f}", axes.c2p(.6, y) + UP * .35,
                                 size=29)
            groups.append(VGroup(bar, label, value))
        header = self.formula(r"\text{Accuracy}\qquad \text{AUC}", UP * 2.55, size=35)
        self.teach("results", [lambda: AnimationGroup(Create(axes), Write(header)),
                               *[lambda g=g: Create(g) for g in groups]])

    def knockout_bars(self):
        self.wipe()
        base = LEFT * 4.8
        top = Line(base + UP, base + RIGHT * 8.482 + UP, color=COLORS["G"], stroke_width=30)
        low = Line(base + DOWN, base + RIGHT * 7.521 + DOWN, color=COLORS["M"], stroke_width=30)
        before = self.formula(r"\text{trained gates}:0.8482", UP * 1.8, COLORS["G"], 42)
        after = self.formula(r"\text{both replaced by ones}:0.7521", DOWN * 1.8, COLORS["M"], 42)
        gap = self.formula(r"\Delta=-0.0961", RIGHT * 4.5, size=36)
        self.teach("knockout", [lambda: AnimationGroup(Create(top), Write(before)),
                                lambda: AnimationGroup(Create(low), Write(after)),
                                lambda: Write(gap),
                                lambda: Circumscribe(gap, color=COLORS["G"])])
