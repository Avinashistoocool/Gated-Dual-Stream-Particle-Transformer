"""Chapter 2: addresses, broadcasting, head shapes and masking."""
from course_studio import CourseScene
from manim import *
from content import COLORS


class Chapter02(CourseScene):
    chapter_number = 2

    def animate_beat(self, key, beat):
        if key == "address":
            jets = VGroup(*[
                Matrix([[12*jet+4*row+channel for channel in range(4)] for row in range(3)],
                       element_to_mobject_config={"font_size": 30}, h_buff=.8, v_buff=.65)
                .set_color(COLORS["X"]) for jet in range(2)])
            jets.arrange(RIGHT, buff=1.6).move_to(DOWN*.1)
            labels = VGroup(*[self.formula(f"X[{i}]", size=33).next_to(jet, UP, buff=.3)
                              for i, jet in enumerate(jets)])
            whole = self.formula(r"X:(2,3,4)", DOWN*2.35, size=36)
            one_jet = self.formula(r"X[1]:(3,4)", DOWN*2.35, size=36)
            scalar = self.formula(r"X[1,2,3]=23", DOWN*2.35, size=36)
            row_value = self.formula(r"X[1,2,:]=(20,21,22,23)", DOWN*2.35, size=36)
            selected_jet = SurroundingRectangle(jets[1], color=COLORS["Q"], buff=.12)
            selected_row = SurroundingRectangle(jets[1].get_rows()[2], color=COLORS["Q"], buff=.12)
            selected_cell = SurroundingRectangle(jets[1].get_entries()[11], color=COLORS["G"], buff=.12)
            final_row = selected_row.copy()
            self.teach(key, [
                lambda: [FadeIn(jets), Write(labels), Write(whole)],
                lambda: [Create(selected_jet), TransformMatchingShapes(whole, one_jet)],
                lambda: [Succession(ReplacementTransform(selected_jet, selected_row),
                                    ReplacementTransform(selected_row, selected_cell)),
                         TransformMatchingShapes(one_jet, scalar)],
                lambda: [ReplacementTransform(selected_cell, final_row),
                         TransformMatchingShapes(scalar, row_value)],
                lambda: Circumscribe(row_value, color=COLORS["Q"]),
            ], equation_pause=True)
            return
        if key == "heads":
            groups = VGroup(*[
                VGroup(*[Square(side_length=.075, stroke_width=.6, color=COLORS["X"],
                                fill_color=COLORS["X"], fill_opacity=.7)
                         for _ in range(16)]).arrange(RIGHT, buff=.014)
                for _ in range(8)]).arrange(RIGHT, buff=.12).move_to(DOWN*.2)
            label = self.formula(r"D=128=8\cdot16", UP*1.6)
            shape = self.formula(r"(B,N,128)", DOWN*2.7, size=34)
            targets = groups.copy().arrange_in_grid(rows=4, cols=2, buff=(1.1,.42)).move_to(DOWN*.6)
            split_shape = self.formula(r"(B,N,8,16)", DOWN*2.7, size=34)
            head_shape = self.formula(r"(B,8,N,16)", DOWN*2.7, size=34)
            self.teach(key, [
                lambda: [Write(label), FadeIn(groups), Write(shape)],
                lambda: [groups.animate.become(targets), TransformMatchingShapes(shape, split_shape)],
                lambda: TransformMatchingShapes(split_shape, head_shape),
                lambda: LaggedStart(*[Circumscribe(group, color=COLORS["Q"]) for group in groups], lag_ratio=.1),
            ], equation_pause=True)
            return
        if key == "mean":
            entries = VGroup(*[self.formula(str(value), size=46) for value in (2,6,0)])
            entries.arrange(RIGHT, buff=1.3).move_to(UP*.6)
            mask = VGroup(*[self.formula(str(value), color=COLORS["M"], size=35) for value in (1,1,0)])
            for entry, flag in zip(entries, mask):
                flag.next_to(entry, DOWN, buff=.65)
            numerator = self.formula(r"2+6+0=8", DOWN*1.5)
            answer = self.formula(r"\frac{2+6}{1+1}=4", DOWN*1.5)
            self.teach(key, [
                lambda: [Write(entries), Write(mask)],
                lambda: [entries[2].animate.set_opacity(.2), Write(numerator)],
                lambda: TransformMatchingShapes(numerator, answer),
                lambda: Circumscribe(answer, color=COLORS["Q"]),
            ], equation_pause=True)
            return
        super().animate_beat(key, beat)
