"""Chapter 14: independent implementation excerpts and executed module tests."""
import json

from chapter_13_complete_forward_pass import *
from chapter_14_source_pages import pages


class Chapter14(FinalTeachingScene):
    chapter_number = 14

    def construct(self):
        pages_file = HERE / "output" / "chapter_14_source_pages.json"
        self.source_pages = pages()
        pages_file.write_text(json.dumps(self.source_pages, indent=2), encoding="utf-8")
        for key, beat in self.beats.items():
            if key.startswith("build_"):
                self.implementation(key)
            elif beat.kind == "code":
                self.code_panel(key, beat.steps[0])
            elif key == "summary":
                self.summary_screen(key, [
                    beat.caption,
                    "Test every module before connecting the full model.",
                    "Audit parameters using the exact configuration.",
                ])
            else:
                self.wipe()
                self.animate_beat(key, beat)
        self.finish()

    def implementation(self, key):
        self.wipe()
        record = self.source_pages[key]
        source = record["source"]
        pages = []
        for page in record["pages"]:
            code = Code(code_string=page, language="python", formatter_style="monokai",
                        add_line_numbers=False, background="rectangle",
                        background_config={"fill_color": "#141F35", "stroke_color": GREY_D},
                        paragraph_config={"font": "Consolas", "font_size": 28})
            if code.width > 12.5 or code.height > 5.2:
                raise ValueError(f"Implementation excerpt overflow: {key}: {code.width}, {code.height}")
            pages.append(code.move_to(UP * .15))
        self.inventory.append({"kind": "implementation", "text": source,
                               "source": record.get("origin", "companion/gds_part_reconstruction.py")})
        first = pages[0]
        actions = [
            lambda: FadeIn(first.background),
            lambda: LaggedStart(*[Write(line) for line in first.code_lines], lag_ratio=.12),
        ]
        for previous, following in zip(pages, pages[1:]):
            actions.append(lambda old=previous, new=following: ReplacementTransform(old, new))
        actions.append(lambda: Circumscribe(VGroup(*pages[-1].code_lines[-2:]), color=COLORS["X"]))
        self.teach(key, actions)
