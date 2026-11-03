"""Unified registry adapter; detailed scene metadata remains chapter-local.

Primary source: docs/GDS-ParT_Technical_Report.tex attention, bias, gates,
expressivity and backpropagation sections. Production RelationalGate and
GatedDualStreamAttention in gds_part_colab.py govern implementation details.
Report-versus-code notes identify masks, denominator clamping and configuration.

Numerical validation:
    py -3.14 docs\\manim_course\\chapter_09_content.py --validate

The shared execute_course, prepare_audio, render_course and finish_course
pipeline owns execution capture, cached neural speech, rendering and delivery.
AttentionLesson adds explicit symbol definitions, semantic colors, probability
bars on a fixed zero-to-one scale, and synchronized equation transformations.
"""
from course_content import b
import chapter_09_content
import chapter_10_content
import chapter_11_content
import chapter_12_content

LESSONS = {9: chapter_09_content, 10: chapter_10_content,
           11: chapter_11_content, 12: chapter_12_content}
CHAPTERS = {}
CODE = {}
EXTRA_SCREEN_TEXT = ["Executed PyTorch output."]
EXTRA_FORMULAS = [r"X", r"R", r"Q", r"K", r"V", r"s_i", r"g_i^Q", r"q_i'",
                  r"r_{ij}", r"\mathcal B_{ij}", r"S_{ij}", r"A_{ij}",
                  r"h_1", r"h_2", r"O", r"A_1", r"A_2"]
for number, lesson in LESSONS.items():
    CHAPTERS[number] = {}
    EXTRA_SCREEN_TEXT.extend(lesson.SUMMARY)
    for key, source in lesson.CODE.items():
        CODE[f"c{number:02}_{key}"] = source
    for block in lesson.BEATS:
        if block["definition"]:
            EXTRA_SCREEN_TEXT.append(block["definition"])
        if block["shape"]:
            EXTRA_FORMULAS.append(block["shape"])
        steps = (f"c{number:02}_{block['code']}",) if block["code"] else tuple(block["steps"])
        kind = "code" if block["code"] else "weights" if block["weights"] else "derive"
        color = "G" if block["diagram"] == "gates" else "B" if block["diagram"] == "bias" else "X"
        CHAPTERS[number][block["key"]] = b(
            block["caption"], block["narration"], *steps, kind=kind, color=color,
            values=tuple(block["weights"] or ()), source=block["source"])
