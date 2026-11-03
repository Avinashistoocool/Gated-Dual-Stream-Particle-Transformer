"""Shared Manim layout and word-boundary-locked neural voiceover."""
from __future__ import annotations

import json
import re
from pathlib import Path

from prepare_audio import CACHE, audio_id, plain, setup_tools
setup_tools()

from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.base import SpeechService
from content import COLORS, CODE

HERE = Path(__file__).resolve().parent
BACKGROUND = "#0B1020"
TEXT = "#EEF3FA"
FONT = "Segoe UI"


class CachedNeuralService(SpeechService):
    """No network calls during rendering; missing narration is a hard failure."""

    def __init__(self):
        super().__init__(cache_dir=str(CACHE), global_speed=1, transcription_model=None)

    def generate_from_text(self, text, cache_dir=None, path=None):
        filename = CACHE / f"{audio_id(text)}.json"
        if not filename.exists():
            raise FileNotFoundError(f"Run prepare_audio.py first. Missing {filename.name}")
        data = json.loads(filename.read_text(encoding="utf-8"))
        if data["text"] != plain(text):
            raise ValueError("Narration cache does not match the scene script")
        return {"input_text": text, "input_data": {"text": text, "service": "cached-edge"},
                "original_audio": data["original_audio"],
                "word_boundaries": data["word_boundaries"]}

    def _wrap_generate_from_text(self, text, path=None, **kwargs):
        # All files are immutable, pre-paced assets. The upstream wrapper appends
        # a shared JSON index non-atomically, which corrupts parallel renders.
        data = self.generate_from_text(" ".join(text.split()), path=path)
        data["final_audio"] = data["original_audio"]
        return data


class TeachingScene(VoiceoverScene):
    chapter = None
    chapter_number = -1

    def add_sound(self, sound_file, *args, **kwargs):
        # Voiceover 0.3.7 measures MP3 duration; Manim mixes the matching lossless PCM.
        path = Path(sound_file)
        if path.name.endswith("-paced.mp3"):
            pcm = path.with_name(path.name.replace("-paced.mp3", ".wav"))
            if not pcm.is_file():
                raise FileNotFoundError(pcm)
            sound_file = str(pcm)
        return super().add_sound(sound_file, *args, **kwargs)

    def setup(self):
        super().setup()
        if config.dry_run:
            # Manim's dry_run setter clears save_last_frame. Voiceover restores
            # this original flag, so both must explicitly skip frame sampling.
            self.renderer.skip_animations = True
            self.renderer._original_skipping_status = True
        self.camera.background_color = BACKGROUND
        self.set_speech_service(CachedNeuralService(), create_subcaption=False)
        self.caption = None
        self.tag = None
        self.pending_clear = []
        self.timeline = []
        self.inventory = []
        self.minimap = MathTex(
            r"\text{Inputs}", r"\longrightarrow", r"\text{Embeddings}", r"\longrightarrow",
            r"\text{Attention}", r"\longrightarrow", r"\text{Prediction}", font_size=28,
            color=GREY_B).to_edge(DOWN, buff=.20)
        self.map_index = 0
        self.map_marker = SurroundingRectangle(self.minimap[0], color=COLORS["X"], buff=.08)
        self.add(self.minimap, self.map_marker)

    def formula(self, tex, position=ORIGIN, color=TEXT, size=48):
        self.inventory.append({"kind": "math", "text": tex})
        item = MathTex(tex, font_size=size, color=color)
        if item.width > 12.3:
            item.scale_to_fit_width(12.3)
        return item.move_to(position)

    def prose(self, text, position=ORIGIN, color=TEXT):
        if len(text.split()) > 12 or not text.endswith((".", "?")):
            raise ValueError(f"Invalid screen sentence: {text}")
        self.inventory.append({"kind": "prose", "text": text})
        item = Text(text, font=FONT, font_size=30, color=color)
        if item.width > 12.8:
            raise ValueError(f"Screen sentence too wide: {text}")
        return item.move_to(position)

    def shape(self, tex, color=TEXT):
        if self.tag is not None:
            self.remove(self.tag)
        self.tag = self.formula(tex, color=color, size=28).to_corner(UR, buff=.38).shift(DOWN * .65)
        self.add(self.tag)

    def definition(self, text, position=DOWN * 2.45, color=TEXT):
        return self.prose(text, position, color)

    def wipe(self):
        keep = [self.caption, self.tag, self.minimap, self.map_marker]
        self.pending_clear = [m for m in self.mobjects if all(m is not k for k in keep)]

    def teach(self, key, actions=(), equation_pause=False):
        caption, text, source = self.chapter[key]
        sentences = re.split(r"(?<=[.!?])\s+", text)
        marked = " ".join(f"<bookmark mark='s{i}'/>{s}" for i, s in enumerate(sentences))
        new_caption = self.prose(caption, UP * 3.48)
        old_caption = self.caption
        self.caption = new_caption
        with self.voiceover(text=marked) as tracker:
            start = tracker.start_t
            if old_caption is not None:
                self.remove(old_caption)
            self.add(new_caption)
            map_index = (self.map_index if self.chapter_number >= 2 else
                         4 if key in {"hybrid", "gds", "exercise_shape", "hadamard_result"} else
                         (2 if self.chapter_number == 1 else 0))
            if map_index != self.map_index:
                marker = SurroundingRectangle(self.minimap[map_index], color=COLORS["X"], buff=.08)
                self.play(ReplacementTransform(self.map_marker, marker), run_time=.2)
                self.map_marker, self.map_index = marker, map_index
            if self.pending_clear:
                self.play(*[FadeOut(m) for m in self.pending_clear], run_time=.25)
                self.pending_clear = []
            for i, _ in enumerate(sentences):
                self.wait_until_bookmark(f"s{i}")
                if i < len(actions) and actions[i] is not None:
                    animation = actions[i]()
                    if animation is not None:
                        if not isinstance(animation, (list, tuple)):
                            animation = [animation]
                        remaining = tracker.time_until_bookmark(f"s{i+1}") if i+1 < len(sentences) else tracker.get_remaining_duration()
                        self.play(*animation, run_time=max(.15, min(1.6, remaining - .1)))
            self.wait_for_voiceover()
            if key == "recap" and self.renderer.time - start < 20:
                self.wait(20 - (self.renderer.time - start))
            if equation_pause:
                self.wait(.75)
        audio = json.loads((CACHE / f"{audio_id(text)}.json").read_text(encoding="utf-8"))
        self.timeline.append({"key": key, "start": start, "audio_duration": audio["duration"],
                              "end": self.renderer.time, "text": text, "source": source,
                              "boundaries": audio["word_boundaries"]})
        self.check_layout(key)
        print(f"CHAPTER {self.chapter_number:02} BEAT {key}: timeline {self.renderer.time:.2f}s",
              flush=True)

    def check_layout(self, key):
        for item in self.mobjects:
            if item.width == 0:
                continue
            if item.get_left()[0] < -7.12 or item.get_right()[0] > 7.12:
                raise ValueError(f"Horizontal overflow in chapter {self.chapter_number}, {key}")
            if item.get_top()[1] > 4.01 or item.get_bottom()[1] < -4.01:
                raise ValueError(f"Vertical overflow in chapter {self.chapter_number}, {key}")

    def code_example(self, key, code_key):
        self.wipe()
        source = CODE[code_key]
        output_file = HERE / "output" / "executed_examples.json"
        outputs = json.loads(output_file.read_text(encoding="utf-8"))
        code = Code(code_string=source, language="python", formatter_style="monokai",
                    add_line_numbers=False, background="rectangle",
                    background_config={"fill_color": "#141F35", "stroke_color": GREY_D},
                    paragraph_config={"font": "Consolas", "font_size": 28})
        if code.width > 7.6:
            raise ValueError(f"Code does not fit at readable font size: {code_key}")
        code.move_to(LEFT * 2.8)
        output = Text(outputs[code_key].rstrip(), font="Consolas", font_size=28,
                      color=COLORS["X"], line_spacing=.8).move_to(RIGHT * 4.1)
        if output.width > 5.2:
            raise ValueError(f"Code output too wide: {code_key}")
        self.inventory.append({"kind": "code", "text": source, "output": outputs[code_key]})
        self.teach(key, [
            lambda: FadeIn(code.background),
            lambda: LaggedStart(*[Write(line) for line in code.code_lines], lag_ratio=.18),
            lambda: Write(output),
            lambda: Circumscribe(output, color=COLORS["X"]),
        ], equation_pause=True)

    def finish(self):
        out = HERE / "output"
        out.mkdir(exist_ok=True)
        (out / f"chapter_{self.chapter_number:02}_timeline.json").write_text(
            json.dumps(self.timeline, indent=2), encoding="utf-8")
        (out / f"chapter_{self.chapter_number:02}_screen_inventory.json").write_text(
            json.dumps(self.inventory, indent=2), encoding="utf-8")
