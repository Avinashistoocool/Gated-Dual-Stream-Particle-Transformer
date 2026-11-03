"""Offline course renderer. Run with the Python environment described in README."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from piper import PiperVoice, SynthesisConfig

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
OUT = HERE / "output"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1600, 900
BG, PANEL = "#0a1020", "#141f35"
WHITE, MUTED, CYAN, GOLD = "#edf4ff", "#9eb2cc", "#49dfd0", "#ffcb75"
FONTS = Path(r"C:\Windows\Fonts")


def font(size, mono=False, bold=False):
    name = "consola.ttf" if mono else ("segoeuib.ttf" if bold else "segoeui.ttf")
    return ImageFont.truetype(str(FONTS / name), size)


def parse():
    chapters = []
    scene = None
    for line in (HERE / "course.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("@chapter "):
            chapters.append({"title": line[9:], "scenes": []})
        elif line.startswith("@scene "):
            scene = {"title": line[7:], "points": [], "speech": [], "code": [],
                     "math": "", "visual": "flow", "practice": 0, "answer": ""}
            chapters[-1]["scenes"].append(scene)
        elif scene is None:
            raise ValueError(f"Content before first scene: {line}")
        elif line.startswith("@point "):
            scene["points"].append(line[7:])
        elif line.startswith("@say "):
            scene["speech"].append(line[5:])
        elif line.startswith("@code "):
            scene["code"].append(line[6:])
        elif line.startswith("@math "):
            scene["math"] = line[6:]
        elif line.startswith("@visual "):
            scene["visual"] = line[8:]
        elif line.startswith("@practice "):
            scene["practice"] = int(line[10:])
        elif line.startswith("@answer "):
            scene["answer"] = line[8:]
        else:
            raise ValueError(f"Unknown directive: {line}")
    for chapter in chapters:
        for scene in chapter["scenes"]:
            if not scene["speech"] or not scene["points"]:
                raise ValueError(f"Incomplete scene: {scene['title']}")
            if scene["practice"] and not scene["answer"]:
                raise ValueError(f"Practice without answer: {scene['title']}")
    return chapters


def wrap(draw, text, xy, width, size=28, fill=WHITE, mono=False, bold=False):
    f = font(size, mono, bold)
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=f) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += size + 10
    return y


def diagram(draw, kind, stage, chapter_index):
    x, y = 880, 300
    if kind in {"attention", "gate", "mask"}:
        values = np.array([[.70, .20, .10], [.15, .55, .30], [.20, .25, .55]])
        if kind == "mask":
            values = np.array([[.78, .22, 0], [.21, .79, 0], [.5, .5, 0]])
        draw.text((x, y - 45), "query i  ->  key j", font=font(26, mono=True), fill=MUTED)
        for i in range(3):
            for j in range(3):
                xx, yy = x + j * 135, y + i * 95
                v = float(values[i, j])
                color = (int(22 + 35 * v), int(45 + 150 * v), int(65 + 140 * v))
                draw.rounded_rectangle((xx, yy, xx + 120, yy + 80), 12, fill=color,
                                       outline=GOLD if i == stage % 3 else None, width=3)
                draw.text((xx + 20, yy + 20), f"{v:.2f}", font=font(29, mono=True), fill=WHITE)
        label = "Illustrative probabilities, NOT trained attention"
        wrap(draw, label, (x, y + 310), 550, 23, MUTED)
        if kind == "gate":
            wrap(draw, "Q * gQ   |   K * gK\nV is not directly gated", (x, y + 390), 550, 29, CYAN)
    elif kind == "tensor":
        for b in range(3):
            xx, yy = x + b * 38, y + b * 45
            draw.rounded_rectangle((xx, yy, xx + 390, yy + 240), 14, fill=PANEL,
                                   outline=CYAN if b == stage % 3 else MUTED, width=3)
            for i in range(1, 4):
                draw.line((xx, yy + i * 60, xx + 390, yy + i * 60), fill=MUTED)
            for j in range(1, 6):
                draw.line((xx + j * 65, yy, xx + j * 65, yy + 240), fill=MUTED)
        wrap(draw, "Batch / particles / channels\nAxes have meanings, not just sizes.", (x, y + 370), 550, 28, CYAN)
    elif kind == "jets":
        center = (x + 240, y + 180)
        for k in range(15):
            angle = k * 2.4
            radius = 35 + (k * 37) % 170
            xx, yy = center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius
            draw.line((*center, xx, yy), fill="#33465c", width=2)
            r = 7 + k % 5
            draw.ellipse((xx-r, yy-r, xx+r, yy+r), fill=GOLD if k % 3 == stage % 3 else CYAN)
        wrap(draw, "A jet is a set of particles.\nPair relations connect the set.", (x, y + 390), 550, 28, CYAN)
    else:
        flows = {
            0: ["School mathematics", "ML building blocks", "Particle + relation streams", "Reconstruct and test"],
            1: ["Define the variables", "Apply the function", "Check the output domain", "Compose the next operation"],
            2: ["Name every axis", "Transform channels", "Track each index", "Verify shapes AND values"],
            3: ["Forward prediction", "Compute scalar loss", "Backward: chain rule", "Optimizer updates parameters"],
            4: ["Token vector x", "Center and normalize", "Learned scale + shift", "Residual: preserve x + update"],
            5: ["Raw particle observations", "Fixed feature transforms", "17-channel particle vector", "Learned particle embedding"],
            6: ["Particle four-vectors", "Four symmetric pair features", "Pair embedding MLP", "Relations R: [B,N,N,64]"],
            9: ["Pair features U", "Shared relation encoder", "Projection: 64 -> 8 heads", "Add to attention logits"],
            10: ["Relation summary s_i", "Separate query/key MLPs", "1 + tanh(logits)", "Scale Q and K channels"],
            11: ["h = LayerNorm(x)", "Conditioned gate + attention", "x1 = x + attention update", "x2 = x1 + FFN(LN(x1))"],
            12: ["Contextual particle states", "CLS reads the particle set", "Normalize pooled state", "Classifier -> ten logits"],
            13: ["Register the components", "Assemble the forward path", "Load identical weights", "Compare outputs + gradients"],
            14: ["Verify tiny synthetic batch", "Prepare real data splits", "Train and select on validation", "Evaluate held-out test data"],
            15: ["State a precise hypothesis", "Choose a controlled comparison", "Measure and quantify uncertainty", "Report conclusion + limitations"],
        }
        labels = flows.get(chapter_index, ["Projected queries and keys", "Scaled dot-product scores", "Masked softmax over keys", "Weighted mixture of values"])
        for i, label in enumerate(labels):
            yy = y + i * 94
            draw.rounded_rectangle((x, yy, x + 510, yy + 66), 12, fill=PANEL,
                                   outline=CYAN if i == stage % 4 else "#35455f", width=3)
            draw.text((x + 22, yy + 16), label, font=font(25), fill=WHITE)
            if i < 3:
                draw.line((x + 250, yy + 68, x + 250, yy + 92), fill=GOLD, width=3)
        wrap(draw, "Follow the operation, then verify its meaning.", (x, y + 400), 550, 25, GOLD)


def slide(chapter, scene, ci, si, stage, practice=False):
    image = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, W, 8), fill=CYAN)
    d.text((64, 34), "FROM CLASS 12 TO GDS-ParT", font=font(22, bold=True), fill=CYAN)
    d.text((1220, 34), f"LESSON {ci+1:02d} / {si+1:02d}", font=font(21, mono=True), fill=MUTED)
    d.text((64, 83), chapter["title"], font=font(24), fill=MUTED)
    bottom = wrap(d, scene["title"], (60, 127), 1460, 44, WHITE, bold=True)
    if bottom > 250:
        raise ValueError(f"Title overflow: {scene['title']}")
    d.rounded_rectangle((48, 260, 810, 760), 20, fill=PANEL)
    y = 285
    for i, point in enumerate(scene["points"]):
        d.ellipse((72, y + 10, 82, y + 20), fill=GOLD if i == stage % len(scene["points"]) else CYAN)
        y = wrap(d, point, (102, y), 660, 27, WHITE if i <= stage or practice else MUTED) + 22
    if y > 755:
        raise ValueError(f"Bullet overflow: {scene['title']} ({y})")
    if scene["code"]:
        y = 280
        code_size = 24
        while max(d.textlength(line, font=font(code_size, mono=True)) for line in scene["code"]) > 675:
            code_size -= 1
            if code_size < 19:
                raise ValueError(f"Code overflow: {scene['title']}")
        for line in scene["code"]:
            d.text((860, y), line, font=font(code_size, mono=True), fill=CYAN)
            y += 37
        if y > 750:
            raise ValueError(f"Code block too tall: {scene['title']}")
    else:
        diagram(d, scene["visual"], stage, ci)
    if scene["math"]:
        y = wrap(d, scene["math"], (64, 785), 1460, 26, GOLD, mono=True)
        if y > 872:
            raise ValueError(f"Equation overflow: {scene['title']}")
    else:
        d.text((64, 805), "Understand -> calculate -> implement -> check", font=font(25), fill=MUTED)
    if practice:
        d.rounded_rectangle((940, 680, 1490, 746), 14, fill="#493b22")
        d.text((966, 698), "YOUR TURN - pause / work / resume", font=font(25, bold=True), fill=GOLD)
    return image


def run(args):
    p = subprocess.run([str(x) for x in args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        raise RuntimeError(f"Command failed: {args}\n{p.stderr[-12000:]}")
    return p


def seconds(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def synth(voice, text, length_scale):
    digest = hashlib.sha256((f"piper-1.4.2-pratham-{length_scale}-" + text).encode()).hexdigest()[:20]
    path = BUILD / f"voice-{digest}.wav"
    if not path.exists():
        with wave.open(str(path), "wb") as wav:
            voice.synthesize_wav(text, wav, SynthesisConfig(length_scale=length_scale, noise_scale=.45))
    return path


def timestamp(t, srt=False):
    ms = round(t * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{',' if srt else '.'}{ms:03}"


def concat_file(paths, target):
    target.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in paths), encoding="utf-8")


def write_player(movie, chapters, subtitles):
    links = "\n".join(f'<button onclick="jumpTo({c["start"]})">'
                      f'{timestamp(c["start"])} &nbsp; {html.escape(c["title"])}</button>' for c in chapters)
    cues = json.dumps(subtitles, ensure_ascii=True).replace("<", "\\u003c")
    (OUT / "watch.html").write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,"><title>Class 12 to GDS-ParT</title><style>
body{{background:#0a1020;color:#edf4ff;font:18px system-ui;max-width:1200px;margin:40px auto;padding:20px}}
video{{width:100%;border-radius:16px}} button{{display:block;background:#141f35;color:#49dfd0;
border:1px solid #33465c;border-radius:8px;padding:12px;margin:8px 0;width:100%;text-align:left;cursor:pointer}}
a{{color:#ffcb75}}</style><h1>Class 12 to GDS-ParT</h1>
<p>Hinglish synthetic narration. English diagrams. Guided coding pauses.
Use the player's speed and subtitle controls. This player works directly from your local folder.</p>
<video id="v" controls preload="metadata"><source src="{html.escape(movie.name)}" type="video/mp4"></video>
<p><a href="transcript.txt">Transcript</a> | <a href="course.srt">Subtitles</a></p>{links}
<script>
const video = document.getElementById('v');
const captions = video.addTextTrack('subtitles', 'Hinglish', 'hi');
for (const [start, end, text] of {cues}) captions.addCue(new VTTCue(start, end, text));
captions.mode = 'showing';
function jumpTo(time) {{ video.currentTime = time; video.play(); }}
</script></html>""", encoding="utf-8")


def existing_subtitles():
    def time_value(text):
        h, m, s = text.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    cues = []
    for block in (OUT / "course.srt").read_text(encoding="utf-8").strip().split("\n\n"):
        _, timing, text = block.split("\n", 2)
        start, end = timing.split(" --> ")
        cues.append((time_value(start), time_value(end), text))
    return cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="Render just the first scene")
    ap.add_argument("--slides-only", action="store_true")
    ap.add_argument("--player-only", action="store_true", help="Refresh the player for an existing completed render")
    ap.add_argument("--length-scale", type=float, default=1.16)
    ap.add_argument("--min-minutes", type=float, default=120)
    args = ap.parse_args()
    if args.player_only:
        manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
        movie = OUT / manifest["movie"]
        if not movie.is_file():
            raise FileNotFoundError(movie)
        write_player(movie, manifest["chapters"], existing_subtitles())
        print("Local player refreshed with self-contained captions.")
        return
    BUILD.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    chapters = parse()
    source_hash = hashlib.sha256((HERE / "course.txt").read_bytes() + Path(__file__).read_bytes()).hexdigest()
    voice = None if args.slides_only else PiperVoice.load(str(HERE / "voices" / "hi_IN-pratham-medium.onnx"))
    audio_path = BUILD / "course-audio.wav"
    course_audio = None
    if not args.slides_only:
        course_audio = wave.open(str(audio_path), "wb")
        course_audio.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
    all_clips, chapter_records, transcript, subtitles = [], [], [], []
    elapsed, speech_time, practice_time = 0.0, 0.0, 0.0
    for ci, chapter in enumerate(chapters):
        start = elapsed
        chapter_clips = []
        transcript.append(f"\nLESSON {ci+1}: {chapter['title']}\n")
        for si, scene in enumerate(chapter["scenes"]):
            transcript.append(f"\n{scene['title']}\n")
            for stage in range(max(3, len(scene["speech"]))):
                slide(chapter, scene, ci, si, stage).save(BUILD / f"slide-{ci:02}-{si:02}-{stage}.png")
            if args.slides_only:
                continue
            pieces = [(text, False) for text in scene["speech"]]
            if scene["practice"]:
                pieces += [("", True), (scene["answer"], False)]
            for pi, (text, practice) in enumerate(pieces):
                if practice:
                    duration = scene["practice"]
                    practice_time += duration
                    course_audio.writeframes(b"\0" * (round(duration * 22050) * 2))
                    transcript.append(f"[Practice: {duration} seconds. {scene['points'][-1]}]")
                else:
                    # Sentence-sized utterances give readable, precisely bounded caption cues.
                    sentences = re.split(r"(?<=[.!?।])\s+", text)
                    audio_parts = [synth(voice, sentence, args.length_scale) for sentence in sentences if sentence.strip()]
                    spoken_duration = sum(seconds(p) for p in audio_parts)
                    duration = math.ceil(spoken_duration * 5) / 5
                    samples = 0
                    for part in audio_parts:
                        with wave.open(str(part), "rb") as wav:
                            if wav.getparams()[:3] != (1, 2, 22050):
                                raise ValueError(f"Unexpected voice audio format: {part}")
                            samples += wav.getnframes()
                            course_audio.writeframes(wav.readframes(wav.getnframes()))
                    course_audio.writeframes(b"\0" * (round(duration * 22050) - samples) * 2)
                    cursor = elapsed
                    for sentence, part in zip(sentences, audio_parts):
                        end = cursor + seconds(part)
                        subtitles.append((cursor, end, sentence))
                        cursor = end
                    speech_time += spoken_duration
                    transcript.append(text)
                stage = min(pi, max(2, len(scene["speech"]) - 1))
                png = BUILD / f"slide-{ci:02}-{si:02}-{stage}.png"
                if practice:
                    png = BUILD / "practice.png"
                    slide(chapter, scene, ci, si, stage, practice=True).save(png)
                clip = BUILD / f"clip-{ci:02}-{si:02}-{pi:02}.mp4"
                cache_key = hashlib.sha256(f"{source_hash}|{args.length_scale}|{ci}|{si}|{pi}".encode()).hexdigest()
                stamp = clip.with_suffix(".key")
                if not clip.exists() or not stamp.exists() or stamp.read_text() != cache_key:
                    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
                           "-framerate", "5", "-i", png]
                    cmd += ["-t", f"{duration:.6f}", "-vf",
                            f"drawbox=x=0:y={H-6}:w={W}:h=6:color=0x20364d:t=fill",
                            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "stillimage", "-crf", "22",
                            "-pix_fmt", "yuv420p", "-r", "5", "-an",
                            "-video_track_timescale", "1000", clip]
                    run(cmd)
                    stamp.write_text(cache_key)
                chapter_clips.append((clip, duration))
                all_clips.append((clip, duration))
                elapsed += duration
            print(f"Lesson {ci+1:02} scene {si+1:02}: {elapsed/60:.1f} min total", flush=True)
            if args.preview:
                break
        if args.slides_only:
            continue
        chapter_records.append({"number": ci+1, "title": chapter["title"], "start": start, "end": elapsed})
        if args.preview:
            break
    if args.slides_only:
        print("All slides fit the frame.")
        return
    course_audio.close()
    metadata = [";FFMETADATA1", "title=Class 12 to GDS-ParT - Hinglish Learning Course",
                "artist=Avinash GDS-ParT project", "comment=Offline synthetic narration; includes guided practice."]
    for ch in chapter_records:
        metadata.extend(["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(ch['start']*1000)}",
                         f"END={round(ch['end']*1000)}", f"title={ch['number']:02} - {ch['title']}"])
    meta = BUILD / "chapters.ffmeta"
    meta.write_text("\n".join(metadata), encoding="utf-8")
    srt = OUT / "course.srt"
    srt.write_text("\n\n".join(f"{i}\n{timestamp(a,True)} --> {timestamp(b,True)}\n{t}"
                             for i, (a, b, t) in enumerate(subtitles, 1)) + "\n", encoding="utf-8")
    (OUT / "course.vtt").write_text("WEBVTT\n\n" + "\n\n".join(
        f"{timestamp(a)} --> {timestamp(b)}\n{t}" for a, b, t in subtitles) + "\n", encoding="utf-8")
    (OUT / "transcript.txt").write_text("\n".join(transcript), encoding="utf-8")
    listing = BUILD / "course-concat.txt"
    listing.write_text("".join(f"file '{p.resolve().as_posix()}'\noutpoint {d:.6f}\nduration {d:.6f}\n"
                               for p, d in all_clips), encoding="utf-8")
    movie = OUT / ("preview.mp4" if args.preview else "GDS-ParT_Hinglish_Course.mp4")
    run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
         "-i", meta, "-i", audio_path, "-i", srt, "-map", "0:v", "-map", "2:a", "-map", "3:0", "-map_metadata", "1",
         "-map_chapters", "1", "-c:v", "copy", "-c:a", "aac", "-b:a", "80k", "-c:s", "mov_text",
         "-metadata:s:s:0", "language=hin", "-movflags", "+faststart", movie])
    report = {"source_sha256": source_hash, "duration_seconds": elapsed, "narration_seconds": speech_time,
              "practice_seconds": practice_time, "chapters": chapter_records, "movie": movie.name,
              "voice": "hi_IN-pratham-medium (offline Piper, synthetic)", "length_scale": args.length_scale}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "chapters.txt").write_text("\n".join(f"{timestamp(c['start'])}  {c['number']:02} {c['title']}"
                                               for c in chapter_records), encoding="utf-8")
    write_player(movie, chapter_records, subtitles)
    print(json.dumps({k: v for k, v in report.items() if k != "chapters"}, indent=2), flush=True)
    if not args.preview and not args.min_minutes <= elapsed / 60 <= 180:
        raise ValueError(f"Course is {elapsed/60:.1f} minutes; requested range is {args.min_minutes}-180.")


if __name__ == "__main__":
    main()
