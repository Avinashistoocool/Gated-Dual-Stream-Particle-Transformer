"""Full decode, format/timing validation and representative frame exports."""
import argparse
import hashlib
import json
from pathlib import Path
from finish_media import call, duration
from course_registry import stem

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def validate(number, raw=False):
    video = OUT / f"{stem(number)}.mp4"
    if raw and number >= 2:
        video = HERE / "media" / f"course_{number:02}" / "videos" / stem(number) / "1080p30" / f"Chapter{number:02}.mp4"
    timeline = json.loads((OUT / f"chapter_{number:02}_timeline.json").read_text(encoding="utf-8"))
    actual = duration(video)
    if abs(actual-timeline[-1]["end"]) > .15:
        raise ValueError(f"Chapter {number}: timeline/media mismatch.")
    info = call(["-i", video, "-f", "ffmetadata", "-"]).stderr
    if "1920x1080" not in info or "30 fps" not in info:
        raise ValueError(f"Chapter {number}: unexpected video format.")
    call(["-loglevel", "error", "-xerror", "-threads", "1", "-i", video,
          "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"])
    frames = []
    selected = [timeline[len(timeline)//4], timeline[len(timeline)//2], timeline[-1]]
    for index, beat in enumerate(selected):
        second = min(beat["end"]-.2, beat["start"]+max(2, (beat["end"]-beat["start"])*.8))
        target = OUT / f"quality_chapter_{number:02}_{index}.png"
        call(["-loglevel", "error", "-y", "-threads", "1", "-ss", f"{second:.3f}", "-i", video,
              "-frames:v", "1", target])
        frames.append({"key": beat["key"], "second": second, "file": target.name})
    with video.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"chapter": number, "file": video.name, "raw": raw, "duration": actual,
              "resolution": "1920x1080", "fps": 30, "full_decode": "passed",
              "timeline_drift": actual-timeline[-1]["end"], "sha256": digest,
              "frames": frames, "perceptual_review": "Samples exported; not an exhaustive human review."}
    (OUT / f"chapter_{number:02}_media_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS chapter {number:02}: full decode, format, timing, three frame samples.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, nargs="+", required=True)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    for number in args.chapter:
        validate(number, args.raw)


if __name__ == "__main__":
    main()
