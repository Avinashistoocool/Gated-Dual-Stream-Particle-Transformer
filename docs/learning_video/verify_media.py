"""Verify the rendered course, including full audio/video decode."""
from __future__ import annotations

import json
import re
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    movie = OUT / manifest["movie"]
    expected = manifest["duration_seconds"]
    assert 7200 <= expected <= 10800, f"Requested duration not met: {expected}"
    assert len(manifest["chapters"]) == 16
    assert manifest["practice_seconds"] == 34 * 60
    assert manifest["narration_seconds"] > 85 * 60
    for previous, following in zip(manifest["chapters"], manifest["chapters"][1:]):
        assert abs(previous["end"] - following["start"]) < .001

    decoded = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(movie),
         "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    if decoded.stderr.strip():
        raise RuntimeError(f"Decode diagnostics require inspection: {decoded.stderr}")
    extracted = subprocess.run(
        [FFMPEG, "-hide_banner", "-i", str(movie), "-f", "ffmetadata", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    metadata = extracted.stdout
    assert metadata.count("[CHAPTER]") == 16
    assert "1600x900" in extracted.stderr
    assert "Audio: aac" in extracted.stderr
    assert "Subtitle: mov_text" in extracted.stderr
    duration = re.search(r"Duration: (\d+):(\d+):([\d.]+)", extracted.stderr)
    assert duration is not None
    actual = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    assert abs(actual - expected) < .3, (actual, expected)

    with wave.open(str(HERE / "build" / "course-audio.wav"), "rb") as wav:
        assert wav.getparams()[:3] == (1, 2, 22050)
        assert abs(wav.getnframes() / wav.getframerate() - expected) < .001
        peaks, rms = [], []
        while raw := wav.readframes(22050 * 30):
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768
            peaks.append(float(np.abs(samples).max()))
            rms.append(float(np.sqrt(np.mean(samples**2))))
    assert max(peaks) > .05, "Audio is effectively silent"
    assert sum(value > .005 for value in rms) > 150, "Too few audible 30-second blocks"

    cues = re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)",
                      (OUT / "course.srt").read_text(encoding="utf-8"))

    def seconds(stamp):
        h, m, s = stamp.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    last_end = 0
    for start, end in cues:
        a, b = seconds(start), seconds(end)
        assert a >= last_end - .002 and b > a
        assert b <= expected + .002
        last_end = b
    assert len(cues) > 700
    assert expected - last_end < 1
    report = {
        "full_decode": "passed",
        "runtime_seconds": actual,
        "chapters": metadata.count("[CHAPTER]"),
        "subtitle_cues": len(cues),
        "resolution": "1600x900",
        "file_bytes": movie.stat().st_size,
        "maximum_pcm_peak": max(peaks),
        "median_30_second_rms": float(np.median(rms)),
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
