"""Generate all authorized Edge neural narration before rendering any chapter."""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg

from content import CHAPTER_00, CHAPTER_01

HERE = Path(__file__).resolve().parent
CACHE = HERE / "voice_cache"
VOICE = "en-IN-NeerjaNeural"
RATE = "-5%"
WPM = 140


def setup_tools():
    tools = HERE / "media" / "tools"
    tools.mkdir(parents=True, exist_ok=True)
    target = tools / "ffmpeg.exe"
    if not target.exists():
        shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), target)
    os.environ["PATH"] = str(tools) + os.pathsep + os.environ.get("PATH", "")
    return target


def plain(text):
    return " ".join(re.sub(r"<bookmark\s+mark=['\"]\w+['\"]\s*/>", "", text).split())


def audio_id(text):
    return hashlib.sha256(f"{VOICE}|{RATE}|{WPM}|{plain(text)}".encode()).hexdigest()[:24]


def wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def voiceover_mp3(metadata):
    data = json.loads(metadata.read_text(encoding="utf-8"))
    pcm = CACHE / (metadata.stem + ".wav")
    mp3 = CACHE / (metadata.stem + "-paced.mp3")
    if not mp3.exists():
        subprocess.run([str(setup_tools()), "-v", "error", "-y", "-i", str(pcm),
                        "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)], check=True)
    data["original_audio"] = mp3.name
    data["pcm_audio"] = pcm.name
    metadata.write_text(json.dumps(data, indent=2), encoding="utf-8")


async def generate(text, semaphore):
    text = plain(text)
    name = audio_id(text)
    meta = CACHE / f"{name}.json"
    if meta.exists() and (CACHE / f"{name}.wav").exists():
        voiceover_mp3(meta)
        return
    async with semaphore:
        raw = CACHE / f"{name}.mp3"
        boundaries = []
        with raw.open("wb") as stream:
            communicator = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
            async for chunk in communicator.stream():
                if chunk["type"] == "audio":
                    stream.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    boundaries.append(chunk)
        if not boundaries or raw.stat().st_size < 1000:
            raise RuntimeError(f"No usable neural narration for {name}")
        decoded = CACHE / f"{name}-raw.wav"
        ffmpeg = setup_tools()
        subprocess.run([str(ffmpeg), "-v", "error", "-y", "-i", str(raw),
                        "-ar", "44100", "-ac", "1", str(decoded)], check=True)
        duration = wav_duration(decoded)
        target = len(text.split()) * 60 / WPM
        tempo = duration / target
        if not .65 <= tempo <= 1.8:
            raise ValueError(f"Unexpected synthesis pace for {name}: tempo={tempo}")
        final = CACHE / f"{name}.wav"
        subprocess.run([str(ffmpeg), "-v", "error", "-y", "-i", str(decoded),
                        "-af", f"atempo={tempo:.9f}", str(final)], check=True)
        cursor = 0
        aligned = []
        for boundary in boundaries:
            word = html.unescape(boundary["text"])
            start = text.lower().find(word.lower(), cursor)
            if start < 0:
                raise ValueError(f"TTS word alignment failed at {word!r} in {name}")
            aligned.append({
                "audio_offset": round(boundary["offset"] / tempo),
                "duration": round(boundary["duration"] / tempo),
                "text_offset": start, "word_length": len(word), "text": word,
                "boundary_type": "Word",
            })
            cursor = start + len(word)
        data = {"text": text, "voice": VOICE, "words_per_minute": WPM,
                "duration": wav_duration(final), "original_audio": final.name,
                "word_boundaries": aligned, "tempo": tempo}
        meta.write_text(json.dumps(data, indent=2), encoding="utf-8")
        voiceover_mp3(meta)
        print(f"Cached {name}: {data['duration']:.1f} seconds", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, nargs="+")
    parser.add_argument("--all", action="store_true", help="Include all available course chapters.")
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    setup_tools()
    semaphore = asyncio.Semaphore(2)
    from course_registry import narration_registry
    registry = narration_registry()
    selected = args.chapter or (sorted(registry) if args.all else [0, 1])
    tasks = [generate_with_retry(item[1], semaphore) for n in selected
             for item in registry[n].values()]
    await asyncio.gather(*tasks)
    print(f"Neural narration cached for chapters {selected}.", flush=True)


async def generate_with_retry(text, semaphore, attempts=4):
    for attempt in range(attempts):
        try:
            return await generate(text, semaphore)
        except Exception:
            if attempt == attempts - 1:
                raise
            await asyncio.sleep(3 * (attempt + 1))


if __name__ == "__main__":
    asyncio.run(main())
