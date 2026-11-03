"""Verify complete cached narration and monotonic boundary timings before assembly."""
import json
import re
from pathlib import Path
from course_registry import narration_registry
from prepare_audio import CACHE, audio_id, plain, wav_duration

HERE = Path(__file__).resolve().parent


def main():
    failures, records = [], []
    for number, chapter in narration_registry().items():
        for key, (_, text, _) in chapter.items():
            path = CACHE / f"{audio_id(text)}.json"
            if not path.is_file():
                failures.append((number, key, "missing metadata"))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            boundaries = data["word_boundaries"]
            if not boundaries or data["text"] != plain(text):
                failures.append((number, key, "text or boundary mismatch"))
                continue
            last = boundaries[-1]
            cursor = 0
            for boundary in boundaries:
                gap = data["text"][cursor:boundary["text_offset"]]
                if re.search(r"[A-Za-z0-9]", gap):
                    failures.append((number, key, f"uncovered narration text: {gap}"))
                cursor = boundary["text_offset"] + boundary["word_length"]
            remaining = data["text"][last["text_offset"] + last["word_length"]:]
            if re.search(r"[A-Za-z0-9]", remaining):
                failures.append((number, key, f"unspoken trailing text: {remaining}"))
            for before, after in zip(boundaries, boundaries[1:]):
                if after["audio_offset"] < before["audio_offset"] or after["text_offset"] <= before["text_offset"]:
                    failures.append((number, key, "non-monotonic boundaries"))
            duration = wav_duration(CACHE / data["pcm_audio"])
            end = (last["audio_offset"] + last["duration"]) / 1e7
            if end > duration + .15:
                failures.append((number, key, f"boundary beyond audio: {end} > {duration}"))
            records.append({"chapter": number, "key": key, "seconds": duration, "words": len(text.split())})
    result = {"blocks": len(records), "failures": failures, "success": not failures,
              "narration_seconds": sum(row["seconds"] for row in records)}
    (HERE / "output" / "course_audio_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
