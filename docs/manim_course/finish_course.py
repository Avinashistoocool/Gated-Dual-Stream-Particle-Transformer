"""Export, normalize, validate, concatenate and package the complete course."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from course_registry import load_course, narration_registry, stem
from course_content import TITLES
from export_review import subtitles, stamp
from finish_media import call, duration, measured_loudness
from prepare_audio import plain

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
ALL_TITLES = {0: "Roadmap and what we will build", 1: "Algebra, vectors and three products", **TITLES}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transcript(number):
    existing = OUT / f"chapter_{number:02}_transcript.md"
    if number < 2 and existing.is_file():
        return existing.read_text(encoding="utf-8")
    registry = narration_registry()
    chapters, code = load_course()
    lines = [f"# Chapter {number}: {ALL_TITLES[number]}", ""]
    for key, (caption, narration, source) in registry[number].items():
        lines += [f"## {caption}", "", narration, "", f"Source: {source}", ""]
        if number >= 2:
            beat = chapters[number][key]
            if beat.kind == "code":
                lines += ["```python", code[beat.steps[0]], "```", ""]
            else:
                lines += [f"- `{formula}`" for formula in beat.steps]
                lines.append("")
    text = "\n".join(lines)
    (OUT / f"chapter_{number:02}_transcript.md").write_text(text, encoding="utf-8")
    return text


def normalize(number):
    target = OUT / f"{stem(number)}.mp4"
    timeline = json.loads((OUT / f"chapter_{number:02}_timeline.json").read_text(encoding="utf-8"))
    registered = narration_registry()[number]
    if [beat["key"] for beat in timeline] != list(registered):
        raise ValueError(f"Chapter {number}: missing, duplicated or reordered narration blocks.")
    for beat in timeline:
        if plain(beat["text"]) != plain(registered[beat["key"]][1]):
            raise ValueError(f"Chapter {number}: rendered narration differs from the registered script.")
    cues, expected = subtitles(number)
    original = (HERE / "media" / f"course_{number:02}" / "videos" / stem(number) /
                "1080p30" / f"Chapter{number:02}.mp4") if number >= 2 else None
    source_hash = sha256(original) if original is not None else None
    captions_hash = sha256(OUT / f"chapter_{number:02}.srt")
    target_hash = sha256(target) if target.is_file() else None
    record_path = OUT / f"chapter_{number:02}_normalization.json"
    previous = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    reusable = bool(target_hash and previous.get("source_sha256") == source_hash
                    and previous.get("target_sha256") == target_hash
                    and previous.get("captions_sha256") == captions_hash
                    and previous.get("verified"))
    if number >= 2 and not reusable:
        measured = measured_loudness(original)
        values = {"measured_I": measured["input_i"], "measured_TP": measured["input_tp"],
                  "measured_LRA": measured["input_lra"], "measured_thresh": measured["input_thresh"],
                  "offset": measured["target_offset"]}
        audio_filter = "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:" + ":".join(
            f"{k}={v}" for k, v in values.items())
        call(["-loglevel", "error", "-y", "-i", original, "-i", OUT / f"chapter_{number:02}.srt",
              "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0", "-c:v", "copy",
              "-af", audio_filter, "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
              "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", target])
    if not target.is_file():
        raise FileNotFoundError(target)
    actual = duration(target)
    if abs(actual-expected) > .15:
        raise ValueError(f"Chapter {number} timeline drift: {actual} vs {expected}")
    measurement = previous["loudness"] if reusable else measured_loudness(target)
    if abs(float(measurement["input_i"])+16) > .5:
        raise ValueError(f"Chapter {number} loudness target missed: {measurement}")
    info = call(["-i", target, "-f", "ffmetadata", "-"]).stderr
    if "1920x1080" not in info or "30 fps" not in info:
        raise ValueError(f"Unexpected video format: {target}")
    record = {"chapter": number, "verified": True, "duration": actual,
              "narration_blocks": len(timeline), "registered_narration_coverage": "passed",
              "source_sha256": source_hash, "target_sha256": sha256(target),
              "captions_sha256": captions_hash, "loudness": measurement}
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Verified chapter {number:02}: {actual:.2f}s, {measurement['input_i']} LUFS", flush=True)
    return target, cues, actual, measurement


def package():
    bundle = OUT / "GDS_ParT_Complete_Companion_Subtitles_Transcripts.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in HERE.glob("*.py"):
            if file.name == "gds_part_colab.py":
                continue
            archive.write(file, f"manim_course/{file.name}")
        for file in HERE.glob("*.md"):
            archive.write(file, f"manim_course/{file.name}")
        for file in HERE.glob("*.ps1"):
            archive.write(file, f"manim_course/{file.name}")
        archive.write(HERE / "requirements.txt", "manim_course/requirements.txt")
        if (HERE / ".gitignore").is_file():
            archive.write(HERE / ".gitignore", "manim_course/.gitignore")
        # These are comparison references, never the independent reconstruction.
        repository = HERE.parents[1]
        model_reference = repository / "gds_part_colab.py"
        cache_reference = repository / "data" / "jetclass_cache.py"
        report_reference = HERE.parent / "GDS-ParT_Technical_Report.pdf"
        if not model_reference.is_file():
            model_reference = HERE / "gds_part_colab.py"
        if not cache_reference.is_file():
            cache_reference = HERE / "data" / "jetclass_cache.py"
        if not report_reference.is_file():
            report_reference = HERE / "reference" / "GDS-ParT_Technical_Report.pdf"
        archive.write(model_reference, "manim_course/gds_part_colab.py")
        archive.write(cache_reference, "manim_course/data/jetclass_cache.py")
        archive.write(report_reference, "manim_course/reference/GDS-ParT_Technical_Report.pdf")
        license_reference = model_reference.parent / "LICENSE"
        if not license_reference.is_file():
            license_reference = HERE / "reference" / "LICENSE"
        if license_reference.is_file():
            archive.write(license_reference, "manim_course/reference/LICENSE")
        for file in (HERE / "companion").rglob("*"):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix in (
                    ".py", ".ipynb", ".md", ".json", ".txt", ".csv"):
                archive.write(file, "manim_course/" + file.relative_to(HERE).as_posix())
        for file in OUT.iterdir():
            if (file.name.startswith(("queue_", "rehearsal_status", "render_status"))
                    or "_authoring_manifest" in file.name
                    or file.name in {"reconstruction_tests.json", "cpu_demo.json",
                                     "parameter_audit.json", "delivery_manifest.json"}):
                continue
            if file.is_file() and file.suffix in (".srt", ".vtt", ".md", ".json", ".ffmeta", ".txt"):
                archive.write(file, "manim_course/output/" + file.name)
    with zipfile.ZipFile(bundle) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Bundle CRC failure: {bad}")
    return bundle


def write_delivery_manifest(bundle):
    report_path = OUT / "course_manifest.json"
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("full_decode") != "passed" or len(report.get("chapters", [])) != 16:
        raise ValueError("A verified sixteen-chapter course is required for delivery.")
    records = [report, *report["chapters"]]
    expected_names = ["GDS_ParT_Complete_Course.mp4",
                      *[f"{stem(number)}.mp4" for number in range(16)]]
    manifest = []
    for record, expected_name in zip(records, expected_names):
        if record["file"] != expected_name:
            raise ValueError("Unexpected filename in course manifest.")
        path = OUT / expected_name
        digest = sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Media changed after assembly: {expected_name}")
        manifest.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": digest})
    manifest.append({"filename": bundle.name, "bytes": bundle.stat().st_size, "sha256": sha256(bundle)})
    (OUT / "delivery_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, nargs="+", help="Normalize/export selected chapters only.")
    parser.add_argument("--bundle-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.bundle_only:
        bundle = package()
        write_delivery_manifest(bundle)
        print(bundle)
        return
    selected = args.chapter or list(range(16))
    if args.chapter is None:
        load_course(require_complete=True)
    records, cues, paths, transcripts = [], [], [], []
    offset = 0.
    for number in selected:
        target, chapter_cues, length, loudness = normalize(number)
        paths.append(target)
        transcripts.append(transcript(number))
        records.append({"number": number, "title": ALL_TITLES[number], "start": offset,
                        "duration": length, "file": target.name, "loudness": loudness,
                        "bytes": target.stat().st_size, "sha256": sha256(target)})
        cues.extend((a+offset, b+offset, text) for a, b, text in chapter_cues)
        offset += length
    if args.chapter is not None:
        return
    listing = OUT / "course-concat.txt"
    listing.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in paths), encoding="utf-8")
    srt = OUT / "GDS_ParT_Complete_Course.en.srt"
    srt.write_text("\n\n".join(f"{i}\n{stamp(a,True)} --> {stamp(b,True)}\n{text}"
                              for i, (a, b, text) in enumerate(cues, 1))+"\n", encoding="utf-8")
    (OUT / "GDS_ParT_Complete_Course.en.vtt").write_text("WEBVTT\n\n"+"\n\n".join(
        f"{stamp(a)} --> {stamp(b)}\n{text}" for a, b, text in cues)+"\n", encoding="utf-8")
    (OUT / "GDS_ParT_Complete_Transcript.md").write_text("\n\n".join(transcripts), encoding="utf-8")
    metadata = [";FFMETADATA1", "title=GDS-ParT - complete mathematics and reconstruction course"]
    for record in records:
        metadata += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={round(record['start']*1000)}",
                     f"END={round((record['start']+record['duration'])*1000)}",
                     f"title=Chapter {record['number']}: {record['title']}"]
    meta = OUT / "course.ffmeta"
    meta.write_text("\n".join(metadata), encoding="utf-8")
    target = OUT / "GDS_ParT_Complete_Course.mp4"
    call(["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
          "-i", srt, "-i", meta, "-map", "0:v:0", "-map", "0:a:0", "-map", "1:0",
          "-map_metadata", "2", "-map_chapters", "2", "-c:v", "copy", "-c:a", "copy",
          "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart", target])
    call(["-loglevel", "error", "-xerror", "-i", target, "-map", "0:v:0", "-map", "0:a:0",
          "-f", "null", "-"])
    actual = duration(target)
    if abs(actual-offset) > .2:
        raise ValueError(f"Complete-course timing drift: {actual} vs {offset}")
    loudness = measured_loudness(target)
    if abs(float(loudness["input_i"])+16) > .5:
        raise ValueError(f"Combined loudness target missed: {loudness}")
    markers = "\n".join(
        f"{stamp(record['start']).split('.')[0]} Chapter {record['number']}: {record['title']}"
        for record in records)
    (OUT / "course_chapters.txt").write_text(markers+"\n", encoding="utf-8")
    report = {"file": target.name, "duration": actual, "bytes": target.stat().st_size,
              "sha256": sha256(target),
              "chapters": records, "subtitles": len(cues), "resolution": "1920x1080", "fps": 30,
              "registered_narration_coverage": "passed",
              "narration_blocks": sum(len(chapter) for chapter in narration_registry().values()),
              "loudness": loudness, "full_decode": "passed",
              "perceptual_review": "Automated bounds checks and sampled frames; no exhaustive human review.",
              "upload": "Not verified at package creation; delivery verification is a separate subsequent step."}
    (OUT / "course_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    bundle = package()
    write_delivery_manifest(bundle)
    print(f"Complete course verified: {actual/60:.2f} minutes, {len(cues)} captions; bundle {bundle.name}.", flush=True)


if __name__ == "__main__":
    main()
