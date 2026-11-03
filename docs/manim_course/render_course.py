"""Bounded parallel rendering with isolated TeX/media and persistent failure logs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import subprocess
import sys
from pathlib import Path
from prepare_audio import setup_tools
from course_registry import stem

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def render(number, dry=False):
    media = HERE / "media" / f"course_{number:02}"
    media.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "manim", "--renderer", "cairo", "-r", "1920,1080",
               "--fps", "30", "--verbosity", "WARNING", "--progress_bar", "none",
               "--media_dir", str(media), "--disable_caching"]
    if dry:
        command.extend(["--dry_run", "--save_last_frame"])
    command.extend([str(HERE / f"{stem(number)}.py"), f"Chapter{number:02}"])
    log = OUT / f"chapter_{number:02}_{'rehearsal' if dry else 'render'}.log"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=HERE, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-35:])
        raise RuntimeError(f"Chapter {number} failed; see {log}\n{tail}")
    print(f"PASS chapter {number:02}: {'rehearsal' if dry else '1080p30 render'}", flush=True)
    return number


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapter", type=int, nargs="+", default=list(range(2, 16)))
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(n not in range(2, 16) for n in args.chapter):
        raise ValueError("This runner preserves Chapters 0-1; use the review runner for them.")
    setup_tools()
    OUT.mkdir(exist_ok=True)
    failures, completed = [], []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(render, number, args.dry_run): number for number in args.chapter}
        for future in as_completed(futures):
            number = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append({"chapter": number, "error": str(error)})
                print(str(error), flush=True)
    report = {"chapters": sorted(completed), "failures": failures, "dry_run": args.dry_run}
    suffix = "_".join(f"{n:02}" for n in args.chapter)
    target = OUT / f"{'rehearsal' if args.dry_run else 'render'}_status_{suffix}.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
