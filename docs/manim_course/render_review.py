"""Render chapters separately; spell-check and run code before any video render."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from prepare_audio import setup_tools

HERE=Path(__file__).resolve().parent
FILES=["chapter_00_roadmap.py","chapter_01_algebra_vectors_products.py"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--chapter",type=int,choices=[0,1])
    args=parser.parse_args()
    setup_tools()
    subprocess.run(["py","-3.14",str(HERE/"companion"/"tests.py")],check=True)
    subprocess.run([sys.executable,str(HERE/"audit.py")],check=True)
    subprocess.run([sys.executable,str(HERE/"export_review.py"),"--text-only"],check=True)
    for i,file in enumerate(FILES):
        if args.chapter is not None and args.chapter!=i:
            continue
        command=[sys.executable,"-m","manim","--renderer","cairo","-r","1920,1080",
                 "--fps","30","--verbosity","WARNING","--progress_bar","none","--media_dir",str(HERE/"media"),
                 "--disable_caching"]
        if args.dry_run:
            command += ["--dry_run"]
        command += [str(HERE/file),f"Chapter{i:02}"]
        subprocess.run(command,cwd=HERE,check=True)
    if args.chapter is None:
        subprocess.run([sys.executable,str(HERE/"export_review.py")],check=True)


if __name__=="__main__":
    main()
