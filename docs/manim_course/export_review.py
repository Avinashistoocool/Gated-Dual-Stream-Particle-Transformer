"""Export transcripts, subtitles, notebook and media commands from the same script."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from content import CHAPTER_00, CHAPTER_01, CODE

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def stamp(seconds, comma=False):
    ms = round(seconds*1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02}{',' if comma else '.'}{ms:03}"


def export_text(write_notebook=False):
    OUT.mkdir(exist_ok=True)
    notebook = []
    for number, chapter in enumerate((CHAPTER_00, CHAPTER_01)):
        lines = [f"# Chapter {number}: narration and visual script", "",
                 "Narration language: clear English. On-screen language: English.", ""]
        for key, (screen, narration, source) in chapter.items():
            lines += [f"## {key}", f"**On screen:** {screen}", "",
                      narration, "", f"**Source:** {source}", ""]
            notebook.append({"cell_type":"markdown","metadata":{},
                             "source":[f"## Chapter {number}: {screen}\n", narration, f"\n\nSource: {source}"]})
        (OUT/f"chapter_{number:02}_transcript.md").write_text("\n".join(lines),encoding="utf-8")
    for key, code in CODE.items():
        notebook += [{"cell_type":"markdown","metadata":{},"source":[f"## Executable calculation: {key}"]},
                     {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":code.splitlines(True)}]
    if write_notebook:
        (OUT/"review_notebook.ipynb").write_text(json.dumps({
            "cells":notebook,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                       "language_info":{"name":"python"}},
            "nbformat":4,"nbformat_minor":5},indent=2),encoding="utf-8")


def subtitles(number):
    timeline=json.loads((OUT/f"chapter_{number:02}_timeline.json").read_text(encoding="utf-8"))
    cues=[]
    for beat in timeline:
        words=beat["boundaries"]
        text_source = " ".join(beat["text"].split())
        for sentence in re.finditer(r"[^.!?]+[.!?]?", text_source):
            first = sentence.start()
            last = sentence.end()
            group = [w for w in words if first <= w["text_offset"] < last]
            if not group:
                continue
            start=beat["start"]+group[0]["audio_offset"]/1e7
            end=beat["start"]+(group[-1]["audio_offset"]+group[-1]["duration"])/1e7
            text=text_source[first:last].strip()
            terms=text.split()
            lines=[" ".join(terms[i:i+12]) for i in range(0,len(terms),12)]
            if len(lines)>3:
                raise ValueError(f"Subtitle exceeds three twelve-word lines: {text}")
            text="\n".join(lines)
            cues.append((start,end,text))
    for i,(start,end,_) in enumerate(cues):
        if end <= start or (i and start < cues[i-1][1]-.03):
            raise ValueError(f"Invalid subtitle timing in chapter {number}")
    (OUT/f"chapter_{number:02}.srt").write_text("\n\n".join(
        f"{i}\n{stamp(a,True)} --> {stamp(b,True)}\n{t}"
        for i,(a,b,t) in enumerate(cues,1))+"\n",encoding="utf-8")
    (OUT/f"chapter_{number:02}.vtt").write_text("WEBVTT\n\n"+"\n\n".join(
        f"{stamp(a)} --> {stamp(b)}\n{t}" for a,b,t in cues)+"\n",encoding="utf-8")
    return cues,timeline[-1]["end"]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--text-only",action="store_true")
    parser.add_argument("--review-notebook",action="store_true")
    args=parser.parse_args()
    export_text(write_notebook=args.review_notebook)
    if not args.text_only:
        for i in range(2):
            cues,duration=subtitles(i)
            print(f"Chapter {i}: {duration:.3f} seconds, {len(cues)} word-timed subtitle cues.")


if __name__=="__main__":
    main()
