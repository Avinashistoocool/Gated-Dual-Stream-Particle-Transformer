"""Normalize each rendered chapter, embed captions, and concatenate the review."""
import json
import re
import subprocess
from pathlib import Path

from prepare_audio import setup_tools
from export_review import OUT, HERE, stamp, subtitles

FFMPEG = setup_tools()
STEMS = ["chapter_00_roadmap", "chapter_01_algebra_vectors_products"]


def call(arguments):
    result = subprocess.run([str(FFMPEG), "-hide_banner", *map(str, arguments)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr[-7000:])
    return result


def measured_loudness(path):
    result = call(["-i", path, "-vn", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
                   "-f", "null", "-"])
    match = re.search(r'\{\s*"input_i".*?\}', result.stderr, re.S)
    if not match:
        raise ValueError("FFmpeg did not return loudness measurements")
    return json.loads(match.group())


def duration(path):
    result = call(["-i", path, "-f", "ffmetadata", "-"])
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", result.stderr)
    if not match:
        raise ValueError(f"No duration for {path.name}")
    return int(match[1])*3600 + int(match[2])*60 + float(match[3])


def main():
    output_files = []
    chapters = []
    all_cues = []
    offset = 0.
    loudness = []
    for i, stem in enumerate(STEMS):
        original = HERE/"media"/"videos"/stem/"1080p30"/f"Chapter{i:02}.mp4"
        if not original.exists():
            raise FileNotFoundError(original)
        cues, timeline_duration = subtitles(i)
        measured = measured_loudness(original)
        values = {
            "measured_I": measured["input_i"], "measured_TP": measured["input_tp"],
            "measured_LRA": measured["input_lra"], "measured_thresh": measured["input_thresh"],
            "offset": measured["target_offset"],
        }
        audio_filter = "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true:" + ":".join(f"{k}={v}" for k,v in values.items())
        target = OUT/f"{stem}.mp4"
        call(["-loglevel","error","-y","-i",original,"-i",OUT/f"chapter_{i:02}.srt",
              "-map","0:v:0","-map","0:a:0","-map","1:0","-c:v","copy",
              "-af",audio_filter,"-ar","48000","-ac","2","-c:a","aac","-b:a","192k",
              "-c:s","mov_text","-metadata:s:s:0","language=eng","-movflags","+faststart",target])
        actual_duration = duration(target)
        if abs(actual_duration-timeline_duration) > .15:
            raise ValueError(f"Chapter {i} timing drift: video={actual_duration}, timeline={timeline_duration}")
        verified = measured_loudness(target)
        if abs(float(verified["input_i"])+16) > .5:
            raise ValueError(f"Chapter {i} loudness missed target: {verified}")
        loudness.append({"chapter":i,**verified})
        chapters.append({"number":i,"start":offset,"duration":actual_duration,"file":target.name})
        all_cues.extend((a+offset,b+offset,t) for a,b,t in cues)
        offset += actual_duration
        output_files.append(target)
    listing = OUT/"review-concat.txt"
    listing.write_text("".join(f"file '{path.resolve().as_posix()}'\n" for path in output_files),encoding="utf-8")
    srt = OUT/"review.srt"
    srt.write_text("\n\n".join(f"{i}\n{stamp(a,True)} --> {stamp(b,True)}\n{t}"
                              for i,(a,b,t) in enumerate(all_cues,1))+"\n",encoding="utf-8")
    (OUT/"review.vtt").write_text("WEBVTT\n\n"+"\n\n".join(
        f"{stamp(a)} --> {stamp(b)}\n{t}" for a,b,t in all_cues)+"\n",encoding="utf-8")
    meta = OUT/"review.ffmeta"
    titles=["Roadmap and what we will build","Algebra, vectors and three products"]
    content=[";FFMETADATA1","title=GDS-ParT - Chapters 0 and 1 review"]
    for chapter,title in zip(chapters,titles):
        content += ["[CHAPTER]","TIMEBASE=1/1000",f"START={round(chapter['start']*1000)}",
                    f"END={round((chapter['start']+chapter['duration'])*1000)}",
                    f"title=Chapter {chapter['number']}: {title}"]
    meta.write_text("\n".join(content),encoding="utf-8")
    review=OUT/"GDS_ParT_Chapters_00_01_REVIEW.mp4"
    call(["-loglevel","error","-y","-f","concat","-safe","0","-i",listing,"-i",srt,"-i",meta,
          "-map","0:v:0","-map","0:a:0","-map","1:0","-map_metadata","2","-map_chapters","2",
          "-c:v","copy","-c:a","copy","-c:s","mov_text","-metadata:s:s:0","language=eng",
          "-movflags","+faststart",review])
    call(["-loglevel","error","-xerror","-i",review,"-map","0:v:0","-map","0:a:0","-f","null","-"])
    actual=duration(review)
    if abs(actual-offset)>.15:
        raise ValueError(f"Concatenation drift: {actual} vs {offset}")
    (OUT/"chapters.txt").write_text("\n".join(
        f"{int(ch['start'])//60:02}:{int(ch['start'])%60:02} Chapter {ch['number']}: {title}"
        for ch,title in zip(chapters,titles)),encoding="utf-8")
    report={"status":"two-chapter review; not the complete course","duration":actual,
            "resolution":"1920x1080","fps":30,"file":review.name,"bytes":review.stat().st_size,
            "chapters":chapters,"loudness":loudness,"subtitles":len(all_cues),
            "full_decode":"passed","upload":"not verified by media assembly; verify delivery separately"}
    (OUT/"review_manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
