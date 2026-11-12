# GDS-ParT: mathematics and independent reconstruction course

This is a **new production**. It is separate from the rejected slide-based video in `docs\learning_video`.
The approved Chapters 0 and 1 are kept. The user authorized the remaining
Chapters 2-15 and a complete independent reconstruction companion.
The production status comes from evidence. Source files alone are not proof of rendering
or upload. For these stages, refer to `QUALITY_CHECKLIST.md` and the generated manifests.

## Content and sources

- `chapter_00_roadmap.py`: particle collections, inputs, streams, reuse of relations, the hybrid architecture, the differences between models and two solved questions.
- `chapter_01_algebra_vectors_products.py`: scalar rules, composition, vectors, indices, sums, means, scaling, Hadamard/inner/outer products, executable code and three solved exercises.
- `content.py`: one English source for narration, captions, sources and executable examples.
- `course_content.py`: the shared lesson schema and Chapters 2-5.
- `content_06_08.py`, `content_09_12.py`, `content_13_15.py`: the later chapters, which have independent authorship.
- `course_registry.py`: combines the chapter registries and does not change the approved review.
- `course_studio.py`: equation transformations, indexed arrays, probability bars and computational graphs.
- The scenes of the later chapters add custom physics geometry, attention arithmetic and a trace of the complete model.
- Primary reference: `docs\GDS-ParT_Technical_Report.pdf`, specially pages 6-12 and Appendices C, D and I.
- Code reference: `gds_part_colab.py` and `data\jetclass_cache.py`.

The approved review uses numbers from the report, or numbers calculated from its illustrative inputs.
For example, the report's functions give 81 and 33 for the input four. These are derived arithmetic, not reported model results.
The solution of the first exercise of Appendix I is `(6,-4)`, `2`, and `[[6,8],[-3,-4]]`.

The continuation also uses small teaching matrices, probability distributions and synthetic batches.
The course introduces each of them explicitly. Their outputs are calculated
and tested. The course does not show them as trained weights or reported physics measurements.
The course keeps all the mandatory report examples and all four exact model counts.

## Voice and timing

The authorized custom API relay explicitly rejected speech generation. No credentials were copied into this project.
The user then authorized alternative speech systems. **Edge TTS, `en-IN-NeerjaNeural`, makes neural English narration correctly.**
It is unofficial, and it can stop operating. Thus, the pipeline generates and caches all the narration **before** rendering.
The service receives only this original educational narration. It does not receive credentials, local files or training data.

FFmpeg tempo adjustment sets the pace of the audio to about 140 words/minute.
The pipeline adjusts the word-boundary timestamps by the same factor and gives them to `manim-voiceover`.
Each teaching block uses `with self.voiceover(text=...)`. Sentence bookmarks set the time of the related animations.
There is no robotic draft voice as a substitute, and there is no long silent exercise interval.

Voiceover 0.3.7 uses MP3 metadata internally. The pipeline makes paced MP3 metadata that agrees with the lossless PCM audio.
Manim mixes the PCM audio. This path does not need SoX, because the global speed of voiceover stays at one.
The final chapters use two-pass loudness normalization. The targets are **-16 LUFS** and **-1.5 dBTP**.
The two encoded chapters measure -16.01 LUFS. Their true peaks are -1.48 and -1.75 dBTP.

## Run on this Windows environment

The isolated animation environment uses Python 3.13. The numerical examples use the Python 3.14 installation that has PyTorch.

```powershell
py -3.13 -m venv docs\manim_course\.venv
& docs\manim_course\.venv\Scripts\python.exe -m pip install -r docs\manim_course\requirements.txt
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\audit.py
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\prepare_audio.py
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\render_review.py --dry-run
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\render_review.py
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\finish_media.py
```

MiKTeX `latex` and `dvisvgm` must be on PATH. `prepare_audio.py` supplies the FFmpeg executable locally.
`--chapter 0` or `--chapter 1` limits the rendering command to one chapter.
The dry run checks the animation construction and the boundaries. It does not write the final video.

All text is programmatic Manim `Text`, `MathTex`, `Matrix` or `Code`. No image generator renders typography.
`content.py` contains the palette. The captions use Segoe UI. The code uses Consolas at font size 28. The mathematics uses LaTeX.
The footer is a map of the components, not an attention result. The moving particle dots are an illustrative set, not sampled physics events.

## Review artifacts

After a successful completion, `output` contains:

- `GDS_ParT_Chapters_00_01_REVIEW.mp4`: a joined **review**, not the full 145-150-minute course.
- Separate chapter MP4s at 1080p and 30 FPS, with embedded English subtitles.
- SRT/VTT subtitles for each chapter and for all chapters together, with full English sentences.
- `chapter_00_transcript.md`, `chapter_01_transcript.md`.
- `chapters.txt`, embedded MP4 chapters and `review_manifest.json`.
- `strings_checked.md`: each audited registered English string, literal formula and executable code block.
- Scene inventories and timing manifests, to check what the render actually made.
- `executed_examples.json`: the actual captured PyTorch output that the video uses.

Then the user made the timings flexible, and gave priority to explanatory depth and animation quality.
This review has a duration of 17 minutes, 11 seconds. There is no claim about a full-course duration for these two chapters.
For the completed checks and the requirements that wait for review or for later chapters, refer to `QUALITY_CHECKLIST.md`.

## Reproducing the course

```powershell
py -3.14 docs\manim_course\companion\test_reconstruction.py
py -3.14 docs\manim_course\execute_course.py
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\audit_course.py
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\prepare_audio.py --all
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\render_course.py --dry-run --jobs 2
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\render_course.py --jobs 2
& docs\manim_course\.venv\Scripts\python.exe docs\manim_course\finish_course.py
```

`--chapter 2 3` selects individual chapters for audio, rehearsal, rendering or
normalization. Each chapter uses its own media directory and TeX directory. The render logs
stay in `output`. During rendering, the cached speech is read-only. If an asset is missing,
the render fails. It does not change to a different voice without a message. The assembly of the
complete course does not include a failed render.

`finish_course.py` does these steps:

1. It keeps Chapters 0-1, which are already normalized, and normalizes the new chapters.
2. It embeds English subtitles.
3. It checks the video format and the drift of the timeline.
4. It assembles the full video with chapter markers.
5. It decodes the full video and measures the loudness.
6. It packages the source, the subtitles, the transcripts and the companion artifacts.

The script uses a verified normalization again only when its recorded evidence agrees with
the SHA-256 hashes of the source video, the target video and the subtitles.
The bundle does not include virtual environments, caches, intermediate media or the
rejected old course. The full-course delivery manifest records SHA-256 hashes.
A local MP4 is not proof of delivery to Google Drive.

The bundle also includes the technical report PDF, so that it is portable. It also includes two
**original reference sources** with clear labels, `gds_part_colab.py` and
`data\jetclass_cache.py`, at the root of the course. The physics chapter uses these references for its direct
demonstrations of the production code. The course does **not** show them
as the independently written reconstruction in `companion`. The bundle does not include JetClass data
or credentials. The independent notebook operates without training data.

## Verified complete-course build

The duration of the complete local build is **2 hours, 33 minutes, 57.01 seconds**:

- `output\GDS_ParT_Complete_Course.mp4`: 1920×1080, 30 FPS, 480,102,120 bytes.
- Sixteen chapter MP4s, each rendered and normalized separately, which include the kept Chapters 0-1.
- **375 narrated teaching blocks**, **1,623 English subtitle cues** and **16 embedded chapter markers**.
- SRT/VTT for the full course and for each chapter, transcripts, chapter timestamps, executable examples and source inventories.
- `output\GDS_ParT_Complete_Companion_Subtitles_Transcripts.zip`: the source, the original reference material,
  the independent companion, the executed notebook, captions, transcripts and validation evidence.
- `output\course_manifest.json` and `delivery_manifest.json`: measured metadata and SHA-256 checksums.

The complete audio/video stream decodes correctly. The integrated loudness is
**-16.00 LUFS**. The measured true peak is **-1.48 dBTP** after AAC encoding
(the normalization filter has a target of -1.5 dBTP). Each chapter measures -16.00 or
-16.01 LUFS. All the authored narration blocks agree with the rendered timelines.

The independent companion passes 24 tests. Its notebook executes 35 code cells
across all sixteen chapters. The 34 displayed snippets of the continuation execute.
18 foundation checks include a second backward pass after the scalar update.
The kept review adds seven executed snippets.

Bounds checks, targeted frame exports and full decode checks are automated evidence.
They are not a full human viewing or an independent perceptual audit of the subtitle synchronization.
The Drive delivery has a separate verification after packaging. Refer to the publication section
and `QUALITY_CHECKLIST.md`. Do not think that a successful local build shows an upload.

## Independent companion

`companion\tests.py` verifies each Chapter 1 calculation and the report exercise.
`companion\gds_part_reconstruction.py` independently implements these items:

- embeddings and relation aggregation
- gates and explicit multihead attention
- residual blocks and class attention
- readout and pair features.

It does not import the production model.
Tests can import the production code only as an independent comparison oracle.
The executed notebook covers all sixteen chapters. Its CPU demonstration is
synthetic memorisation. It is not a JetClass training result or an accuracy result.
For commands, intentional implementation differences,
parameter audits, numerical outputs and validation artifacts, refer to `companion\README.md`.

The four model counts are **1,280,684** (six-block v1), **2,074,420**
(reported shared hybrid), **2,488,116** (default conditioned hybrid) and
**2,024,756** (local bias-only proxy). The last count is not an exact reproduction of
the published Particle Transformer. The relation tensor does not change during a forward
pass. But the conditioned summaries can change with the particle state.

## Publication

The combined `GDS_ParT_Chapters_00_01_REVIEW.mp4` was uploaded through the signed-in Edge session of the user.
After a reload of the [supplied Google Drive folder](https://drive.google.com/drive/folders/1BiAyQ9qpmuspf_isz8S5o-xjsSFthccK), its exact filename was verified.
Its Drive preview opened with a video player. This delivery is the two-chapter review, not the rejected slide video.
Each additional course deliverable must have its own verification after upload.
The successful upload of the review does not show the delivery of later chapters.
Git ignores the generated assets and the environments. The documentation site does not include them.

**The complete-course delivery was verified on 1 November 2026 at 07:56 IST.**
Drive reported **18 uploads complete**. After a reload of the supplied folder, all eighteen exact filenames
stayed in it:

- the complete MP4
- all sixteen individual chapter MP4s
- the bundle of the companion, subtitles, transcripts and source.

`output\drive_upload_verification.json` records that check. The delivered local
files continue to agree with the sealed SHA-256 manifest. The packaged checklist records
the status at build time. This external delivery receipt comes after the packaging.
There is no claim of full Drive playback or of a human perceptual review.
