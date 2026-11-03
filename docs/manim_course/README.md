# GDS-ParT: mathematics and independent reconstruction course

This is a **new production**, separate from the rejected slide-based video in `docs\learning_video`.
The approved Chapters 0 and 1 are preserved. The user has authorized the remaining
Chapters 2-15 and a complete independent reconstruction companion.
Production status is evidence-based: source availability is not proof of rendering
or upload. Consult `QUALITY_CHECKLIST.md` and the generated manifests for those stages.

## Content and sources

- `chapter_00_roadmap.py`: particle collections, inputs, streams, relation reuse, hybrid architecture, model distinctions and two solved questions.
- `chapter_01_algebra_vectors_products.py`: scalar rules, composition, vectors, indices, sums, means, scaling, Hadamard/inner/outer products, executable code and three solved exercises.
- `content.py`: one English source for narration, captions, sources and executable examples.
- `course_content.py`: shared lesson schema and Chapters 2-5.
- `content_06_08.py`, `content_09_12.py`, `content_13_15.py`: independently authored later chapters.
- `course_registry.py`: combines chapter registries without altering the approved review.
- `course_studio.py`: equation transformations, indexed arrays, probability bars and computational graphs.
- Later chapter scenes add custom physics geometry, attention arithmetic and complete model tracing.
- Primary reference: `docs\GDS-ParT_Technical_Report.pdf`, especially pages 6-12 and Appendices C, D and I.
- Code reference: `gds_part_colab.py` and `data\jetclass_cache.py`.

The approved review uses numbers from the report or calculated from its illustrative inputs.
For example, applying the report's functions to input four gives 81 and 33; these are derived arithmetic, not reported model results.
Appendix I's first exercise is solved as `(6,-4)`, `2`, and `[[6,8],[-3,-4]]`.

The continuation also uses explicitly introduced small teaching matrices,
probability distributions and synthetic batches. Their outputs are calculated
and tested, not presented as trained weights or reported physics measurements.
All mandatory report examples and all four exact model counts are retained.

## Voice and timing

The authorized custom API relay explicitly rejected speech generation. No credentials were copied into this project.
The user then authorized alternative speech systems. **Edge TTS, `en-IN-NeerjaNeural`, successfully generates neural English narration.**
It is unofficial and may stop working; all narration is generated and cached **before** rendering.
The service receives only this original educational narration, not credentials, local files or training data.

Audio is paced to approximately 140 words/minute using FFmpeg tempo adjustment.
Word-boundary timestamps are adjusted by the same factor and passed into `manim-voiceover`.
Each teaching block uses `with self.voiceover(text=...)`; sentence bookmarks schedule its corresponding animations.
There is no substitute robotic draft voice and no long silent exercise interval.

Voiceover 0.3.7 uses MP3 metadata internally. The pipeline creates matching paced MP3 metadata and lossless PCM audio;
Manim mixes the PCM audio. SoX is not required for this path because voiceover's global speed remains one.
Final chapters use two-pass loudness normalization targeting **-16 LUFS** and **-1.5 dBTP**.
Both encoded chapters measure -16.01 LUFS; true peaks are -1.48 and -1.75 dBTP, respectively.

## Run on this Windows environment

The isolated animation environment uses Python 3.13. Numerical examples use the existing Python 3.14 installation with PyTorch.

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
The dry run checks animation construction and boundaries without writing the final video.

All text is programmatic Manim `Text`, `MathTex`, `Matrix` or `Code`; no image generator renders typography.
The palette is centralized in `content.py`. Captions use Segoe UI; code uses Consolas at font size 28; mathematics uses LaTeX.
The footer is a component map, not an attention result. The moving particle dots are an illustrative set, not sampled physics events.

## Review artifacts

After successful completion, `output` contains:

- `GDS_ParT_Chapters_00_01_REVIEW.mp4`: joined **review**, not the full 145-150-minute course.
- Separate 1080p, 30 FPS chapter MP4s with embedded English subtitles.
- Per-chapter and combined SRT/VTT subtitles, using full English sentences.
- `chapter_00_transcript.md`, `chapter_01_transcript.md`.
- `chapters.txt`, embedded MP4 chapters and `review_manifest.json`.
- `strings_checked.md`: every audited registered English string, literal formula and executable code block.
- Scene inventories and timing manifests for checking what was actually rendered.
- `executed_examples.json`: actual captured PyTorch output used by the video.

The user subsequently made timings flexible, prioritizing explanatory depth and animation quality.
This review lasts 17 minutes, 11 seconds. No full-course duration claim is made for these two chapters.
See `QUALITY_CHECKLIST.md` for completed checks and requirements still awaiting review or later chapters.

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
normalization. Each chapter uses its own media and TeX directory. Render logs
persist in `output`. Cached speech is read-only during rendering; missing assets
fail rather than silently switching voice. A failed render is not included in
the complete-course assembly.

`finish_course.py` preserves the already normalized Chapters 0-1, normalizes new
chapters, embeds English subtitles, checks video format and timeline drift,
assembles the full video with chapter markers, decodes it completely, measures
loudness, and packages source, subtitles, transcripts and companion artifacts.
Verified normalization is reused only when source-video, target-video and
subtitle SHA-256 hashes all match its recorded evidence.
The bundle excludes virtual environments, caches, intermediate media and the
rejected old course. The full-course delivery manifest records SHA-256 hashes.
An MP4 existing locally does not prove Google Drive delivery.

For portability, the bundle also includes the technical report PDF and two
clearly separate **original reference sources**, `gds_part_colab.py` and
`data\jetclass_cache.py`, at the course root. The physics chapter's direct
production-code demonstrations use those references. They are **not** presented
as the independently written reconstruction in `companion`. No JetClass data
or credentials are included. The independent notebook works without training data.

## Verified complete-course build

The complete local build is **2 hours, 33 minutes, 57.01 seconds**:

- `output\GDS_ParT_Complete_Course.mp4`: 1920×1080, 30 FPS, 480,102,120 bytes.
- Sixteen individually rendered and normalized chapter MP4s, including the preserved Chapters 0-1.
- **375 narrated teaching blocks**, **1,623 English subtitle cues**, and **16 embedded chapter markers**.
- Full-course and per-chapter SRT/VTT, transcripts, chapter timestamps, executable examples and source inventories.
- `output\GDS_ParT_Complete_Companion_Subtitles_Transcripts.zip`: source, original reference material,
  independent companion, executed notebook, captions, transcripts and validation evidence.
- `output\course_manifest.json` and `delivery_manifest.json`: measured metadata and SHA-256 checksums.

The complete audio/video stream decodes successfully. Integrated loudness measures
**-16.00 LUFS**; measured true peak is **-1.48 dBTP** after AAC encoding
(the normalization filter targets -1.5 dBTP). Every chapter measures -16.00 or
-16.01 LUFS. All authored narration blocks match the rendered timelines.

The independent companion passes 24 tests; its notebook executes 35 code cells
across all sixteen chapters. The continuation's 34 displayed snippets execute,
and 18 foundation checks include a second backward pass after the scalar update.
The preserved review adds seven executed snippets.

Bounds checks, targeted frame exports and full decode checks are automated evidence,
not an exhaustive human viewing or independent perceptual subtitle-sync audit.
Drive delivery is verified separately after packaging; see the publication section
and `QUALITY_CHECKLIST.md` rather than inferring upload from local build success.

## Independent companion

`companion\tests.py` verifies every Chapter 1 calculation and the report exercise.
`companion\gds_part_reconstruction.py` independently implements embeddings, relation
aggregation, gates, explicit multihead attention, residual blocks, class attention,
readout and pair features. It does not import the production model.
Tests may import production code only as an independent comparison oracle.
The executed notebook covers all sixteen chapters. Its CPU demonstration is
synthetic memorisation, not a JetClass training or performance result.
See `companion\README.md` for commands, intentional implementation distinctions,
parameter audits, numerical outputs and validation artifacts.

The four model counts are **1,280,684** (six-block v1), **2,074,420**
(reported shared hybrid), **2,488,116** (default conditioned hybrid), and
**2,024,756** (local bias-only proxy). The last is not an exact reproduction of
published Particle Transformer. The relation tensor is fixed within a forward
pass; conditioned summaries can still change with particle state.

## Publication

The combined `GDS_ParT_Chapters_00_01_REVIEW.mp4` was uploaded through the user's signed-in Edge session.
Its exact filename was verified after reloading the [supplied Google Drive folder](https://drive.google.com/drive/folders/1BiAyQ9qpmuspf_isz8S5o-xjsSFthccK),
and its Drive preview opened with a video player. This delivery is the two-chapter review, not the rejected slide video.
Additional course deliverables must each be verified after upload; the review's
successful upload does not establish delivery of later chapters.
Generated assets and environments are ignored by Git and excluded from the documentation site.

**Complete-course delivery verified on 1 November 2026 at 07:56 IST.**
Drive reported **18 uploads complete**, and all eighteen exact filenames persisted
after reloading the supplied folder: the complete MP4, all sixteen individual
chapter MP4s, and the companion/subtitle/transcript/source bundle.
`output\drive_upload_verification.json` records that check; the delivered local
files still match the sealed SHA-256 manifest. The packaged checklist records
build-time status, while this external delivery receipt follows packaging.
No claim of exhaustive Drive playback or human perceptual review is made.
