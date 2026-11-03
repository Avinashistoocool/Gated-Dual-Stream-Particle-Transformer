# Class 12 to GDS-ParT: a Hinglish video course

A beginner-oriented course that builds from school algebra to reconstructing this repository's gated dual-stream particle transformer. The completed video runs **2 hours 22 minutes 52 seconds**, with **16 lessons, 92 scenes, approximately 15,800 narrated words, and 34 minutes of explicitly labelled practice time**. Diagrams, equations and code are in English; narration mixes English technical explanations with Hindi guidance.

The voice is **synthetic**, generated locally with Piper. No narration text, project code or training data is sent to a speech service. Visuals use stepwise highlighting, tensor diagrams, illustrative attention maps, equations and code panels rather than footage of a live instructor. Attention heatmaps labelled illustrative are not measurements from a trained model.

## Watch the rendered video

The local render is in `docs\learning_video\output`:

| File | Purpose |
|---|---|
| `GDS-ParT_Hinglish_Course.mp4` | Full video, embedded chapter markers and selectable Hinglish subtitles |
| `watch.html` | Local browser player with clickable lesson navigation |
| `course.srt`, `course.vtt` | External subtitles |
| `transcript.txt` | Complete narration and exercise instructions |
| `chapters.txt` | Lesson start times |
| `manifest.json` | Measured narration, practice and total durations, source hash and voice settings |

Open the MP4 in VLC or another player supporting chapter markers and `mov_text` subtitles. Alternatively, open `watch.html` directly in a modern browser. Its captions are embedded in the player page, so subtitles and chapter navigation work locally without a web server. Keep the player, MP4 and transcript in the same folder. Ordinary servers without HTTP byte-range support can prevent long-video seeking; opening this player locally avoids that limitation.

Generated media, voice weights, build intermediates and the video Python environment are **local artifacts**, excluded from Git and the documentation site. A repository clone does not include the MP4. The scripts and original course content are versionable; no upload or publication is performed by rendering.

## Learning path

| Lesson | Subject |
|---|---|
| 1 | What a jet classifier predicts; features, parameters and labels |
| 2 | Functions, sums, indices, logs, probability and softmax |
| 3 | Vectors, dot/Hadamard products, matrices, tensors and broadcasting |
| 4 | MLPs, activations, derivatives, backpropagation and cross-entropy |
| 5 | LayerNorm, embeddings and residual connections |
| 6 | Four-vectors, all 17 particle features and padding |
| 7 | All four pair features, symmetry, self-pairs and relation embeddings |
| 8 | Attention as weighted averaging; Q/K/V and worked arithmetic |
| 9 | Multihead layout and the opposing mask conventions in PyTorch APIs |
| 10 | ParT-style additive pair bias and controlled local baselines |
| 11 | Shared summaries, query/key gates, identity initialization and gate gradients |
| 12 | Conditioned per-layer gates and exact pre-LN block ordering |
| 13 | Mean/CLS pooling, classification and permutation invariance |
| 14 | PyTorch reconstruction, module registration and numerical checks |
| 15 | Training, data caching, memory estimates, checkpoints and reproducibility |
| 16 | Metrics, ablations, interpretation limits, MoE and the final project |

Exercises include their answers after the work interval. Pause for longer when needed. The total includes **34 minutes of silent work time**; these intervals are intentional and visually labelled, not missing audio. The course is designed for several study sessions rather than one uninterrupted sitting.

## Reconstruct the model

Use the existing project environment with PyTorch and NumPy:

```powershell
python docs\learning_video\reconstruction.py
python docs\learning_video\reconstruction.py --train-demo
```

[`reconstruction.py`](reconstruction.py) explicitly assembles the top-level model from production component classes. It is a learning scaffold, **not an independent implementation of every component or of the original published ParT**. The exercises ask learners to replace individual operations and compare them against the reference.

The executable checks cover identical-weight forward and gradient agreement, shared/conditioned/sigmoid/no-gate variants, both pooling paths, padding isolation, consistent particle permutation, identity gates, manual attention, LayerNorm arithmetic, pair-feature arithmetic and parameter counts. The wrapper rejects malformed masks, shapes and empty jets. The production implementation is not changed by those educational guards.

The tiny optimization demo fits synthetic inputs. Its loss is a plumbing diagnostic, **not a jet-physics benchmark**. For real data and training, use the repository's existing [setup guide](../guide.md).

## Rebuild the video on Windows

The renderer uses a separate Python 3.13 environment; it does not install video dependencies into the training environment. Network access is needed for dependency and voice downloads, but speech synthesis and rendering are local.

```powershell
py -3.13 -m venv docs\learning_video\.venv
& docs\learning_video\.venv\Scripts\python.exe -m pip install -r docs\learning_video\requirements.txt
New-Item -ItemType Directory -Force docs\learning_video\voices
curl.exe -L --fail -o docs\learning_video\voices\hi_IN-pratham-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx
curl.exe -L --fail -o docs\learning_video\voices\hi_IN-pratham-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json
& docs\learning_video\.venv\Scripts\python.exe docs\learning_video\render.py --slides-only
& docs\learning_video\.venv\Scripts\python.exe docs\learning_video\render.py
```

Rendering uses Windows Segoe UI and Consolas fonts, 1600 x 900 H.264 slides at 5 frames per second, mono AAC speech, MP4 chapter metadata and embedded subtitles. Highlighting advances with narration segments. The low frame rate is intentional for instructional slides, not suitable for live-motion demonstrations.

The original bilingual script is [`course.txt`](course.txt). Its directives define lessons, scenes, points, equations, code, spoken paragraphs, practice intervals and answers. The renderer raises explicit errors for overflowing slide content and for a completed full course outside 120-180 minutes. `--preview` renders only the first scene; use a full render afterward to restore full-course sidecars. `--player-only` refreshes the browser player from an existing completed render without synthesizing or encoding again.

To check a completed full render, run the video environment's Python on `docs\learning_video\verify_media.py`. It decodes the complete audio/video streams and checks actual runtime, chapter count, subtitle timing and audio signal levels.

Narration WAVs are cached by text and synthesis settings. Video segments are cached by script/renderer hash and settings. The complete audio timeline is assembled from PCM samples, with segment boundaries aligned to video frames to prevent accumulated audio/subtitle drift.

## Sources, interpretation and voice attribution

Architecture and preprocessing follow `gds_part_colab.py` and `data\jetclass_cache.py`, inspected at repository commit `546c985`. The accompanying [technical report](../GDS-ParT_Technical_Report.pdf) supplies the broader mathematical reference. Foundational concepts are explained in original course prose.

- Qu, Li and Qian, [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772), and the [JetClass dataset](https://zenodo.org/records/6619768).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Ba, Kiros and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450).
- PyTorch documentation for [scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) and [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

The course distinguishes particle/relation streams from Q/K, shared checkpoints from per-layer defaults, signed score changes from simplistic "larger gate = more attention" claims, and test-time gate reliance from benefits established by retraining. It does not claim that preliminary single-seed differences establish a reliable advantage.

**Voice:** `hi_IN-pratham-medium`, trained by PravalX, distributed in the [Piper voices collection](https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/pratham/medium). The [model card](https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/pratham/medium/MODEL_CARD) identifies the AI4Bharat Indic corpus and lists its dataset license as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The model card is not a comprehensive statement of generated-output rights. Keep attribution with the artifacts and **do not assume commercial redistribution is cleared**; independently check applicable voice/model rights before publishing or monetizing the narration. Piper software, FFmpeg and the project's source code have their own licenses. Voice weights and fonts are not committed here.
