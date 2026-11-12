# Class 12 to GDS-ParT: a Hinglish video course

This course is for beginners. It starts from school algebra and continues until you can build again the gated dual-stream particle transformer of this repository. The completed video has a duration of **2 hours 22 minutes 52 seconds**. It has **16 lessons, 92 scenes, about 15,800 narrated words and 34 minutes of practice time with clear labels**. The diagrams, equations and code are in English. The narration uses English for technical explanations and Hindi for guidance.

The voice is **synthetic**. Piper makes it on the local machine. The course does not send narration text, project code or training data to a speech service. The visuals use highlights step by step, tensor diagrams, illustrative attention maps, equations and code panels. They do not show video of a live instructor. Attention heatmaps with the label "illustrative" are not measurements from a trained model.

## Watch the rendered video

The local render is in `docs\learning_video\output`:

| File | Purpose |
|---|---|
| `GDS-ParT_Hinglish_Course.mp4` | Full video, with embedded chapter markers and Hinglish subtitles that you can select |
| `watch.html` | Local browser player with lesson navigation that you can click |
| `course.srt`, `course.vtt` | External subtitles |
| `transcript.txt` | All the narration and the exercise instructions |
| `chapters.txt` | Start times of the lessons |
| `manifest.json` | Measured durations of the narration, the practice and the total, with the source hash and the voice settings |

Open the MP4 in VLC or in a different player that supports chapter markers and `mov_text` subtitles. Alternatively, open `watch.html` directly in a modern browser. The player page contains its captions. Thus, the subtitles and the chapter navigation operate locally without a web server. Keep the player, the MP4 and the transcript in the same folder. Some servers do not support HTTP byte ranges, and these servers can prevent seeking in a long video. If you open the player locally, this problem does not occur.

The generated media, the voice weights, the intermediate build files and the Python environment for the video are **local artifacts**. Git and the documentation site do not include them. A clone of the repository does not include the MP4. You can put the scripts and the original course content under version control. The render step does not upload or publish anything.

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
| 8 | Attention as a weighted average; Q/K/V and worked arithmetic |
| 9 | Multihead layout and the opposite mask conventions in PyTorch APIs |
| 10 | Additive pair bias as in ParT, and controlled local baselines |
| 11 | Shared summaries, query/key gates, identity initialization and gate gradients |
| 12 | Conditioned per-layer gates and the exact order of the pre-LN block |
| 13 | Mean/CLS pooling, classification and permutation invariance |
| 14 | Reconstruction in PyTorch, module registration and numerical checks |
| 15 | Training, data caching, memory estimates, checkpoints and reproducibility |
| 16 | Metrics, ablations, limits of interpretation, MoE and the final project |

Each exercise shows its answers after the work interval. If you need more time, pause the video. The total includes **34 minutes of silent work time**. These intervals are intentional and have visual labels. They are not missing audio. The design of the course is for several study sessions, not for one long session.

## Reconstruct the model

Use the project environment that has PyTorch and NumPy:

```powershell
python docs\learning_video\reconstruction.py
python docs\learning_video\reconstruction.py --train-demo
```

[`reconstruction.py`](reconstruction.py) builds the top-level model explicitly from the component classes of the production code. It is a learning scaffold. It is **not an independent implementation of each component or of the original published ParT**. The exercises tell learners to replace single operations and compare them with the reference.

The executable checks cover these items:

- the same forward pass and gradients with identical weights
- the shared, conditioned, sigmoid and no-gate variants
- the two pooling paths
- the isolation of padding
- a consistent permutation of the particles
- identity gates and manual attention
- the arithmetic of LayerNorm and of the pair features
- the parameter counts.

The wrapper rejects incorrect masks, incorrect shapes and empty jets. These educational guards do not change the production implementation.

The small optimization demo fits synthetic inputs. Its loss is a diagnostic of the connections in the code. It is **not a jet-physics benchmark**. For real data and training, use the [setup guide](../guide.md) of the repository.

## Rebuild the video on Windows

The renderer uses a separate Python 3.13 environment. It does not install the video dependencies into the training environment. You need network access to download the dependencies and the voice. The speech synthesis and the render occur locally.

```powershell
py -3.13 -m venv docs\learning_video\.venv
& docs\learning_video\.venv\Scripts\python.exe -m pip install -r docs\learning_video\requirements.txt
New-Item -ItemType Directory -Force docs\learning_video\voices
curl.exe -L --fail -o docs\learning_video\voices\hi_IN-pratham-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx
curl.exe -L --fail -o docs\learning_video\voices\hi_IN-pratham-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/pratham/medium/hi_IN-pratham-medium.onnx.json
& docs\learning_video\.venv\Scripts\python.exe docs\learning_video\render.py --slides-only
& docs\learning_video\.venv\Scripts\python.exe docs\learning_video\render.py
```

The render uses these settings:

- the Windows fonts Segoe UI and Consolas
- H.264 slides at 1600 x 900 and 5 frames per second
- mono AAC speech
- MP4 chapter metadata and embedded subtitles.

The highlights move forward with the narration segments. The low frame rate is intentional for instructional slides. It is not satisfactory for demonstrations with live motion.

The original bilingual script is [`course.txt`](course.txt). Its directives define the lessons, scenes, points, equations, code, spoken paragraphs, practice intervals and answers. The renderer shows explicit errors in two conditions:

- The content of a slide is too large for the slide.
- A completed full course has a duration that is not in the range 120-180 minutes.

`--preview` renders only the first scene. After a preview, do a full render to make the sidecar files for the full course again. `--player-only` updates the browser player from a completed render. It does not do the synthesis or the encoding again.

To check a completed full render, run `docs\learning_video\verify_media.py` with the Python of the video environment. It decodes all the audio/video streams. It checks the actual runtime, the number of chapters, the subtitle timing and the levels of the audio signal.

The narration WAVs are in a cache with the text and the synthesis settings as the key. The video segments are in a cache with the hash of the script/renderer and the settings as the key. The full audio timeline comes from PCM samples. The segment boundaries align with video frames, so the audio and the subtitles do not gradually go out of synchronization.

## Sources, interpretation and voice attribution

The architecture and the preprocessing follow `gds_part_colab.py` and `data\jetclass_cache.py` at repository commit `546c985`. The [technical report](../GDS-ParT_Technical_Report.pdf) is the broader mathematical reference. Original course text explains the basic concepts.

- Qu, Li and Qian, [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772), and the [JetClass dataset](https://zenodo.org/records/6619768).
- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
- Ba, Kiros and Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450).
- PyTorch documentation for [scaled dot-product attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), [MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html) and [LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html).

The course keeps these items separate:

- the particle/relation streams and Q/K
- shared checkpoints and per-layer defaults
- signed changes of the score and the simple claim "larger gate = more attention"
- dependence on the gates at test time and benefits that retraining shows.

The course does not claim that preliminary differences from a single seed show a reliable advantage.

**Voice:** `hi_IN-pratham-medium`, trained by PravalX, from the [Piper voices collection](https://huggingface.co/rhasspy/piper-voices/tree/main/hi/hi_IN/pratham/medium). The [model card](https://huggingface.co/rhasspy/piper-voices/blob/main/hi/hi_IN/pratham/medium/MODEL_CARD) identifies the AI4Bharat Indic corpus. It gives the dataset license as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The model card is not a full statement of the rights to the generated output. Keep the attribution with the artifacts.

**Do not think that commercial redistribution is permitted.** Before you publish or monetize the narration, examine the rights to the voice and the model independently. Piper software, FFmpeg and the source code of the project have their own licenses. This repository does not contain the voice weights or the fonts.
