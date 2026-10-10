# Day12 status: Kaggle GPU run in progress

Kaggle phone verification is complete. On 2026-10-10 at 22:55 UTC the private
notebook successfully allocated two Tesla T4 GPUs and passed the runtime
preflight: Python 3.13.15, PyTorch 2.11.0+cu128, 31.3 GiB host RAM,
and sufficient separate temporary storage. Pinned dependencies imported,
the exact guide hashes passed, and FP32 UMT5 prompt encoding completed.
The fixed 20-video render cell was launched. Four completed clips were
visible before refreshing the editor at approximately 23:05 UTC. A downloaded
manifest preserves the first three: 31.59–35.00 seconds per clip and
7.21–7.23 GiB peak CUDA allocation. This establishes T4 feasibility for these
clips, not completion or scientific success. The editor refresh resolved the
save conflict but is still reconnecting; the original compute job was not
explicitly stopped or restarted. Full completion is currently unverified. Runtime evidence is in `run_kaggle/`. A Kaggle draft
save-conflict warning occurred; a notebook backup was downloaded and the
runtime remained active. The access history below records earlier blocks.

Before this Kaggle launch, the main 20-clip comparison had not run. Google sign-in
succeeded on 2026-10-10 at approximately 22:25 UTC: the target Colab notebook
opened with a signed-in account control. Connecting its configured T4 then
showed: "You cannot currently connect to a GPU due to usage limits in Colab."
This is now a GPU allocation blocker, not an authentication failure or a
failed generated-video experiment. No compute purchase was made.

The existing notebook still contains the earlier Day11 output showing 43/104
fresh renders and all 16 parity checks passed. That is saved notebook output,
not a live runtime result. The runtime filesystem and any later completion
could not be verified because GPU connection was refused.

## Kaggle migration check

On 2026-10-10 at approximately 22:47 UTC, Google authentication to Kaggle
succeeded and a notebook named `VJEPA Day12 GPU Preflight` was created.
The accelerator menu showed `GPU T4 x2`, but it and Internet access were
disabled. Kaggle explicitly requires phone verification to enable these
features. The verification dialog includes a reCAPTCHA and phone-number entry;
the user must complete the secure browser handoff. No Kaggle GPU allocation,
runtime preflight, model download or generation has run yet.

The code/input migration was audited against GitHub checkpoint
`eeaa0f946317f4313c76063dc17034fc0f1132f2`: all 30 Day12 files match, and the
Day11 bridge, requirements and trained CNN remain available. Recovered guides
and video-VAE evidence total approximately 39 MB compressed. Core scripts
accept path arguments and do not depend on Colab APIs. A Kaggle launcher can
reuse them without retraining or repeating guide preparation.

Kaggle's saved-output folder must not hold the large model cache. Pinned
upstream downloads total approximately 19.04 GB; FP32 text-encoder disk
offload needs additional temporary space. The Kaggle notebook uses temporary
storage for models and prompt processing, retaining only small prompt caches,
generated outputs and metrics in `/kaggle/working`. Actual GPU memory fit and
free temporary disk still require live preflight after verification.

The CPU stages completed before the interruption:

- Two nine-frame scenes, each with five guide conditions: unchanged source,
  JEPA plus frozen translator, matched no-translator edit, wrong-direction edit,
  and genuine target oracle.
- Both JEPA guides follow all eight commanded motion steps and exactly return
  to the starting guide frame. Source appearance is transported using the
  existing Day11 method; no native temporal JEPA predictor is used.
- The actual Wan video VAE encoded each whole nine-frame JEPA guide into three
  temporal latent slices and decoded it back into nine frames. Its normalized
  condition has shape `[1,32,3,48,48]`; the two streams and nonzero encoded-zero
  reactive stream were verified. This is video encoding/decoding, **not Wan
  transformer denoising**.

The video VAE introduced measurable damage before generation:

| Scene | Guide position error | After video VAE | Correct motion steps |
|---|---:|---:|---:|
| 14000 | 0.717 px | 1.232 px | 8/8 |
| 14001 | 0.642 px | 1.318 px | 8/8 |

Object texture error also increased and first/last frames stopped matching
exactly. Thus the temporal interface works, but this lossy conversion does not
guarantee exact consistency. The image-trained translator is wrapped by
independent image decoding followed by whole-video encoding; it is not a
trained direct temporal adapter.

The preliminary guide comparison did not establish a learned-translator
advantage: the matched nonlearned control was slightly better on object
position and appearance. A controlled actual-generator comparison is still
needed. These are two previously inspected procedural development scenes, not
evidence about real faces, physical reasoning or production quality.

## Interruption and recovery

Workspace maintenance removed the new local checkout before its GitHub commit
completed. The two completed guide bundles, two video-VAE result bundles and
their manifests survived in temporary output directories. Their file hashes
and all 36 recorded array hashes were independently rechecked during recovery.
Those original output bytes and original manifests are preserved here.

The original uncommitted Day12 driver source files did not survive. Their
recorded hashes remain in the original manifests. Reconstructed drivers are
explicitly new source versions; they must not be described as byte-identical
to the code that produced these outputs. Existing Day11 source and trained
checkpoints remain available unchanged in GitHub.

## Evidence and next action

- `run_recovered/guides_manifest.json`: original preparation manifest.
- `run_recovered/video_vae_manifest.json`: original actual video-VAE checks
  and per-frame metrics.
- `run_recovered/evidence/`: lossless, chunked original NPZ/manifest archives.
- `archive_evidence.py`: reassemble and verify either archive.

When Colab GPU access becomes available, inspect the runtime and recover any
earlier outputs first. Then run the fixed small Wan comparison:
two scenes, noise seeds 1111 and 2222, five conditions, nine 384×384 frames,
20 steps, CFG 5, conditioning strength 1, FP16 transformer and FP32 VAE.
No paid video-generation API key is required. T4 feasibility for this exact
configuration still needs an observed GPU run.
