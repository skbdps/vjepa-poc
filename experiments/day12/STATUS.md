# Day12 status: partial completion; Colab sign-in required

The main 20-clip Wan/VACE generator comparison has **not run**. Google Colab
was signed out. The secure account-selection handoff did not complete, so no
GPU or notebook runtime could be inspected. This was an authentication blocker,
not a failed generated-video experiment.

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

After Google sign-in is restored, inspect the existing Colab runtime and
recover any earlier outputs first. Then run the fixed small Wan comparison:
two scenes, noise seeds 1111 and 2222, five conditions, nine 384×384 frames,
20 steps, CFG 5, conditioning strength 1, FP16 transformer and FP32 VAE.
No paid video-generation API key is required. T4 feasibility for this exact
configuration still needs an observed GPU run.
