# Day11 progress: trained direct JEPA → Wan bridge

Status: training and eight-scene development latent evaluation complete.
Full-resolution VAE diagnostics and real Colab VACE rendering are in progress.
No fresh140xx scene or temporal-video test has been accessed. This is a progress
record, not a claim that the broad consistency problem has been solved.

## What was implemented and trained

Dense native V-JEPA2.1 features [1024,24,24] map to the actual normalized Wan VAE
conditioning tensor [16,48,48]. A pinned, scoped pipeline adapter injects these
features directly into the pretrained VACE branch. No intermediate diagnostic
RGB image is generated and re-encoded for conditioning. The JEPA encoder,
126,892,531-parameter Wan VAE and denoiser stay frozen.

Two fixed-budget mappings were trained on64genuineimages from32scenes:
218,112-parameter spatial CNN and65,728-parameter linear baseline. Both use
150epochs, identical data/batch ordering and final-epoch selection. Sixteen
genuine images from eight separate development scenes were evaluated. These
are development results; they are not an untouched final test.

| Genuine-image bridge | Development normalized latent MSE |
|---|---:|
| Training-average latent map | 0.09795445 |
| Linear mapping | 0.02308920 |
| Spatial CNN | 0.01749945 |

## Intervention result before diffusion

The editor receives source JEPA, a supplied source object mask, and prescribed
horizontal displacement. The learned mapping receives features and destination
coordinates only. Target RGB, target masks and target JEPA are supervision or
explicitly marked oracle diagnostics; they are not available to the edit arm.

All8development scenes improve both source-hole and destination latent errors
under the correct edited condition. Genuine-target JEPA and edited JEPA perform
similarly, which separates edit quality from the current reconstruction problem.

| CNN condition/route | Hole error ratio | Destination ratio | Global latent MSE |
|---|---:|---:|---:|
| Unedited true source | 1 | 1 | 0.01369828 |
| Absolute genuine-target JEPA (oracle) | 0.05227 | 0.33831 | 0.01745512 |
| Absolute edited JEPA | 0.08863 | 0.34768 | 0.01783562 |
| Source + translated genuine change (oracle) | 0.30184 | 0.41804 | 0.00536261 |
| Source + translated edited change | 0.32712 | 0.42620 | 0.00538052 |

Regional ratios are means of per-scene ratios versus true-source/no-edit error;
global MSE is the equal-scene mean. Keep these aggregation rules separate.
Protected-region ratio denominators can approach zero, so inspect absolute MSE
and change from the decoded source instead of quoting million-fold ratios.
Wrong-direction and shuffled conditions remain in the saved complete results.
Shuffled residuals fail the destination test even if their balanced average
ratio falls below1; a combined score alone is insufficient.

## Failure and current repair hypothesis

Absolute mapping reconstructs coarse position/color but loses fine texture and
changes untouched content. Adding a source residual protects background and
the distractor, but leaves visible texture ghosts at the old object position.
This also happens with genuine-target JEPA, so edited features alone do not
explain it.

Let A be the real Wan encoder and F the learned bridge. The current residual
route is F(j_edit) + [A(source) - F(j_source)]. The bracketed appearance detail
stays at its old coordinates. A separate development follow-up will test moving
that residual using exact correspondences carried by our copied JEPA tokens.
It will include a transport-without-F local-fill comparator. Such matching is
specific to the explicit token-copy editor; it is not proof of general semantic
alignment or JEPA-specific advantage.

## Execution and audit

- Live Colab: TeslaT4,15,360MiB total/14,913MiB initially free; Python3.13.15,
  Torch2.11.0+cu130. Actual GPU generation is under way.
- Targets and mapping training: cached native FP32 JEPA features; CPU Torch2.9.0,
 4threads. Wan targets are posterior modes, normalized with official statistics.
- Both model weights, full learning curves, source/data bindings, real VAE
  manifests and fixed renderer bundles are saved in `run_2026-10-10/`.
- Independent recount reproduced all selected latent metrics exactly across
  both models/eight scenes, plus available decoded RGB metrics. No train/test
  leakage or checkpoint-selection issue was found.
- Resolved setup failures are retained: cached-prompt validation conflict in
  Diffusers0.35.1, missing optional ftfy, and GPU memory retained by a notebook
  exception. Saved evidence permitted a clean idle-kernel restart.

No claim is made yet about production video, face/part identity, learned physics,
novelty, or superiority to other control signals. Actual diffusion results and
any repair attempts will be added before drawing broader conclusions.
