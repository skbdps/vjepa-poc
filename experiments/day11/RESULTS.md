# Day11 progress: trained direct JEPA → Wan bridge

Status: training, eight-scene development latent evaluation, and the initial
CNN's full-resolution VAE diagnostics are complete. Both initial translators
have completed real VACE rendering on the two fixed development scenes.
The appearance-transport GPU follow-up is running.
No fresh 140xx scene or temporal-video test has been accessed. This is a progress
record, not a claim that the broad consistency problem has been solved.

## What was implemented and trained

Dense native V-JEPA2.1 features [1024,24,24] map to the actual normalized Wan VAE
conditioning tensor [16,48,48]. A pinned, scoped pipeline adapter injects these
features directly into the pretrained VACE branch. No intermediate diagnostic
RGB image is generated and re-encoded for conditioning. The JEPA encoder,
126,892,531-parameter Wan VAE and denoiser stay frozen.

Two fixed-budget mappings were trained on 64 genuine images from 32 scenes:
218,112-parameter spatial CNN and 65,728-parameter linear baseline. Both use
150 epochs, identical data/batch ordering and final-epoch selection. Sixteen
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

All eight development scenes improve both source-hole and destination latent errors
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
ratio falls below 1; a combined score alone is insufficient.

## Actual generator result

All four ordinary-RGB versus direct-latent oracle comparisons produced
bitwise-identical denoised tensors. Both initial models then completed every
declared arm using the same prompt, random seed, scheduler and 20 steps on a
Tesla T4. These are actual diffusion outputs, not VAE reconstructions.

| CNN input | Target-centroid error, scene 13500 | Scene 13501 |
|---|---:|---:|
| True source, no edit | 15.526 px | 16.453 px |
| True target, oracle | 0.599 px | 0.666 px |
| Absolute edited JEPA | 1.919 px | 1.805 px |
| Absolute wrong direction | 29.847 px | 32.874 px |
| Source + translated edited change | 4.260 px | 3.026 px |

The position command survives the real generator. The absolute route loses
texture and changes background; the stationary residual leaves visible ghosts.
The linear comparator also moves the object, so CNN-specific superiority is
not established by localization alone. Complete outputs, negative controls,
runtime logs and independent metric recounts are retained.

## Failure and current repair hypothesis

Absolute mapping reconstructs coarse position/color but loses fine texture and
changes untouched content. Adding a source residual protects background and
the distractor, but leaves visible texture ghosts at the old object position.
This also happens with genuine-target JEPA, so edited features alone do not
explain it.

Let A be the real Wan encoder and F the learned bridge. The current residual
route is F(j_edit) + [A(source) - F(j_source)]. The bracketed appearance detail
stays at its old coordinates. A separate development follow-up moves that
residual using exact correspondences carried by our copied JEPA tokens.
It includes a transport-without-F local-fill comparator. Such matching is
specific to the explicit token-copy editor; it is not proof of general semantic
alignment or JEPA-specific advantage.

| Development follow-up | Mean balanced latent-region ratio |
|---|---:|
| Original stationary residual | 0.376662 |
| Copy source tiles + learned fill | 0.074002 |
| Copy source tiles + local fill, no learned F values | 0.086643 |
| Transported source residual + learned fill | 0.067603 |

These ratios average hole and destination source-relative errors across all
eight development scenes. An independent implementation reproduced every
transport tensor exactly. The no-F comparator is strong: most improvement is
from explicit source appearance transport. A smaller learned contribution
remains to be assessed on untouched scenes. Decoded examples show removal of
most stationary ghosting; actual generator qualification remains pending.

The subsets must be kept separate. On the same two decoded scenes, no-F local
fill beats the learned transported residual in both latent MSE (0.000407776
versus 0.000465629) and decoded RGB MSE (0.000209462 versus 0.000217999).
The learned route wins the eight-scene latent average, not every scene. Mixing
the eight-scene latent mean with the two-scene RGB mean would incorrectly imply
a demonstrated reversal between those metrics. The independent attribution
audit records this correction and complete matched comparisons.

The fixed engineering gate in `qualify_development.py` was published before
inspecting the transport GPU outputs. It requires per-region editing accuracy,
centroid accuracy, protected-content tolerance, negative-control separation,
and exact runtime/interface bindings. Passing it authorizes an untouched test;
it is not a declaration of production reliability.

## Execution and audit

- Live Colab: Tesla T4, 15,360 MiB total / 14,913 MiB initially free;
  Python 3.13.15, Torch 2.11.0+cu130. Initial CNN and linear generation completed.
- Targets and mapping training: cached native FP32 JEPA features; CPU Torch2.9.0,
  four threads. Wan targets are posterior modes, normalized with official statistics.
- Both model weights, full learning curves, source/data bindings, real VAE
  manifests and fixed renderer bundles are saved in `run_2026-10-10/`.
- Independent recount reproduced all latent metrics exactly across
  both models/eight scenes and all transport tensors, plus initial CNN GPU RGB
  metrics to floating-point precision. No train/test
  leakage or checkpoint-selection issue was found.
- Resolved setup failures are retained: cached-prompt validation conflict in
  Diffusers0.35.1, missing optional ftfy, and GPU memory retained by a notebook
  exception. Saved evidence permitted a clean idle-kernel restart.

No claim is made about production video, face/part identity, learned physics,
novelty, or superiority to other control signals. See `METHOD.md` for the exact
interfaces and the reason source appearance is retained separately.
