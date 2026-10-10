# Day12 short-video continuation protocol

This file and the continuation drivers were reconstructed after workspace maintenance removed the uncommitted working tree. They are not claimed to be byte-identical to the code that produced the preserved preparation evidence. Historical manifests retain their original source hashes. No actual Wan video-denoising run is claimed by this protocol.

## Question

Can the saved image-JEPA translator produce useful motion guidance for a small pretrained video generator? The next experiment renders actual nine-frame clips. It tests the image-edit-to-video-conditioning route, not learned JEPA dynamics, object discovery, general physics, face identity or production readiness.

## Fixed inputs and route

- Two previously seen image scenes, seeds 14000 and 14001, are development fixtures. They are not a fresh generalization test.
- The selected object follows nine horizontal offsets: `0,16,32,48,64,48,32,16,0` pixels for scene 14000 and their negatives for scene 14001. Object selection comes from the procedural source mask. Motion is explicitly commanded, not inferred by JEPA.
- The frozen Day11 CNN and exact-token provenance/appearance transport produced independently decoded image guides. The prepared arrays and manifests are immutable and hash checked.
- The official Wan RGB-video conditioning route re-encodes each complete guide sequence with the causal video VAE. Stacked image latents are not injected as video latents. Nine frames map to three native latent time positions.
- Source RGB supplies appearance through the prepared guides. Reference-image slots are not enabled. Target RGB enters the oracle arm and offline scoring only.

## Fixed comparison

Render all five arms for both scenes and both noise seeds, 1111 and 2222: 20 calls total.

| Arm | Input guide | Purpose |
| --- | --- | --- |
| `source_static` | Repeated reconstructed source | Baseline with the same motion prompt but no explicit motion guide |
| `jepa` | Edited JEPA features translated with the frozen CNN and source appearance transport | Primary candidate |
| `no_f` | Same provenance transport, source appearance and deterministic local fill without CNN predictions | Tests the learned translator's added value; not a no-JEPA/no-provenance control |
| `wrong_direction` | Candidate guide with reversed command | Tests whether the generator follows supplied control |
| `oracle` | Procedural target RGB sequence | Diagnostic upper reference for control fidelity |

Use Wan-AI/Wan2.1-VACE-1.3B-diffusers revision `ec4d2cb062b548996b179d493fdd05340de702a1`, diffusers 0.35.1, 384×384, nine frames, 20 inference steps, CFG 5 and conditioning scale 1. Use FP16 transformer, FP32 VAE, native SDPA and four CPU threads. The positive motion prompt varies only by requested left/right direction; all arms of a scene share it. Text encoding uses FP32 UMT5 and stores finite FP16 outputs at sequence length 128.

The renderer creates initial noise once per seed and clones the same tensor for every arm and scene. It saves hashes, final denoised latents, full uint8 RGB arrays, display GIFs, runtime and errors. All masks are zero: the entire guide is treated as known VACE conditioning. There is no target-image loss or training during generation.

## Evaluation and stopping

Evaluate trajectory/centroid error and missing detections, moved-object texture, source-hole repair, unchanged distractor/background, temporal variance and loop closure. Inspect saved videos alongside these procedural metrics. Compare each same-scene/same-noise pair; show both scenes and seeds rather than selecting a favorable clip. A clear result can include failure: distinguishing a broken guide, loss during video-VAE encoding and loss during denoising is useful.

Prepared guide and causal-VAE roundtrip evidence alone cannot establish generator improvement. The final conclusion requires completed actual video renders. Report development observations rather than statistical significance or an unseen-scene claim. If errors, nonfinite tensors or memory failures occur, preserve them and stop; changing precision, settings or code requires a new output directory and a separately described run. Resume only when code, inputs, model bytes and runtime bindings match exactly. Do not overwrite historical evidence or rename partial output as complete.
