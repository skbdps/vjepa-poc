# Native-image readout follow-up to the disclosed Day10 failure

The first Day10 test passed its latent-distance and localization checks but
failed its genuine-target coarse-color gate: the unchanged Day8 video-trained
probe scored zero correct colors on 24 native-image target states. Independent
auditing reproduced the outputs. Even the four zero-shift source states failed
color identification, and copying preserved their original probe colors
exactly. The failure is present before editing. Preserve that run and its
failed overall gate; do not reinterpret it as a complete successful animation.

This bounded follow-up changes only the diagnostic readout's training domain.
The official JEPA encoder, native image interface, source-only selection budget,
token-copy/spatial-fill editor, targets, metrics, and procedural image family
remain unchanged. No full-resolution image decoder, generator, tracker, or SAM
is introduced. A repaired probe cannot establish detailed visual identity.

## Fixed training and development data

Train on 32 independent seeds 13400–13431. Evaluate development validity once
on eight independent seeds 13500–13507. For each seed use frame 15 from the
unchanged Day8 renderer as one still source image and one separately rendered
genuine shifted target image. Both are encoded through the official native
image path, using CPU FP32. No other source frames enter the encoder or probe.
This yields 64 genuine training images and 16 genuine development images.

Training shifts cycle through `[16,-16,32,-32,48,-48,64,-64,80,-80]` pixels in
seed order. Development shifts are `[16,-16,32,-32,48,-48,80,-80]` in seed order.
Retain complete FP32 caches, image hashes, all labels and split specifications.
Neither original Day10 test seeds 13200–13203 nor fresh test seeds 13600–13603
enter training, normalization, model selection, or development assessment.
No edited latent tensors are training examples. Coarse occupancy covers both
objects; RGB targets are genuine 16×16 patch means including any background.

## Fixed model and optimization

Reuse the exact Day8 `FrozenReadout` architecture: train-standardized
1024-dimensional token input, linear layer of width 128, GELU, and a four-output
linear layer followed by sigmoid. Outputs are one occupancy fraction and three
RGB patch means. No coordinates, masks, displacement, scene ID, or motion
command are model inputs. Channel mean and standard deviation use training
features only, with the existing 1e-6 standard-deviation floor.

Freeze seed 1010, 40 epochs, 32 batches per epoch, batch size 512, AdamW learning
rate 0.001, weight decay 0.0001, and CPU FP32 with four threads. Use the unchanged
Day8 loss: occupancy MSE plus mean RGB-channel MSE. Each batch samples foreground
and background equally, uniformly within the corresponding training pool,
with replacement. Both scene color roles are included in the source data.

Use the final epoch only. Pass no development examples to the training helper,
so there is no checkpoint selection, early stopping, schedule adjustment,
architecture sweep, or post-development tuning. Preserve every epoch's losses,
normalization statistics, model checkpoint, exact configuration, and hashes.

## Development gate and fresh test

After the fixed training run, assess all 16 genuine development images once.
Require mean selected centroid error below 16 pixels, every selected centroid
present, and at least 90% correct coarse-color identity among eligible cases.
Report eligible and skipped counts, IoU, occupancy MSE, and RGB MSE. Development
is only a pass/fail check; if it fails, stop without opening fresh test images.

If it passes, publish and byte-verify the trained checkpoint binding,
development report, training provenance, and new test freeze before encoding
fresh seeds 13600–13603. Use exactly the original four-image Day10 geometry and
six-position scheme with these new seeds: first two positive shifts and last
two negative shifts at magnitudes `[0,16,32,48,64,80]`. The return display path
repeats positions and adds no observations. A thin versioned wrapper explicitly
replaces the test specifications and diagnostic probe; the original frozen
editor and evaluator source files remain untouched.

Retain all original Day10 requirements: genuine reference centroid and color
validity; every edited selected centroid present; mean edited centroid error
below 16 pixels; every image's mean source-hole, destination, and balanced
latent error ratio below no edit; nondegenerate primary denominators; and all
exact preservation/identity/revisit guarantees. Additionally require at least
90% edited coarse-color identity among all eligible fresh states, including
zero displacement. Report eligibility/skips and every failed state.

Average nonzero positions within each image, then weight the four images
equally. Results are descriptive; there are no confidence intervals or broad
generalization claims. Report the original failed transfer and this separate
follow-up together. Success would support controlled single-image latent
relocation plus a calibrated coarse readout in this synthetic family, not
photorealistic image animation, automatic motion prediction, background truth
recovery, face consistency, or a new JEPA architecture.

## Publication sequence

Before any new train/development encoding, publish this protocol, source hashes,
all data specifications, and fixed hyperparameters. The pretraining receipt
records `training_encoded_before_verification: false`. After training, publish
a second freeze binding the final checkpoint and passing development evidence;
the fresh-test receipt records `test_encoded_before_verification: false`.
Every benchmark stage validates its relevant source, checkpoint, and evidence
hashes. Independent auditing may recompute normalization, tensor operations,
probe predictions, and metrics from saved caches; it does not constitute an
independent rerun of encoder inference or optimizer training.
