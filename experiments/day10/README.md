# Single-image JEPA animation proof of concept

This experiment asks whether one image, one selected-object mask, and an
explicit movement command can produce a sequence of controlled JEPA edits.
It uses V-JEPA2.1's native image input, without a source video or temporal
background donors. The existing spatial-fill/token-copy operator and frozen
coarse readout are reused; no new generative model is trained.

The [protocol](PROTOCOL.md) fixes four fresh images and five nonzero movements
per image. Target images are independently encoded scoring references only.
Results are pending the pretest freeze and run. The output is a24x24diagnostic
animation, not a photorealistic image-to-video system.
