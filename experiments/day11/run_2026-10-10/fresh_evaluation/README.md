# Frozen eight-scene evaluation

The complete evaluation is stored losslessly in seven ordered archive parts. It includes both translators, all 27 arms, all eight scenes, real Wan targets, scoring data, and full metrics.

Reassemble with `cat full_evaluation.tar.xz.part* > full_evaluation.tar.xz`, verify SHA256 `beb7547f9c111b98801217d3ddcc34821eecfe713be9b9d563a797c6a6819480`, then extract into an empty evaluation directory. Every enclosed file hash is listed in `full_evaluation.manifest.json`.

The renderer's `--bundle-root` is that extracted directory, containing `evaluation_manifest.json` and `cnn/scenes/`. The complete latent evaluation passed its eight per-scene primary thresholds; actual GPU image qualification is still pending. No method was changed after fresh access.
