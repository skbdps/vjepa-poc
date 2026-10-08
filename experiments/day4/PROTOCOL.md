# Persistent video parts: next experiment

This stage tests whether one initial part annotation can remain attached to the same object through motion, overlap, scale changes, and occlusion. The practical deliverable is a selective visual edit driven by predicted part masks. A localized recoloring demo is not generative video editing or a learned latent decoder.

## Before viewing new test results

- Preserve Day 3 as completed development evidence. Its six evaluation clips are no longer a fresh test for improvements developed after seeing them.
- Use separate calibration, development, and final synthetic test seeds. Calibrate presence thresholds only on calibration clips; choose algorithm configuration only using development clips. Freeze the selected configuration before running the final test.
- Give every method only the first frame's part annotation. Later masks are restricted to evaluation, calibration, and diagnostic rendering. V-JEPA features can attend within each 16-frame input window, so this is offline tracking.
- Compare initial V-JEPA global matching, a stronger image template/flow baseline, and persistent feature/motion tracking, with component ablations.
- Report exact counts, per-condition results, and uncertainty resampled by clip. Avoid treating correlated frames as independent samples.
- Evaluate a real car video separately. Establish first-frame door/window masks, propagate them, and compare against sparse human annotations when feasible. Record annotation provenance and do not claim performance on an unannotated clip.
- Save failures and limitations along with successful demonstrations. Do not substitute ground-truth masks into an edit presented as tracked.

## Meaningful progress

Evidence should include a runnable pipeline, an observable selective edit attached to a named part across frames, quantified spill/miss or localization measurements, and an honest comparison with a reasonable baseline. The broader goals of arbitrary faces, large viewpoint changes, 3D understanding, and generated-video consistency remain open until tested directly.
