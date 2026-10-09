# Editor validation record

Validated on 2026-10-09 with the two completed SAM2 mask sequences and in the executed Google Colab notebook.

## Rendering and control checks

- Both real sequences render their initial frames in the same Colab output. Each editor has root-scoped JavaScript and unique IDs for its sequence and mask archive.
- On car-roundabout, enable the window while preserving the enabled blue door: the UI reports two active edits. Window strength changes to 45% while door strength remains 75%.
- Frame scrubbing to frame 32 updates the original and composited views together. Playback reaches the end. Stable part tags can be displayed on the preview.
- The second editor stays at frame zero with only its door enabled while the first editor is manipulated.
- Reset reports zero active edits and zero permitted edit pixels.
- Browser recipe export produces exactly the same parsed JSON as `run_2026-10-09/car_roundabout/dual_part_edit.recipe.json`. The downloaded export is preserved as `browser_export.recipe.json`.
- The Colab embedded file chooser did not open through the automation interface, so browser recipe import is **not end-to-end verified in Colab**. Validation and native replay of the exported recipe passed; no successful browser import is claimed.

## Array-level verification

The build checks exact RLE mask roundtrips (208 part-frame masks across both clips) and frame bounds. The compositing checks cover outside-mask and overlapping-mask preservation. All 104 native replay frames report zero modifications outside enabled, nonoverlapping predicted regions. Invalid recipe schemas, wrong sequence/part IDs, and invalid control values are rejected by the recipe validator. JavaScript/Python color arithmetic was compared on random RGB inputs at multiple strengths.

These checks protect predicted regions before lossy video encoding. They cannot establish anatomical correctness, and browser video compression means browser pixels are not bit-identical to native JPEG replay.

## Scope and remaining limitations

This is a research editor with two manually initialized parts, saved mask tracks, and deterministic color controls. Browser PNG export is implemented but was not separately exercised in this final browser pass. Loading two copies of the exact same generated sequence in one HTML document would reuse its deterministic root ID; use one instance per generated sequence or separate documents. The fresh reproduction notebook was statically checked; the model work was executed in the saved research notebook.
