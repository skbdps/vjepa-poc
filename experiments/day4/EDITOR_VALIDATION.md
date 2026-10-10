# Editor validation record

Validated on 2026-10-09 with the two completed SAM2 mask sequences and in the executed Google Colab notebook.

## Rendering and control checks

- Both real sequences render their initial frames in the same Colab output. Each editor has root-scoped JavaScript and unique IDs for its sequence and mask archive.
- On car-roundabout, enable the window while preserving the enabled blue door: the UI reports two active edits. Window strength changes to 45% while door strength remains 75%.
- Frame scrubbing to frame 32 updates the original and composited views together. Playback reaches the end. Stable part tags can be displayed on the preview.
- The second editor stays at frame zero with only its door enabled while the first editor is manipulated.
- Reset reports zero active edits and zero permitted edit pixels.
- Browser recipe export produces exactly the same parsed JSON as `run_2026-10-09/car_roundabout/dual_part_edit.recipe.json`. The downloaded export is preserved as `browser_export.recipe.json`.
- Recipe import through **Paste a recipe → Apply recipe is end-to-end verified in Colab**. Export fills the JSON textarea; after Reset reports zero edits, applying that exported JSON restores both the blue door at 75% strength and the amber window at 45% strength, with two active edits.
- Applying an invalid recipe schema reports rejection and retains both existing edits. Applying valid JSON again clears the error and restores the settings.
- The Colab embedded file chooser did not open through the automation interface. The native file-picker import path remains **unverified in Colab**; the verified paste path provides an alternative.

The latest browser pass used Colab cell 17 with source pinned to commit `d153282`. It verified the export/reset/paste/apply round trip, invalid-schema rejection without state changes, and subsequent error clearing. The car-roundabout editor was left at frame 32 with stable tags and two active edits; car-shadow remained at frame zero with only its door enabled.

The earlier browser pass used commit `66a91f969f583a245cf3e9f66977f24ae940bb55` and confirmed readable headings and isolated editor styles. Its exported research notebook had 16 valid Python code cells, 102 output blocks, and no cell-error outputs; the GitHub copy strips outputs and execution counts. Those export counts describe that earlier snapshot, not the subsequently updated live notebook.

## Array-level verification

The build checks exact RLE mask roundtrips (208 part-frame masks across both clips) and frame bounds. The compositing checks cover outside-mask and overlapping-mask preservation. All 104 native replay frames report zero modifications outside enabled, nonoverlapping predicted regions. Invalid recipe schemas, wrong sequence/part IDs, and invalid control values are rejected by the recipe validator. JavaScript/Python color arithmetic was compared on random RGB inputs at multiple strengths.

The paste-import change passed JavaScript syntax checks for both rebuilt editors and event-handler checks for valid restore, reset/restore, part-order normalization, error recovery, and isolation between editors. Five invalid-input cases per editor (malformed JSON, wrong sequence, invalid values in a later part, oversized ASCII, and oversized multibyte text) preserved every active part setting. Both file and paste imports use the same recipe validator before replacing the active recipe.

These checks protect predicted regions before lossy video encoding. They cannot establish anatomical correctness, and browser video compression means browser pixels are not bit-identical to native JPEG replay.

## Scope and remaining limitations

This is a research editor with two manually initialized parts, saved mask tracks, and deterministic color controls. Browser PNG export is implemented but was not separately exercised in this final browser pass. Loading two copies of the exact same generated sequence in one HTML document would reuse its deterministic root ID; use one instance per generated sequence or separate documents. The fresh reproduction notebook was statically checked; the model work was executed in the saved research notebook.
