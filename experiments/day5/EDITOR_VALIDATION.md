# Four-part hierarchy editor: completed validation

Validation date: 2026-10-09. Scene: `test_crossing_6200`, selected before viewing its test results. This validates editor behavior and replay protection; it does not overturn the [negative parent-containment experiment](RESULTS.md).

The [standalone editor](run_2026-10-09/hierarchy/interactive_hierarchy_editor.html) uses the saved four child masks and two parent masks. Stable hierarchy IDs were assigned at frame zero; they were not discovered automatically. The [registry](run_2026-10-09/hierarchy/scene_registry.json) records those IDs, prompt masks and source provenance. Later truth masks are not embedded in the editor.

## Browser checks in the research Colab

The browser session exercised separate settings for both cars and all four parts. Switching the selected car retained previously configured part settings. The downloaded recipes contain:

| Stable ID | Enabled | RGB color | Strength |
|---|---|---|---:|
| `car_A.front_door` | Yes | `[42, 137, 240]` | 80% |
| `car_A.window` | Yes | `[249, 184, 53]` | 35% |
| `car_B.front_door` | Yes | `[80, 213, 151]` | 50% |
| `car_B.window` | No | `[219, 102, 218]` | 65%, retained while disabled |

Observed checks:

- Export produced an actual downloaded JSON file whose settings matched the controls.
- Reset cleared the active edits. Pasting the saved recipe restored all four settings, including containment ON.
- An invalid schema was rejected atomically: the previous valid settings remained unchanged.
- Playback reached frame 63, the last of 64 frames. Manual scrubbing reached frame 18.
- A second export saved the final state with containment OFF, retaining the same part settings.

[Browser screenshot](run_2026-10-09/hierarchy/browser_editor.jpg) shows both cars, stable tags and the three enabled edits.

Evidence: [containment-OFF browser export](run_2026-10-09/hierarchy/browser_export.recipe.json) and [containment-ON browser export](run_2026-10-09/hierarchy/browser_export_contained.recipe.json). These contain settings only, with no masks, executable code or model weights. Pasted-recipe import was exercised; this does not claim every possible browser file-chooser path was tested. The Save PNG button was **not verified**.

## Actual browser exports replayed natively

Both downloaded recipes were passed to `replay_hierarchy.py` against the completed frozen prediction caches. Each replay produced a 64-frame video including frame zero, a normalized recipe and a per-frame invariant audit, without model inference. This replay denominator differs from the benchmark, which excludes frame zero.

| Exported mode | Frames | All invariant checks | Video | Audit |
|---|---:|---|---|---|
| Containment OFF | 64 | Passed | [browser_replay.mp4](run_2026-10-09/hierarchy/browser_replay.mp4) | [browser_replay.audit.json](run_2026-10-09/hierarchy/browser_replay.audit.json) |
| Containment ON | 64 | Passed | [browser_replay_contained.mp4](run_2026-10-09/hierarchy/browser_replay_contained.mp4) | [browser_replay_contained.audit.json](run_2026-10-09/hierarchy/browser_replay_contained.audit.json) |

Across all 128 replayed frames, the recorded sums were zero for each invariant:

- Changes outside enabled parts' permitted masks.
- Changes inside original ambiguous child-mask overlaps.
- Changes inside disabled predicted parts.
- Changes outside the original unique-child regions.

The original raw child overlap veto applies regardless of which parts are enabled. Containment ON intersects that permitted region with the corresponding predicted parent; it never releases protected overlap pixels. The CLI verifies complete-run provenance before regenerating the exact source RGB. Each audit records source, prediction-cache, compositor, recipe and output-video hashes.

## Shared implementation and limits

Native replay imports the builder's recipe validator and compositor rather than maintaining a second format. Canonical string IDs bind settings independently of list order. Fabricated CPU/JavaScript tests additionally verify strict recipe validation, matching effective masks, and matching pixel arithmetic on identical RGB arrays: `floor((1-strength)*RGB + strength*color)`.

Browser preview uses compressed embedded video; native replay uses original rendered RGB. The resulting images need not be bit-identical because their inputs differ. Protection invariants apply to RGB arrays **before lossy MP4 encoding**, and refer to predicted masks rather than anatomical correctness.

Containment remains OFF by default because it failed the fixed Day 5 accuracy guards. The bounded [Day 6 other-parent veto](../day6/POSTHOC_DAY5_EXPLORATION.md) also failed all four guards using Day 5 caches as post-hoc development data, so it was not promoted to a fresh GPU validation run. Neither result establishes face identity, 3-D structure or generative consistency. The clean reproduction notebook remains an unexecuted end-to-end recipe for a fresh GPU runtime; the experiment and browser checks were performed in the research Colab.
