# Post-hoc Day5 development: other-parent veto

**The single tested alternative fails all four quality guards. Do not promote it to a fresh held-out run.** This analysis was proposed after viewing Day5 results. Day5's completed twelve-clip cohort is development data for this alternative; the original frozen Day5 results remain unchanged.

The fixed alternative is `M_i & ~P_other` for raw child masks and `E_i & ~P_other` for effective edits. `E_i` excludes every original raw child overlap, including disabled parts. There is no threshold, dilation, parameter search, new inference or new seed.

| Effective edit outcome | Baseline | Other-parent veto |
|---|---:|---:|
| Wrong-car pixels | 839 | 136 |
| Visible mean IoU | 93.0287% | 89.6022% |
| Pooled visible pixel recall | 93.0682% | 90.0176% |
| False presence while absent | 39/354 | 38/354 |
| Crossing mean IoU | 85.9344% | 76.6652% |
| Crossing pooled recall | 83.3765% | 74.9607% |
| Sliver-group mean IoU, 56 visible part-frames | 70.5979% | 59.7977% |
| Sliver-group pooled recall | 82.3748% | 70.4826% |

The alternative removes **703 wrong-car pixels but 211,396 correct visible pixels**. Raw-mask IoU and recall lose 3.3868 and 3.0524 percentage points; effective-edit IoU and recall lose 3.4265 and 3.0505 points. Every loss exceeds the existing 1-point guard. Long-occlusion and scale/camera results are unchanged by this veto; all additional damage occurs during crossings.

## Why the parent masks fail as unconditional constraints

Partitioning the baseline's correctly covered effective pixels by predicted parent support gives:

| Predicted parent support | Correct pixels |
|---|---:|
| Own parent only | 6,220,150 |
| Both parents | 107,020 |
| Other parent only | 104,376 |
| Neither parent | 17,875 |

Day5 own-parent containment lost the last two groups: **104,376 + 17,875 = 122,251** correct pixels. The alternative veto loses the middle two groups: **107,020 + 104,376 = 211,396**. Thus the parent masks disagree with correct child tracking at crossings. These categories describe observed mask geometry; they do not distinguish an internal identity switch from merged or leaking parent masks.

The veto preserves more of the sliver subgroup than own-parent containment, but it removes many correctly covered pixels where both parent masks overlap. Neither unconditional rule passes the overall criterion. Neither repairs missing part masks or proves identity recovery.

## Reproduce and interpret

The [script](explore_other_parent_veto.py) requires the completed Day5 run and its passed independent audit, verifies hashes, regenerates only the twelve existing Day5 scenes, and reproduces the unchanged baseline before evaluating the one alternative.

```bash
python experiments/day6/explore_other_parent_veto.py \
  --run-root /path/to/completed/day5_run \
  --out experiments/day6/day5_development_other_parent_veto.json
```

[Full JSON](day5_development_other_parent_veto.json) includes all raw/effective counts, conditions, clips, sliver metrics, the parent-support partition and provenance. No new GPU inference was used. Deploying this rule would still need the separate parent pass: Day5 measured 560.83 extra seconds over 984.73 child seconds, a 56.95% overhead.

All results here are **post-hoc exploratory development**, not a new held-out comparison or real-video evidence. Keep original overlap protection enabled and parent containment disabled by default. A subsequent candidate needs to address parent ownership errors or explicitly measure the cost and benefit of disclosed correction prompts.
