# Day4 frozen test comparison

The development freeze selected **`no_memory`**. Test scores do not change this choice.

All methods use the same 18 synthetic clips and frame-zero part annotations. Later labels are scoring data only.

| Method | Visible localization ↑ | Hidden false presence ↓ | Recovery ↑ | Identity switches ↓ |
|---|---:|---:|---:|---:|
| `global_vjepa` | 55.4% (1064/1922) | 20.4% (53/260) | 38.9% (7/18) | 93 |
| `template_flow` | 63.2% (1215/1922) | 24.2% (63/260) | 0.0% (0/18) | 18 |
| `persistent_vjepa` | 56.3% (1082/1922) | 11.2% (29/260) | 27.8% (5/18) | 60 |
| `no_motion` | 55.6% (1069/1922) | 10.0% (26/260) | 27.8% (5/18) | 90 |
| `no_context` | 57.2% (1100/1922) | 25.0% (65/260) | 11.1% (2/18) | 65 |
| `no_memory` **(development-selected)** | 53.4% (1026/1922) | 5.4% (14/260) | 27.8% (5/18) | 60 |
| `sam2_1_tiny` | 95.0% (1826/1922) | 4.2% (11/260) | 33.3% (6/18) | 4 |

Paired differences below are `sam2_1_tiny` minus the development-selected `no_memory`.

| Metric | Difference | Paired clip bootstrap 95% interval |
|---|---:|---:|
| Visible localization (positive favors SAM2) | +41.6 pp | [+30.9, +50.1] pp |
| Hidden false presence (negative favors SAM2) | -1.2 pp | [-5.8, +3.4] pp |
| Recovery (positive favors SAM2) | +5.6 pp | [-12.5, +27.8] pp |

Intervals resample whole paired clips, not frames. They describe variation among these generated clips; they do not establish real-video generalization. The JSON includes every SAM2-versus-baseline and selected-variant-versus-baseline comparison.

SAM2 is a specialist mask tracker with sequential memory; V-JEPA supplies frozen patch features with offline attention inside independent 16-frame windows. SAM2 masks are reduced to a patch location for this table. Training objectives, temporal processing, internal resolution, and compute are not matched. SAM2 received quality-100 JPEG inputs; V-JEPA received original rendered RGB.

## SAM2 dense masks (separate readout)

Raw masks are scored on every frame after frame zero, without the calibrated patch-presence threshold. Any nonempty ground-truth part is visible, including thin slivers excluded from the patch table.

| Dense metric | Value |
|---|---:|
| Mean visible part-frame IoU | 94.4% |
| Visible pixel precision | 94.2% |
| Visible pixel recall | 95.1% |
| Pixel precision including hidden targets | 93.5% |
| False mask presence while hidden | 11.1% |

Dense denominators: 3997 visible and 539 absent part-frames across 18 clips. These differ from the patch benchmark denominators.

No model inference, threshold fitting, or configuration selection is performed by this comparison script.
