# Day7 learned persistent-part heads

All predetermined training seeds are reported. Test results select no model.

| Method | Visible localization | False presence while absent | Wrong car given visible | Recovery |
|---|---:|---:|---:|---:|
| global_full | 46.210% (890/1926) | 5.019% (13/259) | 8.827% (170/1926) | 22.222% (4/18) |
| selected_day4 | 49.117% (946/1926) | 10.039% (26/259) | 8.152% (157/1926) | 16.667% (3/18) |
| global_projected_centered | 39.304% (757/1926) | 12.355% (32/259) | 6.957% (134/1926) | 27.778% (5/18) |
| retrieval_only_seed1701 | 91.121% (1755/1926) | 0.386% (1/259) | 4.309% (83/1926) | 5.556% (1/18) |
| predictive_seed1701 | 91.485% (1762/1926) | 1.931% (5/259) | 4.361% (84/1926) | 5.556% (1/18) |
| coordinate_only_seed1701 | 38.577% (743/1926) | 1.931% (5/259) | 2.233% (43/1926) | 0.000% (0/18) |
| retrieval_only_seed1702 | 92.991% (1791/1926) | 5.019% (13/259) | 3.998% (77/1926) | 5.556% (1/18) |
| predictive_seed1702 | 95.846% (1846/1926) | 8.494% (22/259) | 2.752% (53/1926) | 50.000% (9/18) |
| coordinate_only_seed1702 | 38.993% (751/1926) | 8.494% (22/259) | 3.375% (65/1926) | 0.000% (0/18) |
| retrieval_only_seed1703 | 93.769% (1806/1926) | 5.019% (13/259) | 3.479% (67/1926) | 11.111% (2/18) |
| predictive_seed1703 | 91.277% (1758/1926) | 0.772% (2/259) | 4.413% (85/1926) | 5.556% (1/18) |
| coordinate_only_seed1703 | 35.618% (686/1926) | 8.108% (21/259) | 1.661% (32/1926) | 0.000% (0/18) |

`coordinate_only` receives no image/JEPA content; strong performance would expose procedural-position shortcuts.
The predictive arm differs from retrieval_only only in future-target auxiliary loss. Both architectures feed delayed forecasts into later retrieval.
Future latent cosine and paired per-seed clip intervals are in results.json; raw per-target forecasts are in future_rows.csv.
This is supervised part-label learning over a frozen JEPA backbone, not self-supervised JEPA training. No real-video, face, or generative claim follows.
