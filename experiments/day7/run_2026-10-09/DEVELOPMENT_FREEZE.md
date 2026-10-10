# Development freeze — before held-out access

Recorded 2026-10-09T06:34:02.831558+00:00. All nine predetermined 30-epoch runs completed on a Colab Tesla T4. No training adjustment or OOM fallback was used. The 18 test clips had not been opened at this checkpoint.

Freeze SHA-256: `d22db32b2fe60d648170ed38970b6dcb274f322cc5630b020db6d996f319dd62`.

| Arm | Seed | Selected epoch | Development utility |
|---|---:|---:|---:|
| retrieval_only | 1701 | 22 | 95.2282% |
| predictive | 1701 | 28 | 96.0581% |
| coordinate_only | 1701 | 18 | 68.4853% |
| retrieval_only | 1702 | 27 | 93.8846% |
| predictive | 1702 | 29 | 97.3029% |
| coordinate_only | 1702 | 30 | 67.0923% |
| retrieval_only | 1703 | 24 | 94.5070% |
| predictive | 1703 | 30 | 95.6432% |
| coordinate_only | 1703 | 11 | 66.4468% |

Development utility averages: retrieval-only 94.5399%, predictive 96.3347%, position-only 67.3415%. The predictive difference is +1.7948 percentage points; this is development evidence, not a held-out result.

Decision: retain all three seeds and the predetermined budget. Use each earliest best development epoch and the recorded baseline thresholds. Do not tune using test results. Source/model/data/extraction files remain byte-identical to experiment commit `1d9f12a5f51b061e33a4417d2789c203b5d1dc86`; helpers are pinned at `9af4259467caa0bdf7980ef83d856fae1f70f61f`.
