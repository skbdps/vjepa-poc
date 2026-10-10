# Alignment with the original latent-editing idea

Read-only design audit, 2026-10-09. This assessment compares the implemented
Day8 protocol, operators and evaluator with the original
[Day2 move test](https://github.com/skbdps/vjepa-poc/blob/main/experiments/day2/move_test.py).
It makes no claim about Day8 experimental outcomes.

The original test copied selected ball embeddings to a different spatial
location in one temporal slice, filled the source with a background mean, and
displayed separately fitted PCA projections before and after. That was a useful
exploration of representation manipulation. It did not compare the edited
embeddings with a genuinely edited video's encoding, invoke a predictor, or
decode edited RGB frames. A moved cluster in that visualization alone cannot
establish that the edited representation describes the intended scene.

Day8 directly addresses that missing comparison. It independently encodes paired
source and target videos, whose selected ball differs by a known displacement,
and compares source-only latent edits against the target encoding. This tests
whether `edit(encode(source))` approaches `encode(target)`. Separate destination
and vacated-source measurements prevent a global average from hiding an edit
that moves content but fails to remove its original occurrence. This is a
direct continuation of latent editing; the Day3–7 localization and tracking
work addressed an adjacent component.

The frozen occupancy/RGB probe is a useful sanity check, with an important
qualification. It reads each token independently, so relocating a token also
relocates that token's probe output by construction. Successful probe relocation
alone therefore does not demonstrate a coherent target representation. The
independently encoded target comparison supplies the additional evidence. Exact
outside-support preservation and much of distractor preservation are likewise
enforced by the operator's spatial support, rather than learned achievements.

Full-sequence source masks and a displacement for the complete trajectory are
provided as an explicit oracle. Temporal measurements assess how well an edit
is executed across that supplied trajectory. They do not establish propagation
from one edited frame, automatic part selection, or intervention-aware future
prediction. The probe is also a coarse readout, not a video decoder.

Interpret the completed experiment along these failure axes:

- If the probe fails on genuine targets, semantic conclusions are inconclusive.
- If latent errors improve but placement remains inaccurate or ghosts remain,
  report partial representation improvement and the specific remaining failure.
- If probe relocation succeeds without target-latent improvement, copied
  readout outputs do not by themselves validate the intended representation.
- If deterministic transport succeeds, that supports a simpler editing
  mechanism; learning must beat both transport baselines to establish added
  value. Matching geometry-only correction cannot establish a content-input
  advantage, and that comparison never isolates the sequence anchor alone.

Keep the frozen decision rules unchanged. Interpret their relative-improvement
flags alongside absolute placement, ghosting, appearance and temporal errors.
A useful positive result would support controlled latent translation within
this restricted setup, while pixel-video rendering and broader part/identity
control remain separate tests.
