# V-JEPA 2.1 POC on Google TRC TPU

Research toward controllable AI video generation, including V-JEPA 2.1 latent-space manipulation. The consistency experiments below measure the capabilities demonstrated so far.

## Current consistency experiments

The ongoing work on `experiment/part-consistency-colab` tests whether a direct intervention in JEPA features expresses a requested edit consistently across a video.

The active research focus is [Day 8: direct latent editing against matched genuine videos](experiments/day8/README.md). It returns to the original Day 2 relocation idea, with source removal, destination accuracy, appearance and distractor preservation measured separately. Implementation and runtime validation are in progress; no Day 8 editing result is claimed yet.

The earlier [Day 7: a learned predictive JEPA part representation](experiments/day7/README.md) studied tracking, rather than validating latent edits. Nine custom heads were trained and tested on 18 unseen synthetic videos. Visible part localization improved from 46.21% with frozen JEPA matching to 92.87%, but the matched retrieval-only head already reached 92.63%: the extra future-prediction objective did not establish an overall tracking gain. Immediate recovery after occlusion remains unreliable. [Results and failure diagnosis](experiments/day7/RESULTS.md) explain the distinction. The earlier SAM2/editor experiments remain historical comparison evidence and infrastructure.

- [Day 3: initial part-localization diagnostic](experiments/day3/README.md)
- [Day 5: completed whole-car constraint experiment](experiments/day5/README.md) — all twelve held-out clips and an independent numerical audit completed; unconditional containment fails all four accuracy guardrails ([results](experiments/day5/RESULTS.md)).
- [Two-car, four-part hierarchy editor](experiments/day5/run_2026-10-09/hierarchy/interactive_hierarchy_editor.html) — independent stable car/part controls and portable recipes; containment stays off by default. Browser controls, recipe export/import and both containment-mode native replays are verified ([validation](experiments/day5/EDITOR_VALIDATION.md)).
- [Day 4: stronger benchmarks, real-car part masks, and selective editing](experiments/day4/README.md)
- [Offline interactive part editor](experiments/day4/run_2026-10-09/car_roundabout/interactive_part_editor.html) — download and open the HTML; control door/window colors independently, save a recipe, and replay it without another model run.
- [Real-car two-part edit preview](experiments/day4/run_2026-10-09/car_roundabout/dual_part_edit.mp4)
- [Reproduction notebook for Colab T4](experiments/day4/Part_Consistency_Colab.ipynb)
- [Results and limitations](experiments/day4/RESULTS.md)
- [Completed held-out specialist comparison](experiments/day4/run_2026-10-09/comparison/comparison.md)
- [Crossing and identity failure gallery](experiments/day4/run_2026-10-09/failures/README.md)

On eighteen held-out synthetic clips, SAM2.1 Tiny reaches 95.01% visible patch localization versus 53.38% for the selected V-JEPA tracker, a paired difference of +41.62 percentage points [95% interval +30.94, +50.10]. Its separate raw dense-mask mean IoU is 94.42%; raw masks remain the editing default because the supplementary presence gate suppresses thin visible parts. Identity recovery after occlusion remains unresolved, and the two real-car clips demonstrate localized recoloring rather than unseen-video or generative consistency.

The subsequent Day 5 experiment tests independent parent-car masks on twelve new synthetic clips. Parent intersection reduces effective wrong-car paint from 839 to 70 pixels, but removes 122,251 correctly covered pixels and drops effective visible IoU from 93.03% to 90.50%. All four frozen accuracy guards fail. This is a documented negative result for unconditional containment; the useful artifact is the reusable four-part control/replay pipeline. It does not establish face, 3-D or generative consistency. A bounded [Day 6 post-hoc other-parent veto](experiments/day6/POSTHOC_DAY5_EXPLORATION.md) also failed all four guards using these caches as development data, so it was not promoted to a new GPU validation run.

Run metadata, frozen configurations, metrics, small predicted masks, and selected previews are saved with the experiment. Full executed outputs stay in Colab; repository notebooks are kept without cell outputs.

## Infrastructure

- **TPU**: v6e-8 spot, zone `europe-west4-a`, project `llm-training-493207`
- **Storage**: `gs://vjepa-poc-eu` (co-located with TPU)
- **Stack**: Python 3.11, PyTorch 2.9.0, torch_xla 2.9.0, timm 1.0.26

## Recovery from spot preemption

When the TPU gets preempted and recreated:

```bash
# On your Mac — recreate the TPU via QR
gcloud compute tpus queued-resources create vjepa-main-v6-qr \
  --node-id=vjepa-main-v6 --zone=europe-west4-a \
  --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e --spot \
  --labels=owner=sanu,project=vjepa-poc,ephemeral=true

# Wait until ACTIVE, then SSH
gcloud compute tpus tpu-vm ssh vjepa-main-v6 --zone=europe-west4-a

# On the TPU — one-liner recovery
sudo apt install -y git && \
  git clone https://github.com/skbdps/vjepa-poc.git ~/vjepa-poc && \
  cd ~/vjepa-poc && ./bootstrap/bootstrap.sh
```

About 5-7 minutes from fresh TPU to working encoder.

## Layout

- `bootstrap/` — shell scripts that recreate the TPU's environment from scratch, numbered by phase
- `scripts/` — disposable smoke tests and one-off utilities
- `experiments/` — POC work (K-means segmentation, translator training, editability tests)

## Notes

- Weights are mirrored to `gs://vjepa-poc-eu/weights/` for fast re-download inside europe-west4
- Do not commit weights (.pt), venv, or notebook outputs — see .gitignore

