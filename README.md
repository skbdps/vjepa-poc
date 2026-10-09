# V-JEPA 2.1 POC on Google TRC TPU

Controllable AI video generation POC using V-JEPA 2.1 latent-space manipulation.

## Current consistency experiments

The ongoing work on `experiment/part-consistency-colab` tests persistent object-part identity and selective video edits on a Colab T4.

- [Day 3: initial part-localization diagnostic](experiments/day3/README.md)
- [Day 4: stronger benchmarks, real-car part masks, and selective editing](experiments/day4/README.md)
- [Offline interactive part editor](experiments/day4/run_2026-10-09/car_roundabout/interactive_part_editor.html) — download and open the HTML; control door/window colors independently, save a recipe, and replay it without another model run.
- [Real-car two-part edit preview](experiments/day4/run_2026-10-09/car_roundabout/dual_part_edit.mp4)
- [Reproduction notebook for Colab T4](experiments/day4/Part_Consistency_Colab.ipynb)
- [Results and limitations](experiments/day4/RESULTS.md)
- [Completed held-out specialist comparison](experiments/day4/run_2026-10-09/comparison/comparison.md)
- [Crossing and identity failure gallery](experiments/day4/run_2026-10-09/failures/README.md)

On eighteen held-out synthetic clips, SAM2.1 Tiny reaches 95.01% visible patch localization versus 53.38% for the selected V-JEPA tracker, a paired difference of +41.62 percentage points [95% interval +30.94, +50.10]. Its separate raw dense-mask mean IoU is 94.42%; raw masks remain the editing default because the supplementary presence gate suppresses thin visible parts. Identity recovery after occlusion remains unresolved, and the two real-car clips demonstrate localized recoloring rather than unseen-video or generative consistency.

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

