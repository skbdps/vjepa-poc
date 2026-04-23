# V-JEPA 2.1 POC on Google TRC TPU

Controllable AI video generation POC using V-JEPA 2.1 latent-space manipulation.

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
