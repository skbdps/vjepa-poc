# Day12: JEPA guides through an actual video generator

The translator-generated nine-frame guides and CPU causal-video-VAE checks have been preserved. Actual Wan video denoising still needs the authorized CUDA/Colab runtime. See `STATUS.md` for the evidence and access status, and `PROTOCOL.md` for the bounded 20-call experiment.

The continuation drivers were reconstructed after uncommitted local source was removed by workspace maintenance. They do not replace the historical source hashes in saved manifests. The preserved guide arrays are the inputs to the next GPU run, so preparation does not need to be repeated.

Use a CUDA Colab runtime, ideally the previously planned T4. The notebook `pilot.ipynb` provides the following commands from the repository root:

```bash
python -m pip install -r experiments/day11/requirements.txt
python experiments/day12/archive_evidence.py unpack \
  --source experiments/day12/run_recovered/evidence/guides --out /content/day12-guides
python experiments/day12/cache_prompts.py \
  --out /content/day12-prompts --model-cache /content/day12-models
python experiments/day12/render.py --guides /content/day12-guides \
  --prompt-dir /content/day12-prompts --model-cache /content/day12-models \
  --out /content/day12-render
python experiments/day12/assess.py --run-dir /content/day12-render \
  --guides-dir /content/day12-guides --out /content/day12-assessment
```

Prompt encoding and rendering run in separate processes so the large text encoder is released before the video model loads. FP32 text-encoder inference uses Accelerate placement with 10 GiB GPU and 2 GiB CPU budgets plus disk offload. This placement is configured but has not yet been validated on the newly signed-in Colab runtime. No generated-video API key is required for these public model weights.

If rendering was interrupted, use the same render command with `--resume`. The driver rejects changed settings, code, model bytes, input hashes or runtime; use a fresh output path for a changed experiment. Do not run the prompt-cache command over an existing cache directory. Download or archive `/content/day12-render` and `/content/day12-assessment` before ending a Colab session; `/content` storage is temporary.

This stage tests whether the prepared editable image representation can guide video generation. It does not establish JEPA temporal prediction or a learned scene-physics model. The `no_f` control retains exact-token provenance and isolates the CNN's contribution. All clips and metrics must be inspected before any stronger claim.
