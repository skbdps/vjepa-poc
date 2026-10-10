"""Cache two motion prompts with FP32 UMT5; save finite FP16 tensors for Wan.

Reconstructed after workspace maintenance. Run separately from render.py so
the large text encoder is unloaded before the video transformer is loaded.
"""
from __future__ import annotations
import argparse
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
import torch
from common import array_sha256, file_record, sha256, write_json
from config import (DIFFUSERS_VERSION, MAX_SEQUENCE_LENGTH, MODEL_ID,
                    NEGATIVE_PROMPT, PROMPTS, REVISION, SCENES)


def main(args):
    if version("diffusers") != DIFFUSERS_VERSION or not torch.cuda.is_available():
        raise RuntimeError("Pinned diffusers and a CUDA runtime are required")
    if version("transformers") != "4.57.6":
        raise RuntimeError("Require the reviewed Transformers 4.57.6 text encoder")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metadata.json").exists() or any(out.glob("prompt_*.pt")):
        raise ValueError("Use a fresh output directory; existing prompt caches are immutable")
    from diffusers import WanVACEPipeline
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, UMT5EncoderModel
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION,
                    cache_dir=args.model_cache, allow_patterns=["text_encoder/*", "tokenizer/*"]))
    model_files = {str(p.relative_to(snapshot)): dict(sha256=sha256(p), bytes=p.stat().st_size)
                   for component in ("text_encoder", "tokenizer")
                   for p in sorted((snapshot / component).rglob("*")) if p.is_file()}
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), subfolder="tokenizer")
    encoder = UMT5EncoderModel.from_pretrained(
        str(snapshot), subfolder="text_encoder", torch_dtype=torch.float32,
        device_map="auto", max_memory={0: "10GiB", "cpu": "2GiB"},
        offload_folder=str(out / "text_encoder_offload"), offload_state_dict=True,
        low_cpu_mem_usage=True).eval().requires_grad_(False)
    # Reuse the pinned pipeline's text cleaning, tokenization and padding.
    holder = SimpleNamespace(text_encoder=encoder, tokenizer=tokenizer)
    metadata = dict(version="day12_prompt_cache_reconstructed_v1", complete=False,
                    model_id=MODEL_ID, model_revision=REVISION, model_files=model_files,
                    encoder_dtype="float32", saved_dtype="float16", max_sequence_length=MAX_SEQUENCE_LENGTH,
                    encoding_method="WanVACEPipeline._get_t5_prompt_embeds diffusers==0.35.1",
                    negative_prompt=NEGATIVE_PROMPT, prompts={str(k): v for k, v in PROMPTS.items()},
                    device_map={k: str(v) for k, v in encoder.hf_device_map.items()},
                    runtime=dict(torch=torch.__version__, diffusers=version("diffusers"),
                                 transformers=version("transformers"), gpu=torch.cuda.get_device_name()),
                    source_files={name: sha256(Path(__file__).parent / name)
                                  for name in ("cache_prompts.py", "common.py", "config.py")},
                    caches={})
    write_json(out / "metadata.json", metadata)

    def encode(text):
        with torch.inference_mode():
            fp32 = WanVACEPipeline._get_t5_prompt_embeds(
                holder, text, max_sequence_length=MAX_SEQUENCE_LENGTH,
                device=torch.device("cuda:0"), dtype=torch.float32)
        if fp32.shape != (1, MAX_SEQUENCE_LENGTH, 4096) or not torch.isfinite(fp32).all():
            raise FloatingPointError("UMT5 produced a malformed/nonfinite FP32 embedding")
        value = fp32.to(dtype=torch.float16, device="cpu")
        if not torch.isfinite(value).all():
            raise FloatingPointError("Embedding overflowed when stored as FP16")
        return value

    negative = encode(NEGATIVE_PROMPT)
    for seed in SCENES:
        value = dict(prompt_embeds=encode(PROMPTS[seed]), negative_prompt_embeds=negative)
        path = out / f"prompt_{seed}.pt"
        torch.save(value, path)
        metadata["caches"][str(seed)] = dict(file=file_record(path, out),
            arrays={key: array_sha256(tensor.numpy()) for key, tensor in value.items()})
        write_json(out / "metadata.json", metadata)
    metadata["complete"] = True
    write_json(out / "metadata.json", metadata)
    print("COMPLETE", out / "metadata.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-cache")
    main(parser.parse_args())
