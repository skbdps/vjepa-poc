"""Reconstructed Day12 GPU-pilot configuration; not the lost preparation source."""
MODEL_ID = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"
DIFFUSERS_VERSION = "0.35.1"
SCENES = (14000, 14001)
NOISE_SEEDS = (1111, 2222)
ARMS = ("source_static", "jepa", "no_f", "wrong_direction", "oracle")
SHIFTS = {14000: (0, 16, 32, 48, 64, 48, 32, 16, 0),
          14001: (0, -16, -32, -48, -64, -48, -32, -16, 0)}
SETTINGS = dict(height=384, width=384, num_frames=9, num_inference_steps=20,
                guidance_scale=5.0, conditioning_scale=1.0)
MAX_SEQUENCE_LENGTH = 128
PROMPTS = {
    seed: "Two patterned colored balls on a textured background. One ball moves horizontally to the "
          + direction + " and then returns to its original position. The other ball and the background "
          "stay still. Static camera, simple geometric scene."
    for seed, direction in ((14000, "right"), (14001, "left"))
}
NEGATIVE_PROMPT = "text, watermark, blurry, distorted, extra objects"
EXPECTED_GUIDE_HASHES = {
    14000: "794c22894150ab7b65688b6356f16f46f7b79e22f981306494b55e6c1084d219",
    14001: "9fd176e9c0557c21c2e9c7a9a09ef995b4be5f724e2566fb0b8b77584a367cc2",
}
