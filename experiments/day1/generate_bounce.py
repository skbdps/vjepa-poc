"""Generate a 4-second video of a red ball bouncing on a white background.

Parameters chosen for V-JEPA 2.1 inference:
- 384×384 resolution (matches encoder input)
- 30 fps native (smooth for visual inspection)
- 120 frames total (4 sec × 30 fps)
- One bounce with realistic physics

Run: python experiments/day1/generate_bounce.py
Output: experiments/day1/outputs/bounce.mp4
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

# ---------------- parameters ----------------
WIDTH = HEIGHT = 384
FPS = 30
DURATION_SEC = 4.0
NUM_FRAMES = int(FPS * DURATION_SEC)  # 120

# Ball
BALL_RADIUS = 30
BALL_COLOR_BGR = (30, 30, 220)  # opencv is BGR — this is red
BG_COLOR_BGR = (255, 255, 255)  # white

# Physics (units: pixels, seconds)
GRAVITY = 400.0              # px/s² downward
INITIAL_X = 80.0
INITIAL_Y = 80.0             # top area
INITIAL_VX = 70.0            # px/s rightward
INITIAL_VY = 0.0             # no initial vertical velocity
RESTITUTION = 0.75           # velocity retained after bounce
GROUND_Y = HEIGHT - BALL_RADIUS  # center of ball when touching floor = 354
LEFT_WALL = BALL_RADIUS
RIGHT_WALL = WIDTH - BALL_RADIUS

OUTPUT_PATH = Path(__file__).parent / "outputs" / "bounce.mp4"


def simulate() -> list[tuple[float, float]]:
    """Run a simple fixed-step physics sim and return (x, y) per frame."""
    dt = 1.0 / FPS
    x, y = INITIAL_X, INITIAL_Y
    vx, vy = INITIAL_VX, INITIAL_VY
    positions = []
    for _ in range(NUM_FRAMES):
        positions.append((x, y))
        # Integrate
        vy += GRAVITY * dt
        x += vx * dt
        y += vy * dt
        # Floor bounce
        if y >= GROUND_Y:
            y = GROUND_Y
            vy = -vy * RESTITUTION
            # Small damping on horizontal so it visibly settles over time
            vx *= 0.95
        # Walls (defensive — ball shouldn't hit these in 4 sec, but prevent clipping)
        if x < LEFT_WALL:
            x = LEFT_WALL
            vx = -vx
        if x > RIGHT_WALL:
            x = RIGHT_WALL
            vx = -vx
    return positions


def render(positions: list[tuple[float, float]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # mp4v codec, widely decodable by decord / torchvision.io / any player
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {OUTPUT_PATH}")

    for (x, y) in positions:
        frame = np.full((HEIGHT, WIDTH, 3), BG_COLOR_BGR, dtype=np.uint8)
        cv2.circle(
            frame,
            (int(round(x)), int(round(y))),
            BALL_RADIUS,
            BALL_COLOR_BGR,
            thickness=-1,      # filled
            lineType=cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Wrote {NUM_FRAMES} frames to {OUTPUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    positions = simulate()
    print(f"First position: {positions[0]}")
    print(f"Last position: {positions[-1]}")
    print(f"Y range: {min(p[1] for p in positions):.1f} to {max(p[1] for p in positions):.1f}")
    render(positions)
