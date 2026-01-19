from typing import Literal
import numpy as np


def generate_scout(
    volume: np.ndarray,
    scout_type: Literal["frontal", "lateral"] = "frontal",
    wl: int = 40,
    ww: int = 300,
    gamma: float = 0.9
) -> np.ndarray:

    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D (Z, Y, X)")

    # 1️⃣ HU windowing
    min_val = wl - ww / 2
    max_val = wl + ww / 2

    vol = volume.astype(np.float32, copy=False)
    np.clip(vol, min_val, max_val, out=vol)
    vol = (vol - min_val) / (max_val - min_val)

    # 2️⃣ REMOVE AIR (🔥 THIS FIXES WHITE BACKGROUND 🔥)
    vol[volume < -700] = 0.0

    # 3️⃣ X-ray attenuation
    vol = np.exp(-2.8 * vol)

    # 4️⃣ Projection
    if scout_type == "frontal":
        scout = np.sum(vol, axis=1, dtype=np.float32)
    elif scout_type == "lateral":
        scout = np.sum(vol, axis=2, dtype=np.float32)
    else:
        raise ValueError("Invalid scout_type")

    # 5️⃣ Log compression
    scout = np.log1p(scout)

    # 6️⃣ Normalize
    scout -= scout.min()
    scout /= scout.max() + 1e-6

    # 7️⃣ Orientation fix: flip vertically so head is at the top of the image
    # (volume is stored as Z,Y,X; after projection we flip rows so cranial side is up)
    scout = np.flipud(scout)

    # 8️⃣ Gamma correction
    scout = np.power(scout, gamma)

    return (scout * 255).astype(np.uint8)
