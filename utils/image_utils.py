import base64
import io
import numpy as np
from PIL import Image


def array_to_base64_png(image: np.ndarray) -> str:
    """
    Convert a 2D uint8 numpy array to base64 PNG string.
    """

    if image.ndim != 2:
        raise ValueError("Input image must be 2D")

    if image.dtype != np.uint8:
        raise ValueError("Image dtype must be uint8")

    pil_img = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
