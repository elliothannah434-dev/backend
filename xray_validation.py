import numpy as np
import cv2


def is_grayscale_xray(image_bgr: np.ndarray, threshold: float = 15.0):
    """Heuristic: reject likely color photos by measuring channel differences.

    Args:
        image_bgr: OpenCV BGR image (as returned by cv2.imdecode / cv2.imread)
        threshold: higher -> more permissive (accept more images as grayscale)

    Returns:
        (is_valid, message)
    """
    if image_bgr is None:
        return False, "Could not decode image"

    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        return True, "Passed: Single-channel / grayscale-like input"

    # Split channels (B,G,R) then compute mean absolute differences.
    b = image_bgr[:, :, 0].astype(np.float32)
    g = image_bgr[:, :, 1].astype(np.float32)
    r = image_bgr[:, :, 2].astype(np.float32)

    rg_diff = float(np.mean(np.abs(r - g)))
    rb_diff = float(np.mean(np.abs(r - b)))

    max_diff = max(rg_diff, rb_diff)
    if max_diff > threshold:
        return (
            False,
            f"Uploaded image is not an X-Ray Image.",
        )

    return True, "Passed: Image looks grayscale"


def decode_imagefile_to_bgr(image_bytes: bytes):
    """Decode uploaded bytes into an OpenCV BGR image."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

