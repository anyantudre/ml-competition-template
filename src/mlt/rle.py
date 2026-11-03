"""Run-length encoding of binary masks, the usual segmentation submission format.

order="F" numbers pixels top to bottom, then left to right (column-major), as in most Kaggle
segmentation competitions. order="C" numbers them row by row. Pixel numbers start at 1.
"""

from __future__ import annotations

import numpy as np


def rle_encode(mask: np.ndarray, order: str = "F") -> str:
    pixels = np.asarray(mask, dtype=np.uint8).flatten(order=order)
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.flatnonzero(pixels[1:] != pixels[:-1]) + 1
    runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


def rle_pixels(rle) -> set[int]:
    """The set of foreground pixel numbers in an RLE string (empty or NaN gives an empty set)."""
    if not isinstance(rle, str) or not rle.strip():
        return set()
    nums = np.asarray(rle.split(), dtype=int)
    return {p for s, n in zip(nums[0::2], nums[1::2]) for p in range(s, s + n)}


def rle_decode(rle, shape: tuple[int, int], order: str = "F") -> np.ndarray:
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if isinstance(rle, str) and rle.strip():
        nums = np.asarray(rle.split(), dtype=int)
        starts, lengths = nums[0::2] - 1, nums[1::2]
        for s, n in zip(starts, lengths):
            mask[s:s + n] = 1
    return mask.reshape(shape, order=order)
