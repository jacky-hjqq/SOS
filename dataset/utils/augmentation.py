# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Dict, Iterable, Optional

import cv2
import numpy as np
from torchvision import transforms

def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None

def random_rotation(img: np.ndarray,
                    mask: np.ndarray,
                    angle_deg: float,
                    expand: bool = True,
                    bg_value=(255, 255, 255)):
    H, W = mask.shape[:2]
    dtype_img = img.dtype
    dtype_mask = mask.dtype

    # Locate the foreground bounding box.
    bin_fg = (mask != 0)
    ys, xs = np.where(bin_fg)
    if len(xs) == 0:
        return img.copy(), mask.copy()

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1

    img_patch  = img[y0:y1, x0:x1].copy()
    mask_patch = mask[y0:y1, x0:x1].copy()

    # Rotate around the patch center.
    h, w = img_patch.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)

    # Expand the output canvas when requested.
    if expand:
        cos = abs(M[0, 0]); sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w/2 - w/2)
        M[1, 2] += (new_h/2 - h/2)
    else:
        new_w, new_h = w, h

    # Select the border fill value.
    if bg_value is None:
        border_val_img = (255, 255, 255) if img.ndim == 3 else 0
    else:
        border_val_img = tuple(bg_value) if img.ndim == 3 else (bg_value if np.isscalar(bg_value) else 0)

    # Use bilinear interpolation for the image and nearest-neighbor for labels.
    img_rot  = cv2.warpAffine(img_patch,  M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=border_val_img)
    mask_rot = cv2.warpAffine(mask_patch, M, (new_w, new_h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    # Place the rotated patch at the original box center.
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    px = int(round(cx - new_w / 2))
    py = int(round(cy - new_h / 2))

    # Clip the patch to the image bounds.
    xA = max(0, px); yA = max(0, py)
    xB = min(W, px + new_w); yB = min(H, py + new_h)
    if xA >= xB or yA >= yB:
        return img.copy(), mask.copy()

    # Map the overlap to rotated-patch coordinates.
    rxA = xA - px; ryA = yA - py
    rxB = rxA + (xB - xA); ryB = ryA + (yB - yA)

    out_img  = img.copy()
    out_mask = mask.copy()

    # Clear the original foreground before placing the rotated patch.
    old_fg = (mask[y0:y1, x0:x1] != 0)
    if img.ndim == 3:
        bg = np.array(border_val_img, dtype=out_img.dtype)
        out_img[y0:y1, x0:x1][old_fg] = bg
    else:
        bg = 0 if bg_value is None else (int(bg_value) if np.isscalar(bg_value) else 0)
        out_img[y0:y1, x0:x1][old_fg] = bg
    # Clear the original labels as well.
    out_mask[y0:y1, x0:x1] = 0

    # Copy only foreground pixels from the rotated patch.
    alpha = (mask_rot[ryA:ryB, rxA:rxB] != 0)

    if img.ndim == 3:
        patch  = img_rot[ryA:ryB, rxA:rxB]
        region = out_img[yA:yB, xA:xB]
        region[alpha] = patch[alpha]
        out_img[yA:yB, xA:xB] = region
    else:
        patch  = img_rot[ryA:ryB, rxA:rxB]
        region = out_img[yA:yB, xA:xB]
        region[alpha] = patch[alpha]
        out_img[yA:yB, xA:xB] = region

    # Restore the rotated class labels.
    region_m = out_mask[yA:yB, xA:xB]
    patch_m  = mask_rot[ryA:ryB, rxA:rxB]
    region_m[alpha] = patch_m[alpha]
    out_mask[yA:yB, xA:xB] = region_m

    return out_img.astype(dtype_img, copy=False), out_mask.astype(dtype_mask, copy=False)

def random_scaled_crop(
    image: np.ndarray,           # HxW or HxWxC
    mask: np.ndarray,            # HxW integer class IDs
    crop_scale: float,
    focus_on_objects: bool = True,
    focus_strength: float = 2.0,      # Values above 1 favor dense foreground regions.
    background_ids: Iterable[int] | None = (0, -1),
):
    """Crop at a fixed scale and resize to the original dimensions.

    When ``focus_on_objects`` is true, sample windows in proportion to their
    foreground pixel count raised to ``focus_strength``.
    """
    assert image.shape[:2] == mask.shape[:2]
    H, W = mask.shape[:2]

    crop_h = max(1, int(round(H * crop_scale)))
    crop_w = max(1, int(round(W * crop_scale)))

    # Identify foreground pixels.
    if background_ids is None:
        fg = (mask != 0)
    else:
        fg = ~np.isin(mask, list(background_ids))

    # Avoid resampling when the crop covers the full image.
    if crop_h == H and crop_w == W:
        return image.copy(), mask.copy()

    # Select the crop's top-left corner.
    if not focus_on_objects:
        y0 = 0 if H == crop_h else random.randint(0, H - crop_h)
        x0 = 0 if W == crop_w else random.randint(0, W - crop_w)
    else:
        # Compute foreground counts for all windows with an integral image.
        integ = cv2.integral(fg.astype(np.uint8))  # (H+1, W+1)
        hh, ww = H - crop_h + 1, W - crop_w + 1
        if hh <= 0 or ww <= 0:
            # Fall back safely if the requested crop is larger than the image.
            y0 = 0
            x0 = 0
        else:
            sums = (integ[crop_h:, crop_w:]
                    - integ[:-crop_h, crop_w:]
                    - integ[crop_h:, :-crop_w]
                    + integ[:-crop_h, :-crop_w]).astype(np.float64)  # [hh, ww]
            if sums.max() <= 0:
                # Sample uniformly when no foreground is present.
                y0 = 0 if H == crop_h else random.randint(0, H - crop_h)
                x0 = 0 if W == crop_w else random.randint(0, W - crop_w)
            else:
                weights = np.power(sums + 1e-6, focus_strength)
                probs = (weights / weights.sum()).ravel()
                idx = np.random.choice(probs.size, p=probs)
                y0, x0 = divmod(idx, ww)
                y0, x0 = int(y0), int(x0)

    img_c = image[y0:y0+crop_h, x0:x0+crop_w]
    msk_c = mask[y0:y0+crop_h, x0:x0+crop_w]

    # Resize the image linearly and labels with nearest-neighbor interpolation.
    img_out = cv2.resize(img_c, (W, H), interpolation=cv2.INTER_LINEAR)
    msk_out = cv2.resize(msk_c, (W, H), interpolation=cv2.INTER_NEAREST)

    # Preserve channel count and dtypes.
    if image.ndim == 3 and img_out.ndim == 2:
        img_out = img_out[..., None]
    img_out = img_out.astype(image.dtype, copy=False)
    msk_out = msk_out.astype(mask.dtype, copy=False)

    return img_out, msk_out
