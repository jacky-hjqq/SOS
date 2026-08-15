from contextlib import contextmanager
import os
import tempfile

import numpy as np
import torch
from scipy import ndimage


@contextmanager
def condition_feature_cache(model, batch_data, device="cuda"):
    """Cache shared condition masks, object IDs, and DINO features for one run."""
    with tempfile.TemporaryDirectory(prefix="sos_condition_cache_") as cache_dir:
        cache_path = os.path.join(cache_dir, "conditions.pt")
        cond_imgs = batch_data["cond_imgs"].unsqueeze(0).to(device)

        with torch.no_grad():
            cond_features = model.extract_condition_features(cond_imgs)

        torch.save(
            {
                "cond_masks": batch_data["cond_masks"].detach().cpu(),
                "cond_obj_ids": batch_data["cond_obj_ids"].detach().cpu(),
                "cond_features": cond_features.detach().cpu(),
            },
            cache_path,
        )
        del cond_imgs, cond_features

        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        cached["cond_masks"] = cached["cond_masks"].unsqueeze(0).to(device)
        cached["cond_features"] = cached["cond_features"].to(device)
        try:
            yield cached
        finally:
            del cached

def remap_target(target, cond_mask_ids, ignore_idx=-1):
    """
    Map arbitrary target labels to channel indices [0..N-1] for CE,
    while preserving ignore_idx (e.g., -1) exactly.

    target: (B,H,W) or (H,W), ints. Background is `ignore_idx`.
    cond_mask_ids: (N,) or (B,N), ints. cond_mask_ids[:, j] == label id in target for channel j.
    """
    batched = (target.dim() == 3)
    if not batched:
        target = target.unsqueeze(0)  # (1,H,W)

    B, H, W = target.shape
    if cond_mask_ids.dim() == 1:
        cond_mask_ids = cond_mask_ids.unsqueeze(0).expand(B, -1)
    else:
        assert cond_mask_ids.shape[0] == B

    N = cond_mask_ids.shape[1]
    tgt = target.unsqueeze(1)                       # (B,1,H,W)
    ids = cond_mask_ids[:, :, None, None]           # (B,N,1,1)

    # Match only valid condition channels.
    valid_ch = (cond_mask_ids != ignore_idx)[:, :, None, None]  # (B,N,1,1)
    eq = (tgt == ids) & valid_ch                                # (B,N,H,W)

    has_match = eq.any(dim=1)                    # (B,H,W)
    ch_idx = eq.float().argmax(dim=1)            # (B,H,W)

    remapped = torch.full_like(target, ignore_idx)
    remapped[has_match] = ch_idx[has_match]

    return remapped if batched else remapped[0]

def extract_instances(
    out_mask: torch.Tensor,    # [B,C,H,W] class logits
    cls_thresh: float = 0.5,
    min_area: int = 32,
    nms_mode: str = "mask",    # "mask" or "bbox"
    return_masks: bool = True,
    background_id: int = -1,
):
    """Convert each predicted class channel directly into one instance."""
    assert out_mask.ndim == 4
    B, C, H, W = out_mask.shape
    device = out_mask.device

    # Compute class probabilities and masks on the input device.
    probs = torch.sigmoid(out_mask)
    binary_masks = probs > cls_thresh  # [B, C, H, W] bool

    results_all = []

    # Add a threshold-valued background channel before taking argmax.
    bg_prob = torch.full((B, 1, H, W), cls_thresh, device=device)
    combined_probs = torch.cat([bg_prob, probs], dim=1) # [B, C+1, H, W]

    # Index 0 denotes background; 1..C map to original channels 0..C-1.
    global_semantic_indices = torch.argmax(combined_probs, dim=1, keepdim=True)
    global_semantic_mask = global_semantic_indices - 1

    # Remap the default background label when requested.
    if background_id != -1:
        global_semantic_mask[global_semantic_mask == -1] = background_id

    # Extract one instance per non-empty class channel.
    for b in range(B):
        cands = []
        img_binary = binary_masks[b] # [C, H, W]
        img_probs = probs[b]         # [C, H, W]

        for c in range(C):
            m = img_binary[c]
            area = m.sum().item()

            if area < min_area:
                continue

            # Average the class probability over selected pixels.
            score = img_probs[c][m].mean().item()

            # Extract a half-open box [x1, y1, x2, y2].
            coords = torch.where(m)
            y1, x1 = coords[0].min().item(), coords[1].min().item()
            y2, x2 = coords[0].max().item() + 1, coords[1].max().item() + 1

            cands.append({
                "class_id": c,
                "bbox": [x1, y1, x2, y2],
                "area": int(area),
                "score": float(score),
                "mask": m.cpu().numpy() if (nms_mode == "mask" or return_masks) else None
            })

        kept = cands
        img_instances = []
        for it in kept:
            inst = {k: v for k, v in it.items() if k != "mask"}
            if return_masks:
                inst["mask"] = it["mask"]
            img_instances.append(inst)

        results_all.append(img_instances)

    return results_all, global_semantic_mask


def mask_to_bbox(mask):
    """Return the inclusive ``[x1, y1, x2, y2]`` box of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    y1, y2 = np.flatnonzero(rows)[[0, -1]]
    x1, x2 = np.flatnonzero(cols)[[0, -1]]
    return [int(x1), int(y1), int(x2), int(y2)]


def split_instances(binary_mask, min_area=500):
    """Split a binary mask into connected components above ``min_area``."""
    labels, count = ndimage.label(binary_mask, structure=np.ones((3, 3)))
    if count == 0:
        return []

    areas = ndimage.sum(np.ones_like(labels), labels, range(1, count + 1))
    return [
        labels == label
        for label, area in enumerate(areas, start=1)
        if area >= min_area
    ]

def mask_to_rle(binary_mask: np.ndarray):
    """
    Convert binary mask to uncompressed RLE.
    Much faster than Python loop.

    Args:
        binary_mask: numpy array [H, W] with {0,1}
    Returns:
        RLE: {"size": [H, W], "counts": [list of run lengths]}
    """
    h, w = binary_mask.shape
    # flatten in Fortran order
    flat = binary_mask.ravel(order="F").astype(np.uint8)

    # pad with 0 at start and end to catch transitions
    padded = np.pad(flat, (1, 1), mode="constant", constant_values=0)

    # find where values change
    changes = np.where(padded[1:] != padded[:-1])[0]

    # run lengths are diffs between change indices
    counts = np.diff(np.concatenate(([0], changes, [len(flat)]))).tolist()

    return {"size": [h, w], "counts": counts}
