import cv2
import numpy as np
from PIL import Image
from torch.nn import functional as F


def load_benchmark_query(dataset, index):
    """Load only the query image and metadata for benchmark inference."""
    sample = dataset.total_data[index]
    image_path = sample["image_path"]

    with Image.open(image_path) as image:
        image = np.array(image.convert("RGB"))
    image = dataset.crop_img(image, patch_size=dataset.patch_size)
    if dataset.transform is not None:
        image = dataset.transform(image)

    return {
        "scene_id": sample["scene_id"],
        "image_id": sample["image_id"],
        "img_path": image_path,
        "img": image,
    }


def _crop_window(image_shape, bbox, margin_ratio, target_size):
    """Calculate an aspect-ratio-preserving crop window around a box."""
    image_h, image_w = image_shape
    x1, y1, x2, y2 = bbox
    target_h, target_w = target_size

    box_w = x2 - x1
    box_h = y2 - y1
    center_x = x1 + box_w / 2
    center_y = y1 + box_h / 2
    crop_w = box_w * (1 + 2 * margin_ratio)
    crop_h = box_h * (1 + 2 * margin_ratio)

    target_aspect = target_w / target_h
    if crop_w / crop_h > target_aspect:
        crop_h = crop_w / target_aspect
    else:
        crop_w = crop_h * target_aspect

    crop_x1 = center_x - crop_w / 2
    crop_y1 = center_y - crop_h / 2
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > image_w:
        crop_x1 -= crop_x2 - image_w
        crop_x2 = image_w
    if crop_y2 > image_h:
        crop_y1 -= crop_y2 - image_h
        crop_y2 = image_h

    return (
        max(0, int(crop_x1)),
        max(0, int(crop_y1)),
        min(image_w, int(crop_x2)),
        min(image_h, int(crop_y2)),
    )


def crop_around_bbox(image, bbox, margin_ratio=0.5, target_size=(224, 224)):
    """Crop a batched tensor around a box and resize it to ``target_size``."""
    crop_x1, crop_y1, crop_x2, crop_y2 = _crop_window(
        image.shape[-2:], bbox, margin_ratio, target_size
    )
    crop = image[:, :, crop_y1:crop_y2, crop_x1:crop_x2]
    return F.interpolate(
        crop,
        size=target_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )


def restore_mask_to_image(
    cropped_mask,
    image_shape,
    bbox,
    margin_ratio=0.5,
    target_size=(224, 224),
):
    """Restore a cropped binary mask to its original image coordinates."""
    image_h, image_w = image_shape
    crop_x1, crop_y1, crop_x2, crop_y2 = _crop_window(
        image_shape, bbox, margin_ratio, target_size
    )
    resized_mask = cv2.resize(
        cropped_mask.astype(np.uint8),
        (crop_x2 - crop_x1, crop_y2 - crop_y1),
        interpolation=cv2.INTER_NEAREST,
    )

    restored = np.zeros((image_h, image_w), dtype=bool)
    restored[crop_y1:crop_y2, crop_x1:crop_x2] = resized_mask > 0
    return restored

def get_mask(rgb_array, background="auto", tol=5):
    """
    Generate a foreground mask from an RGB numpy image with a black or white background.
    :param rgb_array: Input image (H, W, 3, RGB)
    :param background: "auto" | "black" | "white"
    :param tol: Tolerance for near-black/near-white values (to handle compression artifacts)
    :return: mask (H, W), foreground=255, background=0
    """
    if background == "auto":
        mean_val = np.mean(rgb_array)
        background = "white" if mean_val > 127 else "black"

    if background == "white":
        # Background is close to white → foreground = non-white pixels
        mask = np.any(rgb_array < 255 - tol, axis=-1).astype(np.uint8) * 255
    else:
        # Background is close to black → foreground = non-black pixels
        mask = np.any(rgb_array > tol, axis=-1).astype(np.uint8) * 255

    return mask

def crop_and_resize(image, mask, size=224, crop_rel_pad=0.2, pad_value=(128,128,128)):
    H, W = mask.shape
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        out_img = np.full((size, size, image.shape[2]), pad_value, dtype=image.dtype)
        out_mask = np.zeros((size, size), dtype=np.uint8)
        return out_img, out_mask

    # Compute and pad the foreground bounding box.
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    pad_x = int((x_max - x_min + 1) * crop_rel_pad)
    pad_y = int((y_max - y_min + 1) * crop_rel_pad)
    x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
    x2, y2 = min(W, x_max + pad_x), min(H, y_max + pad_y)

    cropped_img = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]

    # Replace background pixels inside the crop with the padding value.
    cropped_img[cropped_mask == 0] = pad_value

    # Scale the crop without changing its aspect ratio.
    h, w = cropped_img.shape[:2]
    scale = min(size / h, size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized_img = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center the result on a square canvas.
    out_img = np.full((size, size, image.shape[2]), pad_value, dtype=resized_img.dtype)
    out_mask = np.zeros((size, size), dtype=np.uint8)

    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2

    out_img[y0:y0+new_h, x0:x0+new_w] = resized_img
    out_mask[y0:y0+new_h, x0:x0+new_w] = resized_mask

    return out_img, out_mask
