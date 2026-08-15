import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from hydra import initialize, compose
from model.pl_seg import Image_Seg_Module
from torchvision import transforms
from model.infer_utils import extract_instances
from dataset.utils.img_utils import crop_and_resize, get_mask
from dataset.utils.vis_utils import _to_uint8_rgb

def visualize_res(img, mask):
    # Normalize image and mask representations.
    img_np  = _to_uint8_rgb(img)
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    elif torch.is_tensor(mask):
        mask_np = mask.detach().cpu().numpy()
    elif isinstance(mask, np.ndarray):
        mask_np = mask
    else:
        raise TypeError("mask need to be PIL/np/torch")

    if mask_np.ndim != 2:
        raise ValueError(f"mask need to be (H,W), current: {mask_np.shape}")
    H, W = mask_np.shape
    if img_np.shape[:2] != (H, W):
        raise ValueError(f"img and mask size mismatch: img={img_np.shape[:2]}, mask={(H,W)}")

    # Create colored mask overlay with transparency
    overlay = img_np.copy()

    # Create mask for foreground objects (detected objects)
    foreground_mask = mask_np >= 0

    # Apply semi-transparent green color to detected objects
    overlay[foreground_mask] = overlay[foreground_mask] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5

    # Draw bounding boxes for each detected object
    unique_ids = np.unique(mask_np[mask_np >= 0])
    for obj_id in unique_ids:
        obj_mask = (mask_np == obj_id)
        if obj_mask.sum() == 0:
            continue

        # Get bounding box coordinates
        rows = np.any(obj_mask, axis=1)
        cols = np.any(obj_mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        # Draw rectangle on overlay (red box with thickness 10)
        cv2.rectangle(overlay, (xmin, ymin), (xmax, ymax), (255, 0, 0), 10)

    return overlay

if __name__ == "__main__":
    output_base = "output/demo"
    os.makedirs(output_base, exist_ok=True)
    data_base = "/home/tum/Documents/demo"
    test_scene_dir = os.path.join(data_base, "test_scene")
    obj_condtions_dir = os.path.join(data_base, "obj_condition")

    obj_name = "bean_can"
    scene_name = "scene_6"
    query_image_path = os.path.join(test_scene_dir, f"{scene_name}.jpg")

    Transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    # Load config
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="infer_config")
    patch_size = config.patch_size
    condition_size = config.condition_size
    img_size = config.img_size
    ckpt_path = config.ckpt_path

    # Load model
    model = Image_Seg_Module(config, mode='eval')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.cuda().eval()

    # Load data
    img_original = cv2.imread(query_image_path)[:,:,::-1]  # Keep original resolution
    original_size = img_original.shape[:2]  # (H, W)

    # Resize for model input
    img = cv2.resize(img_original, tuple(img_size[::-1]), interpolation=cv2.INTER_LINEAR)
    img = Transformer(img)

    condition_image_path = os.path.join(obj_condtions_dir, f"{obj_name}.png")
    cond_img = cv2.imread(condition_image_path)[:,:,::-1]
    cond_mask = get_mask(cond_img)
    # resize and crop the condition image
    cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=condition_size, crop_rel_pad=0)

    cond_img = Transformer(cond_img)
    cond_mask = torch.from_numpy(cond_mask)

    img = img.unsqueeze(0).cuda()               # [1, 3, H, W]
    cond_imgs = cond_img.unsqueeze(0).unsqueeze(0).cuda()   # [1, N, 3, Hc, Wc]
    cond_masks = cond_mask.unsqueeze(0).unsqueeze(0).cuda() # [1, N, Hc, Wc]

    # Inference
    with torch.no_grad():
        out_masks = model(img, cond_imgs, cond_masks)
        out_masks = out_masks[-1]
    out_masks = F.interpolate(out_masks, tuple(img_size), mode="bilinear", align_corners=False)
    # Further processing can be done here (e.g., post-processing to get final masks)
    instances_by_image, sem_mask = extract_instances(
        out_masks, cls_thresh=config.cls_thresh
    )

    # Visualize at original resolution
    predict = sem_mask.squeeze(0).squeeze(0).cpu().numpy()
    # Resize mask back to original resolution
    predict_original = cv2.resize(predict, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
    vis_pred = visualize_res(img_original, predict_original)
    save_path = os.path.join(output_base, f"{scene_name}_{obj_name}.jpg")
    # Use JPEG with quality 95 for smaller file size (RGB -> BGR conversion needed)
    cv2.imwrite(save_path, vis_pred[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
