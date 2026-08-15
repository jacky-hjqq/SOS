import json
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize
from torch.nn import functional as F
from tqdm import tqdm

from dataset.hb_dataset import HB_Segmentation
from dataset.lmo_dataset import LMO_Segmentation
from dataset.tudl_dataset import TUDL_Segmentation
from dataset.utils.img_utils import (
    crop_around_bbox,
    load_benchmark_query,
    restore_mask_to_image,
)
from dataset.utils.vis_utils import inv_transform, visualize_image_mask_and_templates
from dataset.ycbv_dataset import YCBV_Segmentation
from model.infer_utils import (
    condition_feature_cache,
    extract_instances,
    mask_to_bbox,
    mask_to_rle,
    split_instances,
)


DATASET_TYPES = {
    "ycbv": YCBV_Segmentation,
    "lmo": LMO_Segmentation,
    "hb": HB_Segmentation,
    "tudl": TUDL_Segmentation,
}


def load_model(config):
    """Load the segmentation model and checkpoint on the GPU."""
    from model.pl_seg import Image_Seg_Module

    model = Image_Seg_Module(config, mode="eval")
    checkpoint = torch.load(config.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.cuda().eval()


def load_dataset(config):
    """Create the configured benchmark dataset."""
    try:
        dataset_type = DATASET_TYPES[config.dataset_name]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {config.dataset_name}") from error
    return dataset_type(config)


def refine_instances(model, image, coarse_instances, conditions, config):
    """Refine connected coarse predictions using cached condition features."""
    refinement = config.refinement
    candidates = []
    crops = []

    for instance in coarse_instances:
        class_id = instance["class_id"]
        for component in split_instances(
            instance["mask"], refinement.min_instance_area
        ):
            bbox = mask_to_bbox(component)
            if bbox is None:
                continue

            crops.append(
                crop_around_bbox(
                    image,
                    bbox,
                    margin_ratio=refinement.crop_margin,
                    target_size=tuple(config.input_dims),
                )
            )
            candidates.append({
                "class_id": class_id,
                "obj_id": conditions["cond_obj_ids"][class_id].item(),
                "bbox": bbox,
                "score": instance["score"],
            })

    if not candidates:
        return []

    crop_batch = torch.cat(crops, dim=0)
    class_ids = [candidate["class_id"] for candidate in candidates]
    condition_features = conditions["cond_features"][0, class_ids].unsqueeze(1)
    condition_masks = conditions["cond_masks"][0, class_ids].unsqueeze(1)

    with torch.no_grad():
        logits = model(
            crop_batch,
            cond_x=None,
            cond_x_mask=condition_masks,
            cond_features=condition_features,
        )[-1]
    logits = F.interpolate(
        logits,
        tuple(config.img_size),
        mode="bilinear",
        align_corners=False,
    )

    best_by_object = {}
    for candidate, crop_logits in zip(candidates, logits):
        crop_mask = (
            torch.sigmoid(crop_logits).squeeze(0).cpu().numpy()
            > config.cls_thresh
        )
        mask = restore_mask_to_image(
            crop_mask,
            image_shape=tuple(config.img_size),
            bbox=candidate["bbox"],
            margin_ratio=refinement.crop_margin,
            target_size=tuple(config.img_size),
        )
        area = int(mask.sum())
        if area < refinement.min_instance_area:
            continue

        obj_id = candidate["obj_id"]
        previous = best_by_object.get(obj_id)
        if previous is None or area > previous["area"]:
            best_by_object[obj_id] = {
                "obj_id": obj_id,
                "mask": mask,
                "bbox": mask_to_bbox(mask),
                "score": candidate["score"],
                "area": area,
            }

    return list(best_by_object.values())


def save_visualization(
    image,
    semantic_mask,
    condition_images,
    scene_id,
    image_id,
    config,
):
    """Save the optional benchmark visualization."""
    vis_image = inv_transform(image[0])
    vis_image = cv2.resize(
        vis_image,
        tuple(config.img_size[::-1]),
        interpolation=cv2.INTER_LINEAR,
    )
    visualization = visualize_image_mask_and_templates(
        vis_image,
        semantic_mask.squeeze(0).squeeze(0),
        condition_images,
        list(range(len(condition_images))),
    )
    save_dir = os.path.join(config.output_dir, f"{config.dataset_name}_infer_vis")
    os.makedirs(save_dir, exist_ok=True)
    plt.imsave(
        os.path.join(save_dir, f"{scene_id}_{image_id}.png"),
        visualization.astype(np.uint8),
    )


def main():
    with initialize(version_base=None, config_path="config"):
        config = compose(config_name="infer_config")

    os.makedirs(config.output_dir, exist_ok=True)
    model = load_model(config)
    dataset = load_dataset(config)
    condition_batch = dataset[0]
    first_query = {
        key: condition_batch[key]
        for key in ("scene_id", "image_id", "img_path", "img")
    }
    condition_images = None
    if config.vis_mask_enable:
        condition_images = [
            inv_transform(image) for image in condition_batch["cond_imgs"]
        ]
    results = []

    with condition_feature_cache(model, condition_batch) as conditions:
        for index in tqdm(range(len(dataset))):
            batch = first_query if index == 0 else load_benchmark_query(dataset, index)
            image = batch["img"].unsqueeze(0).cuda()

            with torch.no_grad():
                start = time.time()
                coarse_logits = model(
                    image,
                    cond_x=None,
                    cond_x_mask=conditions["cond_masks"],
                    cond_features=conditions["cond_features"],
                )[-1]

            coarse_logits = F.interpolate(
                coarse_logits,
                tuple(config.img_size),
                mode="bilinear",
                align_corners=False,
            )
            coarse_instances, semantic_mask = extract_instances(
                coarse_logits,
                cls_thresh=config.cls_thresh,
            )
            refined_instances = refine_instances(
                model,
                image,
                coarse_instances[0],
                conditions,
                config,
            )
            inference_time = time.time() - start

            for instance in refined_instances:
                results.append({
                    "scene_id": batch["scene_id"],
                    "image_id": batch["image_id"],
                    "category_id": instance["obj_id"],
                    "bbox": instance["bbox"],
                    "segmentation": mask_to_rle(instance["mask"]),
                    "score": instance["score"],
                    "time": inference_time,
                })

            if config.vis_mask_enable:
                save_visualization(
                    image,
                    semantic_mask,
                    condition_images,
                    batch["scene_id"],
                    batch["image_id"],
                    config,
                )

    result_path = os.path.join(
        config.output_dir,
        f"{config.method_name}_{config.dataset_name}-test.json",
    )
    with open(result_path, "w") as file:
        json.dump(results, file)


if __name__ == "__main__":
    main()
