import os
import json
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from hydra import initialize, compose
from model.pl_seg import Image_Seg_Module
from dataset.ycbv_dataset import YCBV_Segmentation
from dataset.lmo_dataset import LMO_Segmentation
from dataset.hb_dataset import HB_Segmentation
from dataset.tudl_dataset import TUDL_Segmentation
from model.infer_utils import condition_feature_cache, extract_instances, mask_to_rle
from dataset.utils.img_utils import load_benchmark_query
from dataset.utils.vis_utils import inv_transform, visualize_image_mask_and_templates

# Load config
with initialize(version_base=None, config_path="config"):
    config = compose(config_name="infer_config")
output_dir = config.output_dir
os.makedirs(output_dir, exist_ok=True)
dataset_name = config.dataset_name
patch_size = config.patch_size
condition_size = config.condition_size
img_size = config.img_size
ckpt_path = config.ckpt_path
vis_mask_enable = config.vis_mask_enable

# Load model
model = Image_Seg_Module(config, mode='eval')
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=True)
model = model.cuda().eval()

# Initialize the dataset
if config.dataset_name == "ycbv":
    dataset = YCBV_Segmentation(config)
elif config.dataset_name == "lmo":
    dataset = LMO_Segmentation(config)
elif config.dataset_name == "hb":
    dataset = HB_Segmentation(config)
elif config.dataset_name == "tudl":
    dataset = TUDL_Segmentation(config)
else:
    raise ValueError(f"Unsupported dataset: {config.dataset_name}")

# Cache the shared references and their DINO features for this dataset run.
seg_res = []
condition_batch = dataset[0]
first_query = {
    key: condition_batch[key]
    for key in ("scene_id", "image_id", "img_path", "img")
}
vis_cond_imgs = None
vis_cond_mask_ids = None
if vis_mask_enable:
    vis_cond_imgs = [
        inv_transform(cond_img) for cond_img in condition_batch["cond_imgs"]
    ]
    vis_cond_mask_ids = list(range(len(vis_cond_imgs)))

with condition_feature_cache(model, condition_batch) as conditions:
    cond_masks = conditions["cond_masks"]
    cond_obj_ids = conditions["cond_obj_ids"]
    cond_features = conditions["cond_features"]

    for idx in tqdm(range(len(dataset))):
        batch_data = first_query if idx == 0 else load_benchmark_query(dataset, idx)
        scene_id = batch_data["scene_id"]
        image_id = batch_data["image_id"]
        img = batch_data["img"].unsqueeze(0).cuda()

        with torch.no_grad():
            start = time.time()
            out_masks = model(
                img,
                cond_x=None,
                cond_x_mask=cond_masks,
                cond_features=cond_features,
            )
            infer_time = time.time() - start
            out_masks = out_masks[-1]

        out_masks = F.interpolate(
            out_masks, tuple(img_size), mode="bilinear", align_corners=False
        )
        instances_by_image, sem_mask = extract_instances(
            out_masks, cls_thresh=config.cls_thresh
        )

        for instance in instances_by_image[0]:
            class_id = instance["class_id"]
            obj_id = cond_obj_ids[class_id].item()
            mask = instance["mask"]
            seg_res.append({
                "scene_id": scene_id,
                "image_id": image_id,
                "category_id": obj_id,
                "bbox": instance["bbox"],
                "segmentation": mask_to_rle(mask),
                "score": instance["score"],
                "time": infer_time,
            })

        if vis_mask_enable:
            vis_img = inv_transform(img[0])
            predict = sem_mask.squeeze(0).squeeze(0)
            vis_img = cv2.resize(
                vis_img, tuple(img_size[::-1]), interpolation=cv2.INTER_LINEAR
            )
            vis_pred = visualize_image_mask_and_templates(
                vis_img, predict, vis_cond_imgs, vis_cond_mask_ids
            )
            save_dir = f"output/{dataset_name}_infer_vis"
            os.makedirs(save_dir, exist_ok=True)
            plt.imsave(
                os.path.join(save_dir, f"{scene_id}_{image_id}.png"),
                vis_pred.astype(np.uint8),
            )

# save the seg_res as a json file
with open(f"output/{config.method_name}_{dataset_name}-test.json", "w") as f:
    json.dump(seg_res, f)
