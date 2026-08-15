import os
import json
import logging
import numpy as np
import cv2
from PIL import Image
import torch
import torch.utils.data as data
from collections import defaultdict
from torchvision import transforms
import sys
sys.path.append('.')
from dataset.utils.img_utils import crop_and_resize

class YCBV_Segmentation(data.Dataset):
    def __init__(self, config, dataset_name="ycbv"):
        self.patch_size = config.patch_size
        self.condition_size = config.condition_size
        self.data_root = os.path.join(config.YCBV_Dataset.data_root, dataset_name, "test")
        self.render_template_root = os.path.join(config.YCBV_Dataset.condition_dir, dataset_name)
        self.obj_ids = config.YCBV_Dataset.obj_ids

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        ])

        self.total_data = []
        self.annotation = defaultdict(list)
        scenes = sorted(os.listdir(os.path.join(self.data_root)))
        for scene in scenes:
            scene_rgb_dir = os.path.join(self.data_root, scene, "rgb")
            frames = sorted([f for f in os.listdir(scene_rgb_dir) if f.endswith('.png')])
            scene_gt_path = os.path.join(self.data_root, scene, "scene_gt.json")
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)
            for frame in frames:
                frame_id = int(frame.split('.')[0])
                frame_gt = scene_gt[str(frame_id)]
                frame_image_path = os.path.join(scene_rgb_dir, frame)
                instance_info = []
                for inst_id, item in enumerate(frame_gt):
                    obj_id = item['obj_id']
                    obj_mask_path = os.path.join(self.data_root, scene, "mask_visib", f"{frame_id:06d}_{inst_id:06d}.png")
                    instance_info.append({
                        "obj_id": obj_id,
                        "mask_path": obj_mask_path,
                    })
                self.total_data.append({
                    "image_path": frame_image_path,
                    "instance_info": instance_info,
                    "scene_id": int(scene),
                    "image_id": frame_id,
                    })

        self.len_val = len(self.total_data)
        logging.info(f"YCBV Data size: {self.len_val}")

    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]
        instance_info = load_data["instance_info"]

        # load image
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        # resize the image
        img = self.crop_img(img, patch_size=self.patch_size)

        # load mask and load condition images for the objects in the image
        cond_imgs = []
        cond_masks = []
        cond_imgs_path = []
        cond_mask_ids = []
        cond_obj_ids = []
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for inst_id, inst in enumerate(instance_info):
            obj_id = inst['obj_id']
            # load mask
            obj_mask = np.array(Image.open(inst['mask_path']))
            obj_mask = self.crop_img(obj_mask, patch_size=self.patch_size)
            mask[obj_mask > 0] = inst_id + 1
            # load condition image
            template_dir = os.path.join(self.render_template_root, f"obj_{obj_id:06d}")
            image_path = os.path.join(template_dir, "rgb_0.png")
            mask_path = os.path.join(template_dir, "mask_0.png")
            cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
            cond_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # resize and crop the condition image
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)
            cond_mask_ids.append(inst_id + 1)
            cond_obj_ids.append(obj_id)

        # load other negative condition images
        for obj_id in self.obj_ids:
            if obj_id in [inst['obj_id'] for inst in instance_info]:
                continue
            template_dir = os.path.join(self.render_template_root, f"obj_{obj_id:06d}")
            image_path = os.path.join(template_dir, "rgb_0.png")
            mask_path = os.path.join(template_dir, "mask_0.png")
            cond_img = cv2.imread(image_path)[:,:,::-1]  # BGR to RGB
            cond_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # resize and crop the condition image
            cond_img, cond_mask = crop_and_resize(cond_img, cond_mask, size=self.condition_size, crop_rel_pad=0)
            cond_imgs.append(cond_img)
            cond_masks.append(cond_mask)
            cond_imgs_path.append(image_path)
            cond_mask_ids.append(0) # negative samples, set mask id to 0
            cond_obj_ids.append(obj_id)

        cond_imgs = [self.transform(img) for img in cond_imgs]
        cond_imgs = torch.stack(cond_imgs, dim=0)  # (N, 3, H, W)
        cond_masks = np.stack(cond_masks)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)
        cond_mask_ids = torch.tensor(cond_mask_ids, dtype=torch.long)  # (N,)
        cond_obj_ids = torch.tensor(cond_obj_ids)  # (N,)
        # random shuffle the condition images and mask ids
        perm = torch.randperm(cond_imgs.size(0))
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        cond_mask_ids = cond_mask_ids[perm]
        cond_obj_ids = cond_obj_ids[perm]
        cond_mask_ids -= 1 # shift with -1

        mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        # remap the mask ids to cond_imgs indices
        semantic_mask = self.remap_mask(mask, cond_mask_ids)
        # generate targets from semantic mask
        targets = self.generate_targets(semantic_mask, bg_val=-1)

        batch_data = {
            "scene_id": load_data["scene_id"],
            "image_id": load_data["image_id"],
            "img_path": image_path,
            "img": img,
            "mask": mask,
            "cond_imgs": cond_imgs,
            "cond_masks": cond_masks,
            "cond_mask_ids": cond_mask_ids,
            "cond_obj_ids": cond_obj_ids,
            "num_target": len(instance_info),
            "targets": targets,
        }
        return batch_data

    def __len__(self):
        return self.len_val

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int64') - 1
        return torch.from_numpy(target)

    def crop_img(self, img, patch_size):
        h, w = img.shape[:2]
        new_h = (h // patch_size) * patch_size
        new_w = (w // patch_size) * patch_size

        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped_img = img[top : top + new_h, left : left + new_w, ...]

        return cropped_img

    def remap_mask(self, mask, cond_mask_ids):
        """
        Remaps values in the mask to their corresponding index in cond_mask_ids.

        Args:
            mask (torch.Tensor): The original segmentation mask (H, W).
            cond_mask_ids (torch.Tensor): 1D tensor of class IDs.
                                        cond_mask_ids[i] is the class ID that should map to index i.

        Returns:
            torch.Tensor: The remapped mask where values are 0, 1, 2... corresponding to the index in cond_mask_ids.
                        Pixels not found in cond_mask_ids (or originally -1) are set to -1.
        """

        # Important: cond_mask_ids may contain -1 (negative templates / background after shift).
        # Using -1 to index a LUT will silently index the last element and corrupt the mapping.
        device = mask.device
        cond_mask_ids = cond_mask_ids.to(device)

        pos = cond_mask_ids >= 0
        if not torch.any(pos):
            return torch.full_like(mask, -1)

        pos_ids = cond_mask_ids[pos]
        max_id = int(pos_ids.max().item())

        # LUT maps original mask-id -> index into cond_imgs.
        lut = torch.full((max_id + 1,), -1, dtype=torch.long, device=device)
        all_indices = torch.arange(len(cond_mask_ids), device=device)
        lut[pos_ids] = all_indices[pos]

        new_mask = mask.clone()

        # Only ids within LUT range are indexable; everything else becomes -1.
        valid = (mask >= 0) & (mask <= max_id)
        new_mask[valid] = lut[mask[valid]]
        new_mask[(mask < 0) | (mask > max_id)] = -1

        return new_mask

    def generate_targets(self, semantic_mask, bg_val=-1):
        """
        Splits a semantic mask into separate binary masks for each class/instance.

        Args:
            semantic_mask: (H, W) tensor.
            bg_val: Background value to ignore (default -1).

        Returns:
            targets: Dictionary with keys:
                - "masks": (N, H, W) Tensor (0 or 1).
                - "labels": (N,) Tensor (Class IDs).
        """
        # 1. Get unique class IDs, excluding background
        unique_vals = torch.unique(semantic_mask)
        class_ids = unique_vals[unique_vals != bg_val]

        # Handle case with no foreground objects
        if len(class_ids) == 0:
            return {
                "masks": torch.zeros((0, *semantic_mask.shape), dtype=torch.long, device=semantic_mask.device),
                "labels": class_ids
            }

        # 2. Create binary masks using broadcasting
        # Compare (1, H, W) with (N, 1, 1) -> Result is (N, H, W) Boolean Tensor
        binary_masks = (semantic_mask.unsqueeze(0) == class_ids.view(-1, 1, 1))

        targets = {
            "masks": binary_masks.long(),  # Convert Bool to Long (0/1)
            "labels": class_ids,
        }

        return targets


if __name__ == '__main__':
    from hydra import initialize, compose
    # Load config
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="infer_config")
    dataset = YCBV_Segmentation(config)
    print(len(dataset))
    for i in range(10):
        dataset[i]

