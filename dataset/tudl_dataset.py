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

class TUDL_Segmentation(data.Dataset):
    def __init__(self, config, dataset_name="tudl"):
        self.patch_size = config.patch_size
        self.condition_size = config.condition_size
        self.data_root = os.path.join(config.TUDL_Dataset.data_root, dataset_name, "test")
        self.render_template_root = os.path.join(config.TUDL_Dataset.condition_dir, dataset_name)
        self.obj_ids = config.TUDL_Dataset.obj_ids

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
        logging.info(f"TUDL Data size: {self.len_val}")

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
            cond_mask_ids.append(0) # negative samples, set mask id to 0
            cond_obj_ids.append(obj_id)

        cond_imgs = [self.transform(img) for img in cond_imgs]
        cond_imgs = torch.stack(cond_imgs, dim=0)  # (N, 3, H, W)
        cond_masks = np.stack(cond_masks)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)
        cond_mask_ids = torch.tensor(cond_mask_ids, dtype=torch.long)  # (N,)
        cond_obj_ids = torch.tensor(cond_obj_ids)  # (N,)

        cond_mask_ids -= 1 # shift with -1

        mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

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

if __name__ == '__main__':
    from hydra import initialize, compose
    # Load config
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="infer_config")
    dataset = TUDL_Segmentation(config)
    print(len(dataset))
    for i in range(600):
        dataset[i]

