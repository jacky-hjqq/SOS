import os
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

class HB_Segmentation(data.Dataset):
    def __init__(self, config, dataset_name="hb"):
        self.patch_size = config.patch_size
        self.condition_size = config.condition_size
        self.data_root = os.path.join(config.HB_Dataset.data_root, dataset_name, "test")
        self.render_template_root = os.path.join(config.HB_Dataset.condition_dir, dataset_name)
        self.obj_ids = config.HB_Dataset.obj_ids

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
            for frame in frames:
                frame_id = int(frame.split('.')[0])
                frame_image_path = os.path.join(scene_rgb_dir, frame)
                self.total_data.append({
                    "image_path": frame_image_path,
                    "scene_id": int(scene),
                    "image_id": frame_id,
                    })

        self.len_val = len(self.total_data)
        logging.info(f"HB Data size: {self.len_val}")

    def __getitem__(self, index):
        load_data = self.total_data[index]
        image_path = load_data["image_path"]

        # load image
        img = Image.open(image_path).convert('RGB')
        img = np.array(img)
        # resize the image
        img = self.crop_img(img, patch_size=self.patch_size)

        # load mask and load condition images for the objects in the image
        cond_imgs = []
        cond_masks = []
        cond_imgs_path = []
        cond_obj_ids = []
        for obj_id in self.obj_ids:
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
            cond_obj_ids.append(obj_id)

        cond_imgs = [self.transform(img) for img in cond_imgs]
        cond_imgs = torch.stack(cond_imgs, dim=0)  # (N, 3, H, W)
        cond_masks = np.stack(cond_masks)
        cond_masks = torch.tensor(cond_masks, dtype=torch.float32)  # (N, H, W)
        cond_obj_ids = torch.tensor(cond_obj_ids)  # (N,)
        # random shuffle the condition images and mask ids
        perm = torch.randperm(cond_imgs.size(0))
        cond_imgs = cond_imgs[perm]
        cond_masks = cond_masks[perm]
        cond_obj_ids = cond_obj_ids[perm]

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        batch_data = {
            "scene_id": load_data["scene_id"],
            "image_id": load_data["image_id"],
            "img_path": image_path,
            "img": img,
            "cond_imgs": cond_imgs,
            "cond_masks": cond_masks,
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
    dataset = HB_Segmentation(config)
    print(len(dataset))
    for i in range(10):
        dataset[i]


