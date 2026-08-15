import os
import pickle
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import LambdaLR
from model.model import OCDecoder
from dataset.objaverse_dataset import Objaverse_Segmentation
from dataset.gso_dataset import GSO_Segmentation
from dataset.ycbv_dataset import YCBV_Segmentation
from dataset.composed_dataset import create_dataset
from dataset.utils.vis_utils import inv_transform, visualize_pred_gt_mask
from model.infer_utils import extract_instances, remap_target
import torch.distributed as dist
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

class Image_Seg_Module(pl.LightningModule):
    def __init__(self, config, mode="train", cache_dir=".dataset_cache"):
        super().__init__()
        self.config = config
        self.batch_size = config.batch_size
        self.img_size = config.input_dims

        self.net = OCDecoder(
            input_dims=config.input_dims,
            condition_dims=config.condition_dims,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            block_depth=config.block_depth,
            num_heads=config.num_heads,
            dropout=config.dropout,
            model_type=config.backbone,
        )

        self.criterion = MaskClassificationLoss()

        if mode == "train":
            # Load Training Dataset
            if config.train_dataset == "gso":
                # Load or build GSO dataset
                gso_cache_file = os.path.join(cache_dir, f"gso_dataset.pkl")
                if config.use_cache and os.path.exists(gso_cache_file):
                    print(f"Loading GSO dataset from cache: {gso_cache_file}")
                    try:
                        with open(gso_cache_file, 'rb') as f:
                            self.trainset = pickle.load(f)
                        print(f"Successfully loaded GSO dataset: {len(self.trainset)} samples")
                    except Exception as e:
                        print(f"Failed to load GSO cache: {e}")
                        print("Rebuilding GSO dataset...")
                        self.trainset = GSO_Segmentation(config)
                else:
                    self.trainset = GSO_Segmentation(config)

            elif config.train_dataset == "objaverse":
                # Load or build Objaverse dataset
                objaverse_cache_file = os.path.join(cache_dir, f"objaverse_dataset.pkl")
                if config.use_cache and os.path.exists(objaverse_cache_file):
                    print(f"Loading Objaverse dataset from cache: {objaverse_cache_file}")
                    try:
                        with open(objaverse_cache_file, 'rb') as f:
                            self.trainset = pickle.load(f)
                        print(f"Successfully loaded Objaverse dataset: {len(self.trainset)} samples")
                    except Exception as e:
                        print(f"Failed to load Objaverse cache: {e}")
                        print("Rebuilding Objaverse dataset...")
                        self.trainset = Objaverse_Segmentation(config)
                else:
                    self.trainset = Objaverse_Segmentation(config)

            elif config.train_dataset == "all":
                self.trainset = create_dataset(
                                    config,
                                    use_cache=config.use_cache,
                                    use_weighted_sampling=config.use_weighted_sampling)
            # Load Validation Dataset
            self.valset = YCBV_Segmentation(config)

            self.train_loss_epoch = MeanMetric()
            self.val_loss_epoch = MeanMetric(sync_on_compute=False)

    def extract_condition_features(self, cond_images):
        return self.net.extract_condition_features(cond_images)

    def forward(self, x, cond_x, cond_x_mask, cond_features=None):
        return self.net(x, cond_x, cond_x_mask, cond_features=cond_features)

    def training_step(self, batch, batch_idx):
        img = batch["img"]
        target = batch["mask"]
        cond_imgs = batch["cond_imgs"]
        cond_masks = batch["cond_masks"]
        cond_mask_ids = batch["cond_mask_ids"]

        # Forward
        out_mask_per_block = self.forward(img, cond_imgs, cond_masks)

        # Compute supervision
        target_mapped = remap_target(target, cond_mask_ids) # map the target with condition mask ids

        losses_all_blocks = {}
        for i, out_mask in enumerate(list(out_mask_per_block)):
            mask_loss = self.criterion(out_mask, target_mapped)
            out_mask = F.interpolate(out_mask, tuple(self.img_size), mode="bilinear", align_corners=False)
            losses = {"mask_loss": mask_loss}
            losses = {f"{key}{i}": value for key, value in losses.items()}
            losses_all_blocks |= losses

        train_loss = self.criterion.loss_total(losses_all_blocks, self.log, prefix="train")
        self.train_loss_epoch.update(train_loss)

        # Log learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        with torch.no_grad():
            if batch_idx % 100 == 0:
                _, sem_mask = extract_instances(out_mask)
                predict = sem_mask.squeeze(0).squeeze(0)

                # visualize the prediction and target (only for the first batch)
                vis_img = inv_transform(img[0])
                vis_cond_imgs = [inv_transform(cond_img) for cond_img in cond_imgs[0]]
                vis_cond_mask_ids = list(range(len(vis_cond_imgs)))
                gt = target_mapped.detach().cpu().numpy()[0]
                vis_mask = visualize_pred_gt_mask(
                    vis_img, predict, gt, vis_cond_imgs, vis_cond_mask_ids
                )
                self.logger.experiment.log({
                    "train/vis_seg": wandb.Image(vis_mask, caption="train vis pred & gt"),
                })

        return train_loss

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        target = batch["mask"]
        cond_imgs = batch["cond_imgs"]
        cond_masks = batch["cond_masks"]
        cond_mask_ids = batch["cond_mask_ids"]

        with torch.no_grad():
            # Forward
            out_mask_per_block = self.forward(img, cond_imgs, cond_masks)

            # Compute supervision
            target_mapped = remap_target(target, cond_mask_ids) # map the target with condition mask ids

            losses_all_blocks = {}
            for i, out_mask in enumerate(list(out_mask_per_block)):
                mask_loss = self.criterion(out_mask, target_mapped)
                out_mask = F.interpolate(out_mask, tuple(self.img_size), mode="bilinear", align_corners=False)
                losses = {"mask_loss": mask_loss}
                losses = {f"{key}{i}": value for key, value in losses.items()}
                losses_all_blocks |= losses

            val_loss = self.criterion.loss_total(losses_all_blocks, self.log, prefix="val")
            self.val_loss_epoch.update(val_loss)

            # visualization
            if batch_idx % 25 == 0:
                _, sem_mask = extract_instances(out_mask)
                predict = sem_mask.squeeze(0).squeeze(0)

                # visualize the prediction and target (only for the first batch)
                vis_img = inv_transform(img[0])
                vis_cond_imgs = [inv_transform(cond_img) for cond_img in cond_imgs[0]]
                vis_cond_mask_ids = list(range(len(vis_cond_imgs)))
                gt = target_mapped.detach().cpu().numpy()[0]
                vis_mask = visualize_pred_gt_mask(
                    vis_img, predict, gt, vis_cond_imgs, vis_cond_mask_ids
                )
                self.logger.experiment.log({
                    "val/vis_seg": wandb.Image(vis_mask, caption="val vis pred & gt"),
                })

        return val_loss

    def on_train_epoch_start(self):
        self.train_loss_epoch.reset()

    def on_train_epoch_end(self):
        avg_train_loss = self.train_loss_epoch.compute()
        self.log("train_loss_epoch", avg_train_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_loss_epoch.reset()

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_epoch.compute()
        self.log("val_loss_epoch", avg_val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=4,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.base_lr)

        scheduler = self.get_scheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            hold_steps=self.config.hold_steps,
            decay_end=self.config.decay_end_steps,
            base_lr=self.config.base_lr,
            min_lr=self.config.min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    def get_scheduler(self, optimizer, warmup_steps, hold_steps, decay_end, base_lr, min_lr):
        def lr_lambda(current_step):
            # --- Warmup: 0 → base_lr ---
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # --- Hold: constant base_lr ---
            elif current_step < hold_steps:
                return 1.0
            # --- Decay: base_lr → min_lr ---
            elif current_step < decay_end:
                progress = (current_step - hold_steps) / float(decay_end - hold_steps)
                return 1.0 - (1.0 - min_lr / base_lr) * progress
            # --- After decay_end: fixed min_lr ---
            else:
                return min_lr / base_lr

        return LambdaLR(optimizer, lr_lambda)

class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int = 12544,
        oversample_ratio: float = 3.0,
        importance_sample_ratio: float = 0.75,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = 1.0
        self.dice_coefficient = 1.0

    @torch.compiler.disable
    def forward(
        self,
        pred_masks_logits,
        gt_masks,
    ):
        # map the gt_masks to format required by Mask2FormerLoss
        batch_size = pred_masks_logits.shape[0]
        num_q = pred_masks_logits.shape[1]
        idx = torch.arange(num_q, device=pred_masks_logits.device)
        indices = [(idx, idx) for _ in range(batch_size)]
        gt_masks_list = self.extract_masks(gt_masks, num_q)

        loss_masks = self.loss_masks(pred_masks_logits, gt_masks_list, indices)

        # add loss
        loss_total = None
        for loss_key, loss in loss_masks.items():
            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        return loss_total

    def loss_total(self, losses_all_layers, log_fn, prefix="train") -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/{prefix}_{loss_key}", loss, sync_dist=True)
            if loss_total is None:
                loss_total = loss
            else:
                loss_total = torch.add(loss_total, loss)

        log_fn(f"losses/{prefix}_loss", loss_total, sync_dist=True, prog_bar=True)

        return loss_total

    def loss_masks(self, pred_masks_logits, gt_masks, indices):
        loss_masks = super().loss_masks(pred_masks_logits, gt_masks, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=pred_masks_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def extract_masks(self, gt_mask, num_q):
        """
        Args:
            gt_mask (Tensor): Input tensor of shape (B, H, W) containing instance indices.
            num_q (int): Total number of masks to extract (indices from 0 to num_q-1).

        Returns:
            gt_mask_list (list): A list of B tensors, each with shape (num_q, H, W).
        """
        # 1. Get the device of the input tensor to ensure consistency
        device = gt_mask.device

        # 2. Create a query vector containing indices from 0 to num_q-1
        # Shape: (num_q,)
        query_indices = torch.arange(num_q, device=device)

        # 3. Use broadcasting to perform element-wise comparison
        # gt_mask.unsqueeze(1) shape: (B, 1, H, W)
        # query_indices.view(1, num_q, 1, 1) shape: (1, num_q, 1, 1)
        # Resulting masks shape: (B, num_q, H, W)
        masks = (gt_mask.unsqueeze(1) == query_indices.contiguous().view(1, num_q, 1, 1))

        # 4. Convert boolean masks to float (or long) for downstream computations
        # Non-existent indices in gt_mask will result in all-zero slices
        masks = masks.float()

        # 5. Unbind the tensor along the batch dimension to create a list
        # Each tuple item has shape (num_q, H, W).
        gt_mask_list = list(torch.unbind(masks, dim=0))

        return gt_mask_list
