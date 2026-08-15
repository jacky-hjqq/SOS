import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from model.layers.embed import QueryEmbedder, ConditionEmbedder
from model.layers.decoder import Decoder
from model.layers.feature_extractor import DINOv3

class OCDecoder(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int] = (480, 640),
        condition_dims: Tuple[int, int] = (224, 224),
        patch_size: int = 16,
        hidden_size: int = 1024,
        block_depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        model_type: str = "dinov3_vitl16",
        enable_dino_proj: bool = False,
    ):
        super().__init__()

        self.cond_dim = condition_dims
        self.patch_size = patch_size
        self.enable_dino_proj = enable_dino_proj

        if not model_type.startswith("dinov3"):
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.feature_extractor = DINOv3(freeze_weights=True, model_type=model_type)

        if self.enable_dino_proj:
            self.image_proj = nn.Linear(hidden_size, hidden_size)
            self.condition_proj = nn.Linear(hidden_size, hidden_size)

        self.image_embedder = QueryEmbedder(
            input_dims=input_dims,
            patch_size=patch_size,
            embed_dim=hidden_size,
        )

        self.condition_embedder = ConditionEmbedder(
            input_dims=condition_dims,
            patch_size=patch_size,
            embed_dim=hidden_size,
        )

        self.decoder = Decoder(
            patch_size=patch_size,
            hidden_size=hidden_size,
            block_depth=block_depth,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
        )

        # initialize a learnable query token
        self.query_token = nn.Embedding(1, hidden_size)

    def project_features(self, image_features, cond_features):
        """
        Projects the features to the hidden size if projection is enabled.
        Handles tensors with unexpected dimensions and recovers the original shape after projection.
        """
        # Handle image_features (4D tensor)
        if image_features.dim() == 4:  # Expected shape: [batch_size, feature_dim, height, width]
            batch_size, feature_dim, height, width = image_features.shape
            num_patches = height * width
            image_features = image_features.permute(0, 2, 3, 1).reshape(batch_size, num_patches, feature_dim)
            image_features = self.image_proj(image_features)
            image_features = image_features.view(batch_size, height, width, -1).permute(0, 3, 1, 2)  # Recover original shape
        else:
            raise ValueError(f"Unexpected shape for image_features: {image_features.shape}")

        # Handle cond_features (5D tensor)
        if cond_features.dim() == 5:  # Expected shape: [batch_size, num_conditions, feature_dim, height, width]
            batch_size, num_conditions, feature_dim, height, width = cond_features.shape
            num_patches = height * width
            cond_features = cond_features.permute(0, 1, 3, 4, 2).reshape(batch_size * num_conditions, num_patches, feature_dim)
            cond_features = self.condition_proj(cond_features)
            cond_features = cond_features.view(batch_size, num_conditions, height, width, -1).permute(0, 1, 4, 2, 3)  # Recover original shape
        else:
            raise ValueError(f"Unexpected shape for cond_features: {cond_features.shape}")

        return image_features, cond_features

    def extract_condition_features(self, cond_images):
        """Extract reusable DINO features for condition images."""
        return self.feature_extractor(cond_images)

    def forward(self, images, cond_images, cond_masks, cond_features=None):
        # downsample the cond_x_mask to match the patch tokens
        cond_masks_lr = F.max_pool2d(
            cond_masks.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).bool()

        # Extract query features and reuse condition features when provided.
        image_features = self.feature_extractor(images)
        if cond_features is None:
            if cond_images is None:
                raise ValueError("cond_images are required when cond_features are not provided")
            cond_features = self.extract_condition_features(cond_images)

        # linear projection to hidden size (if enabled)
        if self.enable_dino_proj:
            image_features, cond_features = self.project_features(image_features, cond_features)

        # embed conditions and queries with PE
        image_tokens = self.image_embedder(image_features)
        cond_tokens = self.condition_embedder(cond_features)

        # query tokens: expand to num_conditions
        nc = cond_features.shape[1]
        query_tokens = self.query_token.weight.unsqueeze(0).expand(
            images.shape[0], nc, -1
        )  # (b, nc, hidden_size)

        # decode masks
        masks = self.decoder(images, image_tokens, cond_tokens, query_tokens, cond_masks_lr)

        return masks
