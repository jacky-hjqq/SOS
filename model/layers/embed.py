from typing import Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.init import trunc_normal_

def interpolate_embedding(embedding, dim_before, dim_after):
    # 1d embedding
    pe_classes = embedding  # 1 max_num_classes dim
    dim = pe_classes.shape[-1]
    pe_classes = pe_classes.reshape(1, dim, dim_before)
    pe_classes = nn.functional.interpolate(pe_classes, size=dim_after)

    return pe_classes.reshape(1, dim_after, 1, dim)

class LearnedPE(nn.Module):
    def __init__(self, *size: Sequence[int]):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(size))

        self.initialize()

    def initialize(self):
        trunc_normal_(self.pe, std=0.02)

    def __call__(self):
        return self.pe

class QueryEmbedder(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int],
        patch_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.num_rows = input_dims[0] // patch_size
        self.num_cols = input_dims[1] // patch_size
        self.num_patches = self.num_rows * self.num_cols

        self.pe_patches = LearnedPE(
            1,
            self.num_patches,
            embed_dim,
        )

    def forward(self, patch_tokens):
        # patch_tokens: (b, c, ph, pw) where ph = h // patch_size, pw = w // patch_size
        input_dims = patch_tokens.shape[2:]

        # Get position embeddings
        assert (self.num_rows, self.num_cols) == input_dims
        pe = self.pe_patches()

        # Rearrange patch_tokens to (b, ph*pw, c)
        patch_tokens = rearrange(patch_tokens, "b c ph pw -> b (ph pw) c")

        # Add position embeddings
        patch_tokens = patch_tokens + pe

        return patch_tokens


class ConditionEmbedder(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int],
        patch_size: int,
        embed_dim: int,
        max_num_classes: int = 50,
    ):
        super().__init__()

        self.num_rows = input_dims[0] // patch_size
        self.num_cols = input_dims[1] // patch_size
        self.num_patches = self.num_rows * self.num_cols
        self.max_num_classes = max_num_classes

        self.pe_patches = LearnedPE(
            1,
            1,
            self.num_patches,
            embed_dim,
        )
        self.pe_classes = LearnedPE(
            1,
            max_num_classes,
            1,
            embed_dim,
        )

    def _get_pos_embeddings(
        self, input_dims: Tuple[int, int], num_classes: int
    ):
        patch_embedding = self._get_patch_embeddings(input_dims)
        class_embedding = self._get_class_embeddings(num_classes)

        assert patch_embedding.ndim == class_embedding.ndim, (
            f"Patch embedding and class embedding should have the same number of dimensions, "
            f"but got {patch_embedding.ndim} and {class_embedding.ndim}"
        )

        return patch_embedding + class_embedding

    def _get_patch_embeddings(self, input_dims: Tuple[int, int]):
        if (self.num_rows, self.num_cols) == input_dims:
            return self.pe_patches()
        else:
            raise NotImplementedError(
                "Currently only support templates of uniform shape"
            )

    def _get_class_embeddings(self, num_classes: int):
        if num_classes == self.max_num_classes:
            return self.pe_classes()
        elif num_classes > self.max_num_classes:
            class_embed = interpolate_embedding(
                self.pe_classes(), self.max_num_classes, num_classes
            )
            return class_embed

        return self.pe_classes()[:, :num_classes]

    def forward(self, patch_tokens):
        # x: (b, n, c, h, w)
        nc, input_dims= patch_tokens.shape[1], patch_tokens.shape[3:]

        # Get position embeddings
        assert (self.num_rows, self.num_cols) == input_dims
        pe = self._get_pos_embeddings(input_dims, nc)

        # Rearrange patch_tokens to (b, n, ph*pw, c)
        patch_tokens = rearrange(patch_tokens, "b n c ph pw -> b n (ph pw) c")

        # Add position embeddings
        patch_tokens = patch_tokens + pe

        return patch_tokens
