import torch
from torch import nn
import math
from torch.nn import functional as F
from einops import rearrange
from model.layers.scale_block import ScaleBlock
from model.layers.attention import (
    MaskedCrossAttention,
    SelfAttention,
)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(hidden_size)
        self.mask_cross_attn_cond = MaskedCrossAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mask_cross_attn_img = MaskedCrossAttention(hidden_size, num_heads, dropout)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.self_attn = SelfAttention(hidden_size, num_heads, dropout)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm4 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        image_tokens: torch.Tensor,      # [B, ph*pw, c_dim]
        condition_tokens: torch.Tensor,  # [B, nc, ph_c*pw_c, c_dim]
        query_tokens: torch.Tensor,      # [B, nc, c_dim]
        attn_mask: torch.Tensor = None,  # Optional attention mask
        cond_masks: torch.Tensor = None, # Optional condition masks
    ):
        # 1. Condition Cross-Attention (Local 1-to-1 frame mapping)
        B, N_c, D = query_tokens.shape
        _, _, S_cond, _ = condition_tokens.shape

        # Reshape query to [(B*N), 1, D]
        q_reshaped = self.norm1(query_tokens).contiguous().view(B * N_c, 1, D)
        # Reshape condition to [(B*N), S_cond, D]
        c_reshaped = condition_tokens.contiguous().view(B * N_c, S_cond, D)
        # Reshape cond_masks to [num_heads*B*N, 1, S_cond]
        cond_masks = cond_masks.transpose(0, 1) # (N_heads, B*N, S_cond) -> (B*N, N_heads, S_cond)
        cond_masks = cond_masks.unsqueeze(2).reshape(-1, 1, S_cond)  # (B*N, N_heads, S_cond) -> (B*N, N_heads, 1, S_cond) -> (N_heads*B*N, 1, S_cond)

        # Parallel computation on batch dimension:
        attn_out = self.mask_cross_attn_cond(
            q_reshaped, c_reshaped,
            memory_mask=cond_masks,
            memory_key_padding_mask=None,
        )
        query_tokens = query_tokens + attn_out.contiguous().view(B, N_c, D)

        # 2. Image Cross-Attention (Global context)
        q_norm = self.norm2(query_tokens)
        query_tokens = self.mask_cross_attn_img(
            q_norm,
            image_tokens,
            memory_mask=attn_mask,
            memory_key_padding_mask=None,
        )

        # 3. Self-Attention
        query_tokens = query_tokens + self.self_attn(self.norm3(query_tokens))

        # 4. FFN
        query_tokens = query_tokens + self.mlp(self.norm4(query_tokens))

        return query_tokens

class Decoder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        hidden_size: int,
        num_heads: int,
        block_depth: int,
        mlp_ratio=4.0,
        dropout: float = 0.0,
        num_blocks=4,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.block_depth = block_depth
        self.num_blocks = num_blocks

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(block_depth)
            ]
        )

        # mask decoder
        self.decoder_norm = nn.LayerNorm(hidden_size)
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        num_upscale = max(1, int(math.log2(patch_size)) - 2)
        self.upscale = nn.Sequential(
            *[ScaleBlock(hidden_size) for _ in range(num_upscale)],
        )

    def forward(self, images, image_tokens, condition_tokens, query_tokens, cond_masks):
        predictions_mask = []
        target_size = (images.shape[2]//self.patch_size, images.shape[3]//self.patch_size)

        # initialize cond_masks
        cond_masks = ~(cond_masks.bool()) # inverse the mask to match the expected format of nn.MultiheadAttention
        cond_masks = (cond_masks.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)).bool()
        cond_masks = cond_masks.detach()
        cond_masks[torch.where(cond_masks.sum(-1) == cond_masks.shape[-1])] = False

        # initial prediction heads
        outputs_mask, attn_mask = self.forward_prediction_heads(query_tokens, image_tokens, attn_mask_target_size=target_size)

        # transformer blocks
        for i, block in enumerate(self.blocks):
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            query_tokens = block(image_tokens, condition_tokens, query_tokens, attn_mask, cond_masks)
            # Compute loss every 2 blocks (at layers 2, 4, 6, 8, etc.)
            if (i + 1) % 2 == 0:
                outputs_mask, attn_mask = self.forward_prediction_heads(query_tokens, image_tokens, attn_mask_target_size=target_size)
                predictions_mask.append(outputs_mask)

        outputs_mask = self.forward_prediction_heads(query_tokens, image_tokens, attn_mask_target_size=target_size, return_attn_mask=False)

        return predictions_mask

    def forward_prediction_heads(self, output, image_tokens, attn_mask_target_size, return_attn_mask=True):
        decoder_output = self.decoder_norm(output)
        mask_embed = self.mask_head(decoder_output)
        image_tokens = rearrange(image_tokens, "b (ph pw) c -> b c ph pw",
                                ph=(attn_mask_target_size[0]),
                                pw=(attn_mask_target_size[1]))
        image_tokens = self.upscale(image_tokens)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, image_tokens)

        if return_attn_mask is False:
            return outputs_mask

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_mask, attn_mask
