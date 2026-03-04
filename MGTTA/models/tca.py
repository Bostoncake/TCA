# Note: This file implements TCA (Token Condensation as Adaptation, ICCV 2025)
# adapted for the timm ViT backbone used in MGTTA.
# Original TCA paper: https://arxiv.org/abs/...
# Original CLIP-based implementation is in TCA/clip/model.py (ResidualAttentionBlock_Ours)
# and TCA/clip/utils.py (coreset_averaging, k_center_greedy).
#
# Key adaptations vs. the original CLIP implementation:
#  - timm uses batch-first layout [B, N, C]; original TCA uses [N, B, C].
#  - prune_r (int) sets the number of tier-3 tokens removed per layer,
#    following the same convention as EViT's prune_token_by_layer.
#  - The domain-aware class-token cache (requires CLIP text weights) is omitted;
#    only the token condensation block is ported here as a standalone baseline.

from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import math

from models.vpt import PromptViT


# ---------------------------------------------------------------------------
# Schedule helpers (identical to evit.py)
# ---------------------------------------------------------------------------

def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int, str]) -> List[int]:
    """
    Process a constant r or r schedule into a per-layer list.

    r can be:
     - int   : same value applied to every layer.
     - str   : underscore-separated per-layer values, e.g. "0_0_0_4_0_0_4_0_0_4_0_0".
     - list  : explicit per-layer list.
     - tuple : (r, inflect) for a linearly interpolated schedule.
    """
    inflect = 0
    if isinstance(r, list):
        if len(r) < num_layers:
            r = r + [0] * (num_layers - len(r))
        return list(r)
    elif isinstance(r, tuple):
        r, inflect = r
    elif isinstance(r, str):
        try:
            r = [int(layer_r) for layer_r in r.strip().split("_")]
            if len(r) < num_layers:
                r = r + [0] * (num_layers - len(r))
            return list(r)
        except Exception:
            raise NotImplementedError("r should be a string like: 4_4_4 or 0_0_0_4_0_0_4_0_0_4_0_0")
    elif isinstance(r, int):
        pass
    else:
        raise NotImplementedError

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)
    return [int(min_val + step * i) for i in range(num_layers)]


# ---------------------------------------------------------------------------
# Index helpers (identical to evit.py / TCA/clip/model.py)
# ---------------------------------------------------------------------------

def complement_idx(idx: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor; complement is along the last dimension.
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


# ---------------------------------------------------------------------------
# Coreset averaging (adapted from TCA/clip/utils.py)
# Input shape adapted: [B, n_tokens, C]  (timm batch-first format)
# Original TCA shape:  [n_tokens, B, C]  (CLIP sequence-first format)
# ---------------------------------------------------------------------------

def k_center_greedy(token: torch.Tensor, num_cluster: int) -> torch.Tensor:
    """
    Greedy farthest-point sampling.

    Args:
        token:       [n, C] float tensor of token embeddings.
        num_cluster: number of cluster centers to select.

    Returns:
        centers: [num_cluster, C] selected center embeddings.
    """
    n = token.shape[0]
    num_cluster = min(num_cluster, n)

    # Random initial center
    start = torch.randint(n, (1,)).item()
    center_indices = [start]
    min_distances = torch.norm(token.float() - token[start].float(), dim=1)

    for _ in range(num_cluster - 1):
        next_center = torch.argmax(min_distances).item()
        center_indices.append(next_center)
        new_distances = torch.norm(token.float() - token[next_center].float(), dim=1)
        min_distances = torch.minimum(min_distances, new_distances)

    return torch.stack([token[i] for i in center_indices])


def coreset_averaging(token: torch.Tensor, num_centers: int) -> torch.Tensor:
    """
    Cluster tokens via k-center greedy selection, then compute an
    inverse-distance-weighted average within each cluster.

    Args:
        token:       [B, n_tokens, C]
        num_centers: number of output representative tokens per sample.

    Returns:
        weighted_tokens: [B, num_centers, C]  (same dtype as input)
    """
    B, n, C = token.shape
    num_centers = min(num_centers, n)
    weighted_tokens = torch.zeros(B, num_centers, C, device=token.device, dtype=token.dtype)

    for b in range(B):
        token_b = token[b]                                           # [n, C]
        centers = k_center_greedy(token_b, num_cluster=num_centers)  # [num_centers, C]

        distances = torch.cdist(token_b.float(), centers.float())    # [n, num_centers]
        cluster_assignment = torch.argmin(distances, dim=1)          # [n]

        for i in range(num_centers):
            cluster_tokens = token_b[cluster_assignment == i]        # [k, C]
            if cluster_tokens.shape[0] == 0:
                continue

            cluster_center = centers[i]
            token_distances = torch.norm(
                cluster_tokens.float() - cluster_center.float(), dim=1
            )  # [k]

            # Inverse-distance weighting; protect against exact matches
            epsilon = 1e-8
            token_distances = token_distances.clamp(min=epsilon)
            weights = 1.0 / token_distances
            weights = weights / weights.sum()

            weighted_tokens[b, i] = (cluster_tokens * weights.unsqueeze(-1)).sum(dim=0)

    return weighted_tokens


# ---------------------------------------------------------------------------
# Modified timm Attention: returns per-head CLS attention for token ranking
# ---------------------------------------------------------------------------

class TCAPromptViTAttention(Attention):
    """
    Drop-in replacement for timm Attention that additionally returns the
    per-head CLS-to-patch attention map used by TCAPromptViTBlock.

    Extra attribute set by apply_patch_TCAPromptViT:
        tca_r        (int): tokens to remove in this layer; 0 means passthrough.
        num_prompts  (int): number of VPT prompt tokens inserted after CLS.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Identical to timm Attention.forward — recomputed to expose attn weights.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Skip token ranking if no pruning is configured for this layer.
        if self.tca_r <= 0:
            return x, None

        # ------------------------------------------------------------------
        # Cross-head token ranking (TCA paper, §3.2)
        # cls_attn_heads: [B, H, n_patch]  — CLS-to-patch attention per head
        # ------------------------------------------------------------------
        n_patch = N - 1 - self.num_prompts          # patch tokens only
        cls_attn_heads = attn[:, :, 0, 1 + self.num_prompts:]  # [B, H, n_patch]

        # Rank tokens within each head (ascending → position 0 = least attended)
        sort_indices = torch.sort(cls_attn_heads, dim=2)[1]     # [B, H, n_patch]
        ranking_positions = torch.zeros_like(sort_indices, dtype=torch.float)
        ranking_positions.scatter_(
            2,
            sort_indices,
            torch.arange(n_patch, device=x.device, dtype=torch.float)
            .unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1),
        )
        # Average rank across heads, normalized to [0, 1]
        # Higher value → more consistently attended → more important
        avg_ranking = ranking_positions.mean(dim=1) / max(n_patch - 1, 1)  # [B, n_patch]

        # Raw CLS attention averaged across heads (used for topk selection)
        cls_attn = cls_attn_heads.mean(dim=1)  # [B, n_patch]

        return x, cls_attn


# ---------------------------------------------------------------------------
# Modified timm Block: three-tier token condensation
# ---------------------------------------------------------------------------

class TCAPromptViTBlock(Block):
    """
    Drop-in replacement for timm Block implementing TCA's three-tier
    token condensation at the layers where tca_r > 0.

    Three tiers (based on CLS-attention score, highest → lowest):
      Tier 1 — kept as-is              : top (n_patch - 3*tca_r) tokens
      Tier 2 — coreset-averaged        : next (2*tca_r) tokens
                                         → num_coreset_centers representatives
      Tier 3 — attention-weighted fuse : remaining tca_r tokens → 1 extra token

    Net token count after condensation:
      (n_patch - 3*tca_r) + num_coreset_centers + 1

    Extra attributes set by apply_patch_TCAPromptViT:
        tca_r              (int): tokens to remove (tier-3 count).
        num_coreset_centers (int): coreset cluster centers for tier-2 (default 4).
        num_prompts        (int): VPT prompt count.
    """

    def _drop_path1(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        tmp, cls_attn = self.attn(self.norm1(x))
        x = x + self._drop_path1(tmp)

        if cls_attn is not None:
            # ----------------------------------------------------------------
            # Token layout: [CLS | prompts (num_prompts) | patch tokens (n_patch)]
            # ----------------------------------------------------------------
            n_patch = N - 1 - self.num_prompts
            r = self.tca_r

            # Tier sizes
            n_tier3  = r                           # fused into 1 extra token
            n_tier12 = n_patch - n_tier3           # = left_tokens
            n_tier2  = 2 * r                       # clustered
            n_tier1  = n_tier12 - n_tier2          # kept as-is

            # Guard: clamp in case of extreme r values
            n_tier1 = max(n_tier1, 0)
            n_tier2 = max(n_tier2, 0)
            num_centers = min(self.num_coreset_centers, max(n_tier2, 1))

            patch_tokens = x[:, 1 + self.num_prompts:, :]  # [B, n_patch, C]

            # ----------------------------------------------------------------
            # Tier 1+2 selection (top n_tier1 + n_tier2 by cls_attn)
            # ----------------------------------------------------------------
            n_top = n_tier1 + n_tier2
            _, idx_top = torch.topk(cls_attn, n_top, dim=1, largest=True, sorted=True)
            idx_tier1 = idx_top[:, :n_tier1]   # [B, n_tier1]
            idx_tier2 = idx_top[:, n_tier1:]   # [B, n_tier2]

            # ----------------------------------------------------------------
            # Tier 3: complement of the top selection → fuse into 1 extra token
            # ----------------------------------------------------------------
            compl = complement_idx(idx_top, n_patch)  # [B, n_tier3]
            non_topk = torch.gather(
                patch_tokens, dim=1,
                index=compl.unsqueeze(-1).expand(-1, -1, C)
            )
            non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, n_tier3]
            extra_token = torch.sum(
                non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
            )  # [B, 1, C]

            # ----------------------------------------------------------------
            # Tier 1: gather and keep
            # ----------------------------------------------------------------
            x_keep = torch.gather(
                patch_tokens, dim=1,
                index=idx_tier1.unsqueeze(-1).expand(-1, -1, C)
            )  # [B, n_tier1, C]

            # ----------------------------------------------------------------
            # Tier 2: gather and coreset-average
            # ----------------------------------------------------------------
            x_cluster = torch.gather(
                patch_tokens, dim=1,
                index=idx_tier2.unsqueeze(-1).expand(-1, -1, C)
            )  # [B, n_tier2, C]
            x_merged = coreset_averaging(x_cluster, num_centers=num_centers)  # [B, num_centers, C]

            # ----------------------------------------------------------------
            # Reassemble: [CLS | prompts | tier1 | tier2-centers | extra]
            # ----------------------------------------------------------------
            prefix = x[:, : 1 + self.num_prompts, :]
            x = torch.cat([prefix, x_keep, x_merged, extra_token], dim=1)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# apply_patch helper — identical structure to apply_patch_EViTPromptViT
# ---------------------------------------------------------------------------

def make_tca_class(transformer_class):
    class TCAPromptViT(transformer_class):
        """
        Mixin that adds set_tca_r() to any PromptViT subclass.
        """

        def set_tca_r(
            self,
            tca_r: Union[List[int], int, str],
            prune_layer: List[int] = None,
            num_coreset_centers: int = 4,
        ):
            if prune_layer is None:
                prune_layer = [3, 6, 9]

            tca_r_list = parse_r(len(self.vit.blocks), tca_r)

            for i, block in enumerate(self.vit.blocks):
                if i in prune_layer:
                    r = tca_r_list[i]
                    print(
                        f"[TCA] Layer {i}: prune_layer=True, "
                        f"tca_r={r}, num_coreset_centers={num_coreset_centers}."
                    )
                    block.tca_r = r
                    block.attn.tca_r = r
                    block.num_coreset_centers = num_coreset_centers
                else:
                    block.tca_r = 0
                    block.attn.tca_r = 0
                    block.num_coreset_centers = num_coreset_centers

        def forward(self, *args, **kwargs):
            raise NotImplementedError

        def forward_features(self, *args, **kwargs):
            raise NotImplementedError

    return TCAPromptViT


def apply_patch_TCAPromptViT(
    model: PromptViT,
    tca_r: Union[List[int], int, str] = 4,
    prune_layer: str = "3,6,9",
    num_coreset_centers: int = 4,
):
    """
    Patch a PromptViT (wrapping a timm VisionTransformer) to use TCA's
    hierarchical token condensation at the specified layers.

    Args:
        model:              PromptViT instance to patch (modified in-place).
        tca_r:              Number of tier-3 tokens removed per pruning layer.
                            Can be an int, underscore-separated str schedule
                            (e.g. "4_4_4"), or list.  Follows the same
                            convention as EViT's prune_token_by_layer.
        prune_layer:        Comma-separated layer indices where TCA is applied
                            (default "3,6,9", same as EViT).
        num_coreset_centers: Number of cluster centers for tier-2 tokens
                            (default 4, as in the original TCA paper).
    """
    TCAPromptViT = make_tca_class(model.__class__)
    model.__class__ = TCAPromptViT

    prune_layers = [int(l) for l in prune_layer.strip().split(",")]
    model.set_tca_r(
        tca_r=tca_r,
        prune_layer=prune_layers,
        num_coreset_centers=num_coreset_centers,
    )

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = TCAPromptViTBlock
            module.num_prompts = model.num_prompts
        elif isinstance(module, Attention):
            module.__class__ = TCAPromptViTAttention
            module.num_prompts = model.num_prompts
