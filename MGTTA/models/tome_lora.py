# Note: This file is based on https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import math
from models.vpt import PromptViT


def do_nothing(x, mode=None):
    return x

def bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
    class_token: bool = False,
    distill_token: bool = False,
    num_prompts: int = 0,
) -> Tuple[Callable, Callable]:
    """
    Applies ToMe with a balanced matching set (50%, 50%).

    Input size is [batch, tokens, channels].
    r indicates the number of tokens to remove (max 50% of tokens).

    Extra args:
     - class_token: Whether or not there's a class token.
     - distill_token: Whether or not there's also a distillation token.

    When enabled, the class token and distillation tokens won't get merged.
    """
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1
    if num_prompts > 0:
        protected += num_prompts

    # We can only reduce by a maximum of 50% tokens
    t = metric.shape[1]
    r = min(r, (t - protected) // 2)

    if r <= 0:
        return do_nothing, do_nothing

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf
        if num_prompts > 0:
            for i in range(num_prompts):
                indicator = (protected - num_prompts + i) % 2
                idx = (protected - num_prompts + i) // 2
                if indicator:
                    scores[..., :, idx] = -math.inf
                else:
                    scores[..., idx, :] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]  # Unmerged Tokens
        src_idx = edge_idx[..., :r, :]  # Merged Tokens
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            # Sort to ensure the class token is at the start
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            assert num_prompts == 3 or num_prompts == 0
            if num_prompts == 3:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:2], dst[:, 1:2], unm[:, 2:], dst[:, 2:]], dim=1)
            else:
                return torch.cat([unm, dst], dim=1)

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        n, _, c = unm.shape

        src = dst.gather(dim=-2, index=dst_idx.expand(n, r, c))

        out = torch.zeros(n, metric.shape[1], c, device=x.device, dtype=x.dtype)

        out[..., 1::2, :] = dst
        out.scatter_(dim=-2, index=(2 * unm_idx).expand(n, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=(2 * src_idx).expand(n, r, c), src=src)

        return out

    return merge, unmerge

def merge_wavg(
    merge: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")

    x = x / size
    return x, size


def merge_source(
    merge: Callable, x: torch.Tensor, source: torch.Tensor = None
) -> torch.Tensor:
    """
    For source tracking. Source is an adjacency matrix between the initial tokens and final merged groups.
    x is used to find out how many tokens there are in case the source is None.
    """
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device)[None, ...].expand(n, t, t)

    source = merge(source, mode="amax")
    return source


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int, str]) -> List[int]:
    """
    Process a constant r or r schedule into a list for use internally.

    r can take the following forms:
     - int: A constant number of tokens per layer.
     - Tuple[int, float]: A pair of r, inflection.
       Inflection describes there the the reduction / layer should trend
       upward (+1), downward (-1), or stay constant (0). A value of (r, 0)
       is as providing a constant r. (r, -1) is what we describe in the paper
       as "decreasing schedule". Any value between -1 and +1 is accepted.
     - List[int]: A specific number of tokens per layer. For extreme granularity.
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
        except:
            raise NotImplementedError("r should be a string, like: 8_9_10_11")
    elif isinstance(r, int):
        pass
    else:
        raise NotImplementedError

    min_val = int(r * (1.0 - inflect))
    max_val = 2 * r - min_val
    step = (max_val - min_val) / (num_layers - 1)

    return [int(min_val + step * i) for i in range(num_layers)]


class ToMeBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMeAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)


class ToMePromptViTBlock(Block):
    """
    Modifications:
     - Apply ToMe between the attention and mlp blocks
     - Compute and propogate token size and potentially the token sources.
    """

    def _drop_path1(self, x):
        return self.drop_path1(x) if hasattr(self, "drop_path1") else self.drop_path(x)

    def _drop_path2(self, x):
        return self.drop_path2(x) if hasattr(self, "drop_path2") else self.drop_path(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: this is copied from timm.models.vision_transformer.Block with modifications.
        attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
        x_attn, metric = self.attn(self.norm1(x), attn_size)
        x = x + self._drop_path1(x_attn)

        r = self._tome_info["r"].pop(0)
        if r > 0:
            # Apply ToMe here
            merge, _ = bipartite_soft_matching(
                metric,
                r,
                self._tome_info["class_token"],
                self._tome_info["distill_token"],
                self._tome_info["num_prompts"],
            )
            if self._tome_info["trace_source"]:
                self._tome_info["source"] = merge_source(
                    merge, x, self._tome_info["source"]
                )
            x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class ToMePromptViTAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this is copied from timm.models.vision_transformer.Attention with modifications.
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply proportional attention
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Return k as well here
        return x, k.mean(1)
    
class LoRALinear(nn.Module):
    """
    A Linear layer with LoRA (Low-Rank Adaptation).
    Wraps an existing linear layer and adds trainable low-rank matrices A and B.
    Output = original_output + x @ A @ B * scaling
    """
    
    def __init__(
        self, 
        original_linear: nn.Linear, 
        rank: int = 4, 
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming and B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward
        result = self.original_linear(x)
        # Add LoRA contribution
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return result + lora_out
    
    def merge_weights(self):
        """Merge LoRA weights into original weights for inference efficiency."""
        with torch.no_grad():
            self.original_linear.weight.add_(
                (self.lora_A @ self.lora_B).T * self.scaling
            )
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from original weights."""
        with torch.no_grad():
            self.original_linear.weight.sub_(
                (self.lora_A @ self.lora_B).T * self.scaling
            )


class ToMeLoRAAttention(Attention):
    """
    Attention with LoRA applied to specified projections AND ToMe support.
    Returns the mean of k over heads for ToMe metric computation.
    """
    
    def apply_lora(
        self, 
        rank: int = 4, 
        alpha: float = 1.0, 
        dropout: float = 0.0,
        target_modules: List[str] = None,
    ):
        """
        Apply LoRA to this attention layer.
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha (scaling factor)
            dropout: Dropout rate for LoRA
            target_modules: Which projections to apply LoRA to. 
                           Options: 'q', 'k', 'v', 'proj' (output projection)
                           Default: ['q', 'v']
        """
        if target_modules is None:
            target_modules = ['q', 'v']
        
        self._lora_target_modules = target_modules
        self._lora_rank = rank
        self._lora_alpha = alpha
        
        # The qkv is a single linear layer, so we need to handle it specially
        # We'll create separate LoRA adapters for q, k, v portions
        embed_dim = self.qkv.in_features
        
        if 'q' in target_modules:
            self.lora_q_A = nn.Parameter(torch.zeros(embed_dim, rank))
            self.lora_q_B = nn.Parameter(torch.zeros(rank, embed_dim))
            nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_q_B)
        
        if 'k' in target_modules:
            self.lora_k_A = nn.Parameter(torch.zeros(embed_dim, rank))
            self.lora_k_B = nn.Parameter(torch.zeros(rank, embed_dim))
            nn.init.kaiming_uniform_(self.lora_k_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_k_B)
        
        if 'v' in target_modules:
            self.lora_v_A = nn.Parameter(torch.zeros(embed_dim, rank))
            self.lora_v_B = nn.Parameter(torch.zeros(rank, embed_dim))
            nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_v_B)
        
        if 'proj' in target_modules:
            self.proj = LoRALinear(self.proj, rank=rank, alpha=alpha, dropout=dropout)
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_scaling = alpha / rank
        
        # Freeze original qkv weights
        self.qkv.weight.requires_grad = False
        if self.qkv.bias is not None:
            self.qkv.bias.requires_grad = False

    def forward(
        self, x: torch.Tensor, size: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        
        # Original qkv projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply LoRA adaptations
        x_dropped = self.lora_dropout(x) if hasattr(self, 'lora_dropout') else x
        
        if hasattr(self, 'lora_q_A'):
            lora_q = (x_dropped @ self.lora_q_A @ self.lora_q_B * self.lora_scaling)
            lora_q = lora_q.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q + lora_q
        
        if hasattr(self, 'lora_k_A'):
            lora_k = (x_dropped @ self.lora_k_A @ self.lora_k_B * self.lora_scaling)
            lora_k = lora_k.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k = k + lora_k
        
        if hasattr(self, 'lora_v_A'):
            lora_v = (x_dropped @ self.lora_v_A @ self.lora_v_B * self.lora_scaling)
            lora_v = lora_v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + lora_v
        
        # Standard attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply proportional attention for ToMe
        if size is not None:
            attn = attn + size.log()[:, None, None, :, 0]
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Return k as well for ToMe metric
        return x, k.mean(1)


def make_tome_class(transformer_class):
    class ToMeVisionTransformer(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().forward(*args, **kwargs)

    return ToMeVisionTransformer

def make_tome_class_ToMePromptViT(transformer_class):
    class ToMePromptViT(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def layers_cls_features(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.vit.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().layers_cls_features(*args, **kwargs)
        
        def layers_cls_features_with_prompts(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.vit.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None

            return super().layers_cls_features_with_prompts(*args, **kwargs)
        
        def forward(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError
        
        def forward_features(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

    return ToMePromptViT


def apply_patch_ToMe(
    model: VisionTransformer, tome_r: int, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMeVisionTransformer = make_tome_class(model.__class__)

    model.__class__ = ToMeVisionTransformer
    model.r = tome_r
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.cls_token is not None,
        "distill_token": False,
    }

    if hasattr(model, "dist_token") and model.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention

def make_tome_class_ToMeLoRAPromptViT(transformer_class):
    """Create a ToMe + LoRA compatible PromptViT class."""
    
    class ToMeLoRAPromptViT(transformer_class):
        def layers_cls_features(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.vit.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            return super().layers_cls_features(*args, **kwargs)
        
        def layers_cls_features_with_prompts(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info["r"] = parse_r(len(self.vit.blocks), self.r)
            self._tome_info["size"] = None
            self._tome_info["source"] = None
            return super().layers_cls_features_with_prompts(*args, **kwargs)
        
        def get_lora_parameters(self) -> List[nn.Parameter]:
            """Get all LoRA parameters for optimization."""
            lora_params = []
            for name, param in self.named_parameters():
                if 'lora_' in name:
                    lora_params.append(param)
            return lora_params
        
        def merge_lora_weights(self):
            """Merge all LoRA weights for efficient inference."""
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    module.merge_weights()
        
        def forward(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError
        
        def forward_features(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError
    
    return ToMeLoRAPromptViT

def apply_patch_ToMePromptViT_LoRA(
    model: PromptViT, 
    tome_r: int, 
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_layers: str = None,
    lora_targets: List[str] = None,
    trace_source: bool = False, 
    prop_attn: bool = True,
):
    """
    Applies ToMe and LoRA to a PromptViT model.
    
    Args:
        model: The PromptViT model to patch
        tome_r: Number of tokens to merge per layer (or schedule)
        lora_rank: Rank for LoRA matrices
        lora_alpha: Alpha scaling factor for LoRA
        lora_dropout: Dropout rate for LoRA
        lora_layers: Comma-separated layer indices for LoRA (None = all layers)
        lora_targets: Which projections to apply LoRA to ('q', 'k', 'v', 'proj')
        trace_source: Whether to track token sources
        prop_attn: Whether to use proportional attention
    """
    if lora_targets is None:
        lora_targets = ['q', 'v']
    
    # Parse LoRA layers
    num_layers = len(model.vit.blocks)
    if lora_layers is not None:
        lora_layer_indices = [int(l) for l in lora_layers.strip().split(",")]
    else:
        lora_layer_indices = list(range(num_layers))
    
    # Apply ToMe + LoRA class transformation
    ToMeLoRAPromptViT = make_tome_class_ToMeLoRAPromptViT(model.__class__)
    model.__class__ = ToMeLoRAPromptViT
    model.r = tome_r
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.vit.cls_token is not None,
        "distill_token": False,
        "num_prompts": model.num_prompts if hasattr(model, 'num_prompts') else 0,
    }
    
    if hasattr(model.vit, "dist_token") and model.vit.dist_token is not None:
        model._tome_info["distill_token"] = True
    
    # Apply patches to blocks
    for layer_idx, block in enumerate(model.vit.blocks):
        block.__class__ = ToMePromptViTBlock
        block._tome_info = model._tome_info
        
        # Replace attention module and optionally apply LoRA
        for name, module in block.named_children():
            if isinstance(module, Attention):
                module.__class__ = ToMeLoRAAttention
                
                if layer_idx in lora_layer_indices:
                    module.apply_lora(
                        rank=lora_rank,
                        alpha=lora_alpha,
                        dropout=lora_dropout,
                        target_modules=lora_targets,
                    )