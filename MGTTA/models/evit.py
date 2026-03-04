# Note: This file is based on https://github.com/youweiliang/evit/blob/master/evit.py

from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import math
from models.vpt import PromptViT


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

def do_nothing(x, mode=None):
    return x

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

class EViTPromptViTBlock(Block):
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
        B, N, C = x.shape
        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x))
        x = x + self._drop_path1(tmp)

        if index is not None:
            non_cls_prompt = x[:, 1+self.num_prompts:]
            x_others = torch.gather(non_cls_prompt, dim=1, index=index)  # [B, left_tokens, C]

            if True:        # We use fuse_token in EViT by default
                compl = complement_idx(idx, N - 1 - self.num_prompts)
                non_topk = torch.gather(non_cls_prompt, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))

                non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)
                extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class EViTPromptViTAttention(Attention):
    """
    Modifications:
     - Apply proportional attention
     - Return the mean of k over heads from attention
    """

    def forward(
        self, x: torch.Tensor
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
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        left_tokens = N - 1 - self.num_prompts
        if self.keep_rate < 1:  # double check the keep rate
            left_tokens = math.ceil(self.keep_rate * (N - 1 - self.num_prompts))
        elif isinstance(self.keep_rate, float):
            return x, None, None, None, left_tokens
        elif isinstance(self.keep_rate, int):   # keep_rate > 1 when using ToMe-like schedule
            left_tokens -= self.keep_rate
        
        if left_tokens == N - 1 - self.num_prompts:
            return x, None, None, None, left_tokens
        assert left_tokens >= 1
        cls_attn = attn[:, :, 0, 1+self.num_prompts:]  # [B, H, N-1]
        cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
        _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens]
        index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

        return x, index, idx, cls_attn, left_tokens


def make_evit_class_EViTPromptViT(transformer_class):
    class EViTPromptViT(transformer_class):
        """
        Modifications:
        - Initialize r, token size, and token sources.
        """

        def set_keep_rate(self, keep_rate: Tuple = None, base_keep_rate: float = None, prune_layer: list = [3,6,9], prune_token_by_layer: int = None):
            if keep_rate is not None:
                assert len(keep_rate) == len(self.vit.blocks)
                for i in range(len(self.vit.blocks)):
                    self.vit.blocks[i].keep_rate = keep_rate[i]
                    self.vit.blocks[i].attn.keep_rate = keep_rate[i]
            elif base_keep_rate is not None:
                layer_keep_rate = math.pow(base_keep_rate, float(1/3))
                for i in range(len(self.vit.blocks)):
                    if i in prune_layer:        # [3, 6, 9] for ViT-B
                        print(f"Set layer index {i} as evit pruning layer, rate {layer_keep_rate:.2f}.")
                        self.vit.blocks[i].keep_rate = layer_keep_rate
                        self.vit.blocks[i].attn.keep_rate = layer_keep_rate
                    else:
                        self.vit.blocks[i].keep_rate = 1.0
                        self.vit.blocks[i].attn.keep_rate = 1.0
            elif prune_token_by_layer is not None:
                for i in range(len(self.vit.blocks)):
                    print(f"Set layer index {i} as evit pruning layer, prune {prune_token_by_layer[i]:d} tokens.")
                    self.vit.blocks[i].keep_rate = prune_token_by_layer[i]
                    self.vit.blocks[i].attn.keep_rate = prune_token_by_layer[i]
            else:
                raise NotImplementedError
        
        def forward(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError
        
        def forward_features(self, *args, **kwargs) -> torch.Tensor:
            raise NotImplementedError

    return EViTPromptViT

def apply_patch_EViTPromptViT(
    model: PromptViT, keep_rate: Tuple = None, base_keep_rate: float = None, prune_layer: str = "3,6,9", prune_token_by_layer: int = None
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    EViTPromptViT = make_evit_class_EViTPromptViT(model.__class__)

    model.__class__ = EViTPromptViT
    prune_layer = [int(layer) for layer in prune_layer.strip().split(",")]
    if prune_token_by_layer is not None:
        prune_token_by_layer = parse_r(len(model.vit.blocks), prune_token_by_layer)
    model.set_keep_rate(keep_rate, base_keep_rate, prune_layer, prune_token_by_layer)

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = EViTPromptViTBlock
            module.num_prompts = model.num_prompts
        elif isinstance(module, Attention):
            module.__class__ = EViTPromptViTAttention
            module.num_prompts = model.num_prompts