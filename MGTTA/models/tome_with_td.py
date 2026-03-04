# Note: This file is based on https://github.com/facebookresearch/ToMe/blob/main/tome/patch/timm.py

from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Block, VisionTransformer
import math
from models.vpt import PromptViT
from timm.models.layers import trunc_normal_


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
        
        # token dispatch
        if self.td_rate < 1:
            dispatch_logits = x[:, 1:, :] @ self.dispatch_prompt
            token_select = _gumbel_sigmoid(dispatch_logits, top_k=self.td_rate, hard=True, training=False)
            token_select = torch.cat([token_select.new_ones(x.shape[0], 1, 1, requires_grad=True), token_select.unsqueeze(-1)], dim=1)
            x_td = token_select * x

            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x_td))))
        else:

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

def apply_patch_ToMePromptViT_td(
    model: PromptViT, tome_r: int, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    ToMePromptViT = make_tome_class_ToMePromptViT(model.__class__)

    model.__class__ = ToMePromptViT
    model.r = tome_r
    model._tome_info = {
        "r": model.r,
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "prop_attn": prop_attn,
        "class_token": model.vit.cls_token is not None,
        "distill_token": False,
        "num_prompts": model.num_prompts,
    }

    if hasattr(model.vit, "dist_token") and model.vit.dist_token is not None:
        model._tome_info["distill_token"] = True

    for module in model.modules():
        if isinstance(module, Block):
            module.__class__ = ToMePromptViTBlock
            module._tome_info = model._tome_info
            module.forward = module.__class__.forward.__get__(module, module.__class__)
        elif isinstance(module, Attention):
            module.__class__ = ToMePromptViTAttention

def _gumbel_sigmoid(
    logits, tau=5, hard=False, eps=1e-10, training = True, top_k=0.8
):
    # this is based on https://github.com/NUS-HPC-AI-Lab/Dynamic-Tuning/blob/d1744f0b9366f79ad9b78f586e479af34e81807a/models/dynamic_adapter.py#L25
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # TODO: complete this when necessary to cut down time
        # n_toks = int(top_k*logits.shape[1])
        # _, indices = torch.topk(y_soft, n_toks, dim=1)
        
        # B, N = logits.shape
        # y_hard = torch.zeros(B, n_toks, N, dtype=torch.int)

        # # Create batch indices for scatter
        # batch_indices = torch.arange(B).unsqueeze(1).expand(B, top_k)
        # row_indices = torch.arange(top_k).unsqueeze(0).expand(B, top_k)

        # y_hard[batch_indices, row_indices, indices] = 1
        # y_hard = torch.zeros_like(
        #     logits, memory_format=torch.legacy_contiguous_format
        # ).masked_fill(y_soft > threshold, 1.0)


        n_toks = int(top_k*logits.shape[1])
        _, indices = torch.topk(y_soft, n_toks, dim=1)
        
        B, N = logits.shape
        y_hard = torch.zeros(B, N, dtype=torch.int).cuda()

        # Create batch indices for scatter
        batch_indices = torch.arange(B).unsqueeze(1).expand(B, n_toks)
        y_hard[batch_indices, indices] = 1
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

def forward_block_td_pretraining_wo_tome(self, x):
    if self.td_rate < 1:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))

        # token dispatch
        dispatch_logits = x[:, 1:, :] @ self.dispatch_prompt
        token_select = _gumbel_sigmoid(dispatch_logits, top_k=self.td_rate, hard=True, training=True)
        token_select = torch.cat([token_select.new_ones(x.shape[0], 1, 1, requires_grad=True), token_select.unsqueeze(-1)], dim=1)
        x_td = token_select * x

        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x_td))))
    else:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    return x

def apply_td_pretraining_wo_tome(
    model: PromptViT, td_layers: str = "3,6,9", td_rate: float = 0.8
):
    td_layers = [int(layer) for layer in td_layers.strip().split(",")]
    for i, block in enumerate(model.vit.blocks):
        if i in td_layers:
            block.td_rate = td_rate
            block.dispatch_prompt = nn.Parameter(torch.zeros(model.vit.embed_dim,))
            trunc_normal_(block.dispatch_prompt, std=.02)       # copied from ViT nn.Linear init
            bound_method = forward_block_td_pretraining_wo_tome.__get__(block, block.__class__)
            setattr(block, 'forward', bound_method)
        else:
            block.td_rate = 1