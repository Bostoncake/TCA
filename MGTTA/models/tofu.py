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
            # !!!: note that this also ensures the prompt tokens at the front

    def merge(x: torch.Tensor, x_no_size: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        src_no_size, dst_no_size = x_no_size[..., ::2, :], x_no_size[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        unm_no_size = src_no_size.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        unm_norm = torch.norm(unm_no_size, dim=-1).unsqueeze(-1).expand(-1, -1, c)
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        src_no_size = src_no_size.gather(dim=-2, index=src_idx.expand(n, r, c))
        src_norm = torch.norm(src_no_size, dim=-1).unsqueeze(-1).expand(-1, -1, c)
        dst_norm = torch.norm(dst_no_size, dim=-1).unsqueeze(-1).expand(-1, -1, c)
        dst_norm = dst_norm.scatter_reduce(-2, dst_idx.expand(n, r, c), src_norm, reduce="amax")
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1), torch.cat([unm_norm[:, :1], dst_norm[:, :1], unm_norm[:, 1:], dst_norm[:, 1:]], dim=1).detach()
        else:
            assert num_prompts == 3 or num_prompts == 0
            if num_prompts == 3:
                return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:2], dst[:, 1:2], unm[:, 2:], dst[:, 2:]], dim=1), torch.cat([unm_norm[:, :1], dst_norm[:, :1], unm_norm[:, 1:2], dst_norm[:, 1:2], unm_norm[:, 2:], dst_norm[:, 2:]], dim=1).detach()
            else:
                return torch.cat([unm, dst], dim=1), torch.cat([unm_norm, dst_norm], dim=1).detach()

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
    
    def merge_size(x: torch.Tensor, mode="mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        else:
            return torch.cat([unm, dst], dim=1)

    return merge, unmerge, merge_size

def bipartite_soft_pruning(
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
    merge: Callable, merge_size: Callable, x: torch.Tensor, size: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x, x_norm = merge(x * size, x, mode="sum")
    size = merge_size(size, mode="sum")

    x = x / size
    x = x / (torch.norm(x, dim=-1).unsqueeze(-1).expand(-1, -1, x.shape[-1])) * x_norm      # Tofu MLERP merging
    return x, size

def merge_pruning(
    merge: Callable, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the merge function by taking a weighted average based on token size.
    Returns the merged tensor and the new token sizes.
    """

    x = merge(x, mode="sum")
    return x

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


def parse_r(num_layers: int, r: Union[List[int], Tuple[int, float], int]) -> List[int]:
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


class TofuPromptViTBlock(Block):
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
        if self.tofu_type == "prune":
            x_attn, metric = self.attn(self.norm1(x))
            x = x + self._drop_path1(x_attn)

            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _ = bipartite_soft_pruning(
                    metric,
                    r,
                    self._tome_info["class_token"],
                    self._tome_info["distill_token"],
                    self._tome_info["num_prompts"],
                )
                x = merge_pruning(merge, x)
        else:
            attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
            x_attn, metric = self.attn(self.norm1(x), attn_size)
            x = x + self._drop_path1(x_attn)

            r = self._tome_info["r"].pop(0)
            if r > 0:
                # Apply ToMe here
                merge, _, merge_size = bipartite_soft_matching(
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
                x, self._tome_info["size"] = merge_wavg(merge, merge_size, x, self._tome_info["size"])

        x = x + self._drop_path2(self.mlp(self.norm2(x)))
        return x


class TofuPromptViTAttention(Attention):
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

def make_tome_class_TofuPromptViT(transformer_class):
    class TofuPromptViT(transformer_class):
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

    return TofuPromptViT

def apply_patch_TofuPromptViT(
    model: PromptViT, tofu_r: int, tofu_sep = 6, trace_source: bool = False, prop_attn: bool = True
):
    """
    Applies ToMe to this transformer. Afterward, set r using model.r.

    If you want to know the source of each token (e.g., for visualization), set trace_source = true.
    The sources will be available at model._tome_info["source"] afterward.

    For proportional attention, set prop_attn to True. This is only necessary when evaluating models off
    the shelf. For trianing and for evaluating MAE models off the self set this to be False.
    """
    TofuPromptViT = make_tome_class_TofuPromptViT(model.__class__)

    model.__class__ = TofuPromptViT
    model.r = tofu_r
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
            module.__class__ = TofuPromptViTBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = TofuPromptViTAttention
    
    for i in range(len(model.vit.blocks)):
        if i < tofu_sep:
            model.vit.blocks[i].tofu_type = "prune"
            model.vit.blocks[i].attn.tofu_type = "prune"
        else:
            model.vit.blocks[i].tofu_type = "merge"
            model.vit.blocks[i].attn.tofu_type = "merge"