"""
Copyright to FOA Authors ICML 2024
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

from torch.autograd import Variable
from models.vpt import PromptViT
import cma
import numpy as np
import time
import math

from utils.cli_utils import accuracy, AverageMeter
from calibration_library.metrics import ECELoss
from queue import PriorityQueue
from quant_library.quant_layers.matmul import *
from models.tome_pyra import PYRA

RUNNING_IMAGNET_R = False

class FOA_BP(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self,
                model:PromptViT,
                optimizer,
                fitness_lambda = 30,
                use_distribution_loss = False,
                dist_loss_lambda = 0):
        super().__init__()
        self.optimizer = optimizer
        self.fitness_lambda = fitness_lambda

        self.model = model
        self.embed_dim = model.vit.embed_dim
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.hist_stat = None

        self.use_ori_img_stats = False
        self.ori_img_means = None
        self.ori_img_stds = None
        self.img_feat_lambda = 1

    def _update_hist(self, batch_mean):
        """Update overall test statistics, Eqn. (9)"""
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = 0.9 * self.hist_stat + 0.1 * batch_mean
            
    def _get_shift_vector(self):
        """Calculate shift direction, Eqn. (8)"""
        if self.hist_stat is None:
            return None
        else:
            return self.train_info[1][-self.embed_dim:] - self.hist_stat

    def forward(self, x):
        shift_vector = self._get_shift_vector()
        return_img_feat = self.use_ori_img_stats
        outputs, batch_mean = forward_and_get_loss(x, self.model, self.optimizer, self.fitness_lambda, self.train_info, shift_vector, self.imagenet_mask, return_img_feat=return_img_feat, img_feat_stats=(self.ori_img_stds, self.ori_img_means), img_feat_lambda=self.img_feat_lambda)
        self._update_hist(batch_mean[-self.embed_dim:])
        return outputs
    
    def obtain_origin_stat(self, train_loader, return_img_feat=False, count_flops=False):
        print('===> begin calculating mean and variance')
        self.model.eval()
        features = []
        img_feats = []
        with torch.no_grad():
            for _, dl in enumerate(train_loader):
                images = dl[0].cuda()
                feature = self.model.layers_cls_features(images, return_img_feat)
                if return_img_feat:
                    feature, img_feat = feature
                    img_feats.append(img_feat)
                features.append(feature)
                if count_flops:
                    break       # only process one batch when counting flops to save time
            features = torch.cat(features, dim=0)
            self.train_info = torch.std_mean(features, dim=0) # occupy 0.2MB 
            if return_img_feat:
                img_feats = torch.cat(img_feats, dim=0).permute(1, 0, 2)        # [layers, sample_n, dim]
                img_stds = []
                img_means = []
                for i in range(img_feats.shape[0]):
                    layer_i_img_feats = img_feats[i]
                    layer_i_std, layer_i_mean = torch.std_mean(layer_i_img_feats, dim=0)
                    img_stds.append(layer_i_std)
                    img_means.append(layer_i_mean)
                img_stds = torch.cat(img_stds, dim=0)
                img_means = torch.cat(img_means, dim=0)
                torch.save(img_stds.detach().cpu(), "statistics/origin_std.pt")
                torch.save(img_means.detach().cpu(), "statistics/origin_mean.pt")
        print('===> calculating mean and variance end')

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.hist_stat = None
    
    def load_origin_img_stats_and_set_lambda(self, img_feat_lambda):
        self.ori_img_means = torch.load("statistics/origin_mean.pt").cuda()
        self.ori_img_stds = torch.load("statistics/origin_std.pt").cuda()
        self.use_ori_img_stats = True
        self.img_feat_lambda = img_feat_lambda
    
    def init_cls_token(self):
        nn.init.normal_(self.model.vit.cls_token, std=1e-6)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def copy_model_only(model):
    source_model = deepcopy(model)
    for param in source_model.parameters():
        param.detach_()
    return source_model

criterion_mse = nn.MSELoss(reduction='mean').cuda()

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_get_loss(images, model:PromptViT, optimizer, fitness_lambda, train_info, shift_vector, imagenet_mask, return_img_feat=False, img_feat_stats=None, img_feat_lambda=1):
    features = model.layers_cls_features_with_prompts(images, return_img_feat=return_img_feat)
    if return_img_feat and img_feat_stats[0] is not None and img_feat_stats[1] is not None:
        features, img_feats = features
        img_std, img_mean = torch.std_mean(img_feats, dim=0)
        img_std = img_std.view(-1)
        img_mean = img_mean.view(-1)
        img_std_mse, img_mean_mse = criterion_mse(img_std, img_feat_stats[0]), criterion_mse(img_mean, img_feat_stats[1])
        img_token_loss = img_feat_lambda * (img_std_mse + img_mean_mse)
    
    batch_std, batch_mean = torch.std_mean(features, dim=0)
    std_mse, mean_mse = criterion_mse(batch_std, train_info[0]), criterion_mse(batch_mean, train_info[1])
    # NOTE: $lambda$ should be 20 for ImageNet-R!!
    discrepancy_loss = fitness_lambda * (std_mse + mean_mse)

    cls_features = features[:, -model.vit.embed_dim:]
    del features

    output = model.vit.head(cls_features)
    if imagenet_mask:
        output = output[:, imagenet_mask]
    entropy_loss = softmax_entropy(output).mean()
    loss = discrepancy_loss + entropy_loss

    # with torch.no_grad():
    #     if shift_vector is not None:
    #         output = model.vit.head(cls_features + 1. * shift_vector)
    #         if imagenet_mask:
    #             output = output[:, imagenet_mask]
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return output, batch_mean

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

# def configure_model(model):
#     """Configure model for use with tent."""
#     # train mode, because tent optimizes the model to minimize entropy
#     model.train()
#     # disable grad, to (re-)enable only what tent updates
#     model.requires_grad_(False)
#     # configure norm for tent updates: enable grad + force batch statisics
#     for m in model.modules():
#         if isinstance(m, nn.BatchNorm2d):
#             m.requires_grad_(True)
#             # force use of batch stats in train and eval modes
#             m.track_running_stats = False
#             m.running_mean = None
#             m.running_var = None
#         if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
#             m.requires_grad_(True)
#     return model

# def collect_params(model):
#     """Collect the affine scale + shift parameters from batch norms.
#     Walk the model's modules and collect all batch normalization parameters.
#     Return the parameters and their names.
#     Note: other choices of parameterization are possible!
#     """
#     params = []
#     names = []
#     for nm, m in model.named_modules():
#         if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
#             for np, p in m.named_parameters():
#                 if np in ['weight', 'bias']:  # weight is scale, bias is shift
#                     params.append(p)
#                     names.append(f"{nm}.{np}")
#     return params, names

# Modify for PYRA
def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
        if isinstance(m, PYRA):
            m.requires_grad_(True)
    # TODO: change if cls_token is trainable
    for nm, m in model.named_parameters():
        if "cls_token" in nm:
            m.requires_grad_(True)
        # if "dispatch_prompt" in nm:
        #     m.requires_grad_(True)
        if "protect_prompt" in nm:
            m.requires_grad_(True)
        if "cls_ssf" in nm:
            m.requires_grad_(True)
    return model

def configure_model_baselines(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if isinstance(m, PYRA):
            for np, p in m.named_parameters():
                if 'pyra' in np:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    # TODO: change if cls_token is trainable
    for nm, m in model.named_parameters():
        if "cls_token" in nm:
            params.append(m)
            names.append(f"{nm}")
    return params, names

def collect_params_baselines(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def collect_params_sep_cls(model, lr_model, lr_cls):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_with_lr = []
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if isinstance(m, PYRA):
            for np, p in m.named_parameters():
                if 'pyra' in np:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    # for nm, m in model.named_parameters():
    #     if "dispatch_prompt" in nm:
    #         params.append(m)
    #         names.append(f"{nm}")
    for nm, m in model.named_parameters():
        if "protect_prompt" in nm:
            params.append(m)
            names.append(f"{nm}")
        if "cls_ssf" in nm:
            params.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": params,
        "lr": lr_model
    })
    # TODO: change if cls_token is trainable
    cls_token = []
    for nm, m in model.named_parameters():
        if "cls_token" in nm:
            cls_token.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": cls_token,
        "lr": lr_cls
    })
    return params_with_lr, names

def collect_params_sep_cls_sep_ssf(model, lr_model, lr_cls, lr_ssf):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_with_lr = []
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if isinstance(m, PYRA):
            for np, p in m.named_parameters():
                if 'pyra' in np:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    # for nm, m in model.named_parameters():
    #     if "dispatch_prompt" in nm:
    #         params.append(m)
    #         names.append(f"{nm}")
    for nm, m in model.named_parameters():
        if "protect_prompt" in nm:
            params.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": params,
        "lr": lr_model
    })
    # TODO: change if cls_token is trainable
    cls_token = []
    for nm, m in model.named_parameters():
        if "cls_token" in nm:
            cls_token.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": cls_token,
        "lr": lr_cls
    })
    cls_ssf = []
    for nm, m in model.named_parameters():
        if "cls_ssf" in nm:
            cls_ssf.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": cls_ssf,
        "lr": lr_ssf
    })
    return params_with_lr, names

def collect_params_peft(model, lr_model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params_with_lr = []
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    for nm, m in model.named_parameters():
        if "_prompts" in nm or "lora" in nm:
            params.append(m)
            names.append(f"{nm}")
    params_with_lr.append({
        "params": params,
        "lr": lr_model
    })
    return params_with_lr, names

# Modify for PYRA pre-training
def configure_model_pyra_pretrain(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # only configure PYRA for pre-training updates
    for m in model.modules():
        if isinstance(m, PYRA):
            m.requires_grad_(True)
    return model

def collect_params_pyra_pretrain(model):
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, PYRA):
            for np, p in m.named_parameters():
                if 'pyra' in np:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

# Modify for Token Dispatch pre-training
def configure_model_td_pretrain(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # only configure PYRA for pre-training updates
    for nm, m in model.named_modules():
        if "dispatch_prompt" in nm:
            m.requires_grad_(True)
    return model

def collect_params_td_pretrain(model):
    params = []
    names = []
    for nm, m in model.named_parameters():
        if "dispatch_prompt" in nm:
            params.append(m)
            names.append(f"{nm}")
    return params, names

# Modify for protect prompt pre-training
def configure_model_protect_prompt_pretrain(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # only configure PYRA for pre-training updates
    for nm, m in model.named_parameters():
        if "protect_prompt" in nm:
            m.requires_grad_(True)
    return model

def collect_params_protect_prompt_pretrain(model):
    params = []
    names = []
    for nm, m in model.named_parameters():
        if "protect_prompt" in nm:
            params.append(m)
            names.append(f"{nm}")
    return params, names