""" ConViT Model

@article{d2021convit,
  title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  author={d'Ascoli, St{\'e}phane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  journal={arXiv preprint arXiv:2103.10697},
  year={2021}
}

Paper link: https://arxiv.org/abs/2103.10697
Original code: https://github.com/facebookresearch/convit, original copyright below
"""
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
'''These modules are adapted from those of timm, see
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
'''

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, PatchEmbed, Mlp
from timm.models.vision_transformer_hybrid import HybridEmbed
import math

import torch
import torch.nn as nn


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'fixed_input_size': True,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # ConViT
    'convit_tiny_sn': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_tiny.pth"),
    'convit_small_sn': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_small.pth"),
    'convit_base_sn': _cfg(
        url="https://dl.fbaipublicfiles.com/convit/convit_base.pth")
}
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class GPSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,pen_for_qkv=None,
                 locality_strength=1.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.pen_for_qkv=pen_for_qkv
        self.locality_strength = locality_strength

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_mask = nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.pos_proj = nn.Linear(3, num_heads)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))
        self.rel_indices: torch.Tensor = torch.zeros(1, 1, 1, 3)  # silly torchscript hack, won't work with None
    def init_uv(self):
        Wq,Wk=self.qk.weight.chunk(2,dim=0)
        u1 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        u2 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        u3 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        
        v1 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        v2 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        v3 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        
        u1.data = l2normalize(u1.data)
        u2.data = l2normalize(u2.data)
        u3.data = l2normalize(u3.data)
        v1.data = l2normalize(v1.data)
        v2.data = l2normalize(v2.data)
        v3.data = l2normalize(v3.data)

        self.register_parameter("Wq_u", u1)
        self.register_parameter("Wk_u", u2)
        self.register_parameter("Wv_u", u3)
        self.register_parameter("Wq_v", v1)
        self.register_parameter("Wk_v", v2)
        self.register_parameter("Wv_v", v3)
        self.dict={0:'Wq',1:'Wk',2:'Wv'}
        
    def calculate_per_head(self,W,i):
        W=W.reshape(W.shape[0],self.num_heads,-1)#[384,6,64]
        u = getattr(self, self.dict[i]+"_u")#[6,384]
        v = getattr(self, self.dict[i]+"_v")#[6,64]
        Sigma=0
        for h in range(self.num_heads):
            with torch.no_grad():
                w=W[:,h,:]#[384,64]
                v[h]=l2normalize(torch.matmul(w.T,u[h]))
                u[h]=l2normalize(torch.matmul(w,v[h]))
        for h in range(self.num_heads):
            sigma=torch.matmul(torch.matmul(u[h],W[:,h,:]),v[h])
            Sigma+=sigma**2
        return Sigma
            
    def calculate(self):
        W=list(self.qk.weight.chunk(2,dim=0))
        W.append(self.v.weight)
        spe_loss=0
        for i in range(3):
            if self.pen_for_qkv[i]:
                sigma=self.calculate_per_head(W[i],i)
                spe_loss+=self.pen_for_qkv[i]*sigma
        return spe_loss
    def forward(self, x):
        B, N, C = x.shape
        if self.rel_indices is None or self.rel_indices.shape[1] != N:
            self.rel_indices = self.get_rel_indices(N)
        attn = self.get_attention(x)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        v = self.v_mask(v)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.training:
            spe_loss=self.calculate()
            return x,spe_loss
        else:
            return x

    def get_attention(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]
        pos_score = self.rel_indices.expand(B, -1, -1, -1)
        pos_score = self.pos_proj(pos_score).permute(0, 3, 1, 2)
        patch_score = (q @ k.transpose(-2, -1)) * self.scale
        patch_score = patch_score.softmax(dim=-1)
        pos_score = pos_score.softmax(dim=-1)

        gating = self.gating_param.view(1, -1, 1, 1)
        attn = (1. - torch.sigmoid(gating)) * patch_score + torch.sigmoid(gating) * pos_score
        attn /= attn.sum(dim=-1).unsqueeze(-1)
        attn = self.attn_drop(attn)
        return attn

    def get_attention_map(self, x, return_map=False):
        attn_map = self.get_attention(x).mean(0)  # average over batch
        distances = self.rel_indices.squeeze()[:, :, -1] ** .5
        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / distances.size(0)
        if return_map:
            return dist, attn_map
        else:
            return dist

    def local_init(self):
        self.v.weight.data.copy_(torch.eye(self.dim))
        locality_distance = 1  # max(1,1/locality_strength**.5)

        kernel_size = int(self.num_heads ** .5)
        center = (kernel_size - 1) / 2 if kernel_size % 2 == 0 else kernel_size // 2
        for h1 in range(kernel_size):
            for h2 in range(kernel_size):
                position = h1 + kernel_size * h2
                self.pos_proj.weight.data[position, 2] = -1
                self.pos_proj.weight.data[position, 1] = 2 * (h1 - center) * locality_distance
                self.pos_proj.weight.data[position, 0] = 2 * (h2 - center) * locality_distance
        self.pos_proj.weight.data *= self.locality_strength

    def get_rel_indices(self, num_patches: int) -> torch.Tensor:
        img_size = int(num_patches ** .5)
        rel_indices = torch.zeros(1, num_patches, num_patches, 3)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        rel_indices[:, :, :, 2] = indd.unsqueeze(0)
        rel_indices[:, :, :, 1] = indy.unsqueeze(0)
        rel_indices[:, :, :, 0] = indx.unsqueeze(0)
        device = self.qk.weight.device
        return rel_indices.to(device)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,pen_for_qkv=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.pen_for_qkv=pen_for_qkv
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.v_mask = nn.Identity()
    def init_uv(self):
        Wq,Wk,Wv=self.qkv.weight.chunk(3,dim=0)
        u1 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        u2 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        u3 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]).normal_(0, 1), requires_grad=False)
        
        v1 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        v2 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        v3 = torch.nn.Parameter(Wq.data.new(self.num_heads,Wq.shape[0]//self.num_heads).normal_(0, 1), requires_grad=False)
        
        u1.data = l2normalize(u1.data)
        u2.data = l2normalize(u2.data)
        u3.data = l2normalize(u3.data)
        v1.data = l2normalize(v1.data)
        v2.data = l2normalize(v2.data)
        v3.data = l2normalize(v3.data)

        self.register_parameter("Wq_u", u1)
        self.register_parameter("Wk_u", u2)
        self.register_parameter("Wv_u", u3)
        self.register_parameter("Wq_v", v1)
        self.register_parameter("Wk_v", v2)
        self.register_parameter("Wv_v", v3)
        self.dict={0:'Wq',1:'Wk',2:'Wv'}
        
    def calculate_per_head(self,W,i):
        names = locals()
        W=W.reshape(W.shape[0],self.num_heads,-1)#[384,6,64]
        u = getattr(self, self.dict[i] + "_u")
        v = getattr(self, self.dict[i] + "_v")
        Sigma=0
        # print(self.num_heads)
        for h in range(self.num_heads):
            # print(h)
            with torch.no_grad():
                w=W[:,h,:]
                # print(w._version)
                v[h]=l2normalize(torch.matmul(w.T,u[h]))
                u[h]=l2normalize(torch.matmul(w,v[h]))
                # print(v._version)
                # print(u._version)
                # print(v.shape,u.shape)
        for h in range(self.num_heads):
            sigma=torch.matmul(torch.matmul(u[h],W[:,h,:]),v[h])
            Sigma+=sigma**2

        return Sigma
            
    def calculate(self):
        W=self.qkv.weight.chunk(3,dim=0)
        spe_loss=0
        for i in range(3):
            if self.pen_for_qkv[i]:

                sigma=self.calculate_per_head(W[i],i)
                spe_loss+=self.pen_for_qkv[i]*sigma
        return spe_loss
    def get_attention_map(self, x, return_map=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_map = (q @ k.transpose(-2, -1)) * self.scale
        attn_map = attn_map.softmax(dim=-1).mean(0)

        img_size = int(N ** .5)
        ind = torch.arange(img_size).view(1, -1) - torch.arange(img_size).view(-1, 1)
        indx = ind.repeat(img_size, img_size)
        indy = ind.repeat_interleave(img_size, dim=0).repeat_interleave(img_size, dim=1)
        indd = indx ** 2 + indy ** 2
        distances = indd ** .5
        distances = distances.to(x.device)

        dist = torch.einsum('nm,hnm->h', (distances, attn_map)) / N
        if return_map:
            return dist, attn_map
        else:
            return dist

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        v = self.v_mask(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.training:
            spe_loss=self.calculate()

            return x,spe_loss
        else:
            return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_gpsa=True, pen_for_qkv=None,**kwargs):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.use_gpsa = use_gpsa
        if self.use_gpsa:
            self.attn = GPSA(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,pen_for_qkv=pen_for_qkv, **kwargs)
        else:
            self.attn = MHSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,pen_for_qkv=pen_for_qkv)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.training:
            attn_x,spe_loss=self.attn(self.norm1(x))
        else:
            attn_x=self.attn(self.norm1(x))
        x = x + self.drop_path(attn_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.training:
            return x,spe_loss#这一层的奇异值loss
        else:
            return x


class ConViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, global_pool=None,
                 local_up_to_layer=3, locality_strength=1., use_pos_embed=True,pen_for_qkv=None,args=None):
        super().__init__()
        embed_dim *= num_heads
        self.num_classes = num_classes
        self.local_up_to_layer = local_up_to_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.locality_strength = locality_strength
        self.use_pos_embed = use_pos_embed

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        if self.use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=True,
                locality_strength=locality_strength,pen_for_qkv=pen_for_qkv)
            if i < local_up_to_layer else
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_gpsa=False,pen_for_qkv=pen_for_qkv)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        for n, m in self.named_modules():
            if hasattr(m, 'local_init'):
                m.local_init()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def init_sn(self,seed):
        import random
        import numpy as np
        import torch.backends.cudnn as cudnn
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        for blk in self.blocks:
            blk.attn.init_uv()
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.use_pos_embed:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        if self.training:
            Spe_loss=0
            for u, blk in enumerate(self.blocks):
                if u == self.local_up_to_layer:
                    x = torch.cat((cls_tokens, x), dim=1)
                x,spe_loss = blk(x)
                Spe_loss+=spe_loss
        else:
            for u, blk in enumerate(self.blocks):
                if u == self.local_up_to_layer:
                    x = torch.cat((cls_tokens, x), dim=1)
                x = blk(x)

        x = self.norm(x)
        if self.training:
            return x[:, 0],Spe_loss
        else:
            return x[:, 0]

    def forward(self, x):
        if self.training:
            x,spe_loss = self.forward_features(x)
            x = self.head(x)
            return x,spe_loss
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x

def resize_pos_embed(posemb, posemb_new):
    gs_old = int(math.sqrt(posemb.shape[1]))
    gs_new = int(math.sqrt(posemb_new.shape[1]))
    posemb_grid = posemb.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    return posemb_grid

def checkpoint_filter_fn(state_dict, model, args):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if k == 'pos_embed' and v.shape != model.pos_embed.shape:
            v = resize_pos_embed(v, model.pos_embed)
        elif k == 'patch_embed.proj.weight' and v.shape != model.patch_embed.proj.weight.shape:
            if v.shape[2]%args['patch_size'] ==0:
                v = v.reshape(*v.shape[:2],
                    args['patch_size'], v.shape[2]//args['patch_size'],
                    args['patch_size'], v.shape[3]//args['patch_size']).sum(dim=[3,5])
        out_dict[k] = v
    return out_dict

def _create_convit(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    
    return build_model_with_cfg(
        ConViT, variant, pretrained,
        pretrained_cfg=default_cfgs[variant],
        pretrained_filter_fn=partial(checkpoint_filter_fn, args=kwargs), #
        **kwargs)



def convit_tiny_sn(pretrained=False, patch_size=16,args=None,**kwargs):
    model_args = dict(
        patch_size=patch_size,
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_tiny_sn', pretrained=pretrained, args=args, **model_args)
    return model



def convit_small_sn(pretrained=False, patch_size=16, args=None,**kwargs):
    model_args = dict(
        patch_size=patch_size,
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,#pen_for_qkv=pen_for_qkv,
        num_heads=9, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_small_sn', pretrained=pretrained, args=args,**model_args)
    return model



def convit_base_sn(pretrained=False,  patch_size=16, args=None,**kwargs):
    model_args = dict(
        patch_size = patch_size,
        local_up_to_layer=10, locality_strength=1.0, embed_dim=48,
        num_heads=16, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model = _create_convit(variant='convit_base_sn', pretrained=pretrained, args = args, **model_args)
    return model