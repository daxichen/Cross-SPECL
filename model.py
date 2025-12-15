import math
from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import random
from math import sqrt
from functools import partial, reduce
from operator import mul
from losses import PrototypePLoss, MultiDomainPrototypePLoss
from con_loss import SupConLoss

def get_optimizer(name, params, **kwargs):
        name = name.lower()
        optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
        optim_cls = optimizers[name]

        return optim_cls(params, **kwargs)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8,
                 mask_radio=0.1, mask_alpha=0.5,
                 noise_mode=1,
                 perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layer_flag=0, gauss_or_uniform=0, use_spectrum_noise=True):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

        self.mask_radio = mask_radio

        self.noise_mode = noise_mode
        self.noise_layer_flag = noise_layer_flag

        self.alpha = mask_alpha

        self.eps = 1e-6
        self.factor = uncertainty_factor
        self.p = perturb_prob
        self.gauss_or_uniform = gauss_or_uniform
        
        self.use_spectrum_noise = use_spectrum_noise

    def _reparameterize(self, mu, std, epsilon_norm):
        # epsilon = torch.randn_like(std) * self.factor
        epsilon = epsilon_norm * self.factor
        mu_t = mu + epsilon * std
        return mu_t

    def spectrum_noise(self, img_fft, ratio=1.0, noise_mode=1,
                       gauss_or_uniform=0):
        """Input image size: ndarray of [H, W, C]"""
        """noise_mode: 1 amplitude; 2: phase 3:both"""
        if random.random() > self.p:
            return img_fft
        batch_size, h, w, c = img_fft.shape

        img_abs, img_pha = torch.abs(img_fft), torch.angle(img_fft)

        # img_abs = torch.fft.fftshift(img_abs, dim=(1, 2))
        img_abs = torch.fft.fftshift(img_abs, dim=(1))

        h_crop = int(h * sqrt(ratio))
        w_crop = int(w * sqrt(ratio))
        h_start = h // 2 - h_crop // 2
        w_start = 0
        # w_start = w // 2 - w_crop // 2

        img_abs_ = img_abs.clone()
        if noise_mode != 0:
            miu_of_elem = torch.mean(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                        keepdim=True)
            var_of_elem = torch.var(img_abs_[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :], dim=0,
                                    keepdim=True)
            sig_of_elem = (var_of_elem + self.eps).sqrt()  # 1xHxWxC

            if gauss_or_uniform == 0:
                epsilon_sig = torch.randn_like(
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                gamma = epsilon_sig * sig_of_elem * self.factor
            elif gauss_or_uniform == 1:
                epsilon_sig = torch.rand_like(
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :]) * 2 - 1.  # U(-1,1)
                gamma = epsilon_sig * sig_of_elem * self.factor
            else:
                epsilon_sig = torch.randn_like(
                    img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :])  # BxHxWxC N(0,1)
                gamma = epsilon_sig * self.factor

            img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] = \
                img_abs[:, h_start:h_start + h_crop, w_start:w_start + w_crop, :] + gamma
        # img_abs = torch.fft.ifftshift(img_abs, dim=(1 , 2))  # recover
        img_abs = torch.fft.ifftshift(img_abs, dim=(1))  # recover
        img_mix = img_abs * (np.e ** (1j * img_pha))
        return img_mix

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        if self.training:
            if self.noise_mode != 0 and self.noise_layer_flag == 1 and self.use_spectrum_noise:
                x = self.spectrum_noise(x, ratio=self.mask_radio, noise_mode=self.noise_mode,
                                        gauss_or_uniform=self.gauss_or_uniform)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        # x = x.real
        x = x.reshape(B, N, C)
        return x
    
    
class BlockLayerScale(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, h=14, w=8, init_values=1e-5,
                 mask_radio=0.1, mask_alpha=0.5, noise_mode=1,
                 perturb_prob=0.5, uncertainty_factor=1.0,
                 layer_index=0, noise_layers=[0, 1, 2, 3], gauss_or_uniform=0,):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if layer_index in noise_layers:
            noise_layer_flag = 1
        else:
            noise_layer_flag = 0
        self.filter = GlobalFilter(dim, h=h, w=w,
                                   mask_radio=mask_radio,
                                   mask_alpha=mask_alpha,
                                   noise_mode=noise_mode,
                                   perturb_prob=perturb_prob,
                                   uncertainty_factor=uncertainty_factor,
                                   noise_layer_flag=noise_layer_flag, gauss_or_uniform=gauss_or_uniform, use_spectrum_noise=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        self.layer_index = layer_index  # where is the block in

    def forward(self, input):
        x = input
        x = x + self.drop_path(self.gamma * self.mlp(self.norm2(self.filter(x = self.norm1(x)))))
        return x
    def enable_spectrum_noise(self, enable=True):
        self.filter.use_spectrum_noise = enable  # 设置filter的use_spectrum_noise属性
    


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=8, patch_size=2, in_chans=48, embed_dim=128):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def init_weights(self, stop_grad_conv1=0):
        val = math.sqrt(6. / float(self.in_chans * reduce(mul, self.patch_size, 1) + self.embed_dim))
        nn.init.uniform_(self.proj.weight, -val, val)
        nn.init.zeros_(self.proj.bias)

        if stop_grad_conv1:
            self.proj.weight.requires_grad = False
            self.proj.bias.requires_grad = False

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # BxCxHxW -> BxNxC , N=(8/4)^2=4, C=embed_dim=128

        return x


class DownLayer(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=56, dim_in=64, dim_out=128):
        super().__init__()
        self.img_size = img_size
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.proj = nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
        self.num_patches = img_size * img_size // 4

    def forward(self, x):
        B, N, C = x.size()
        x = x.view(B, self.img_size, self.img_size, C).permute(0, 3, 1, 2)
        x = self.proj(x).permute(0, 2, 3, 1)
        x = x.reshape(B, -1, self.dim_out)

        return x


class GFNetPyramid(nn.Module):

    def __init__(self, img_size=8, band =48, patch_size=4,  in_chans=48, num_classes=7, embed_dim=[128, 256], depth=[2, 2], n_heads=8, d_head=32,
                 mlp_ratio=[4, 4],
                 drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001, no_layerscale=False, dropcls=0,
                 mask_radio=0.1, mask_alpha=0.5, noise_mode=1,  
                 perturb_prob=0.5,
                 uncertainty_factor=1.0,
                 noise_layers=[0, 1], gauss_or_uniform=0, ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.n_outputs = 256
        self.num_classes = num_classes
        self.band = band
        self.num_features = self.embed_dim = embed_dim[-1]  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        

        self.patch_embed = nn.ModuleList()

        patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        
        num_patches = patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim[0]))

        self.patch_embed.append(patch_embed)
        
        
        grid0 = img_size // patch_size           # stage0 token 高/宽
        grid1 = grid0 // 2                       # stage1 经过 DownLayer(stride=2) 后 token 高/宽
        sizes = [grid0, grid1]

        for i in range(1):
            patch_embed = DownLayer(sizes[i], embed_dim[i], embed_dim[i + 1])
            num_patches = patch_embed.num_patches
            self.patch_embed.append(patch_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        cur = 0
        for i in range(2):
            h = sizes[i]
            w = h // 2 + 1
            
            blk = nn.Sequential(*[
                    BlockLayerScale(
                        dim=embed_dim[i], mlp_ratio=mlp_ratio[i],
                        drop=drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer, h=h, w=w,
                        init_values=init_values,
                        mask_radio=mask_radio, mask_alpha=mask_alpha, noise_mode=noise_mode,
                        perturb_prob=perturb_prob,
                        uncertainty_factor=uncertainty_factor,
                        layer_index=i,
                        noise_layers=noise_layers, gauss_or_uniform=gauss_or_uniform,
                    )
                    for j in range(depth[i])
            ])
            self.blocks.append(blk)
            cur += depth[i]

        self.norm = norm_layer(embed_dim[-1])

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
            x_branch1 = x 
            x_branch2 = x

            for i in range(2):
                # Branch 1: Low-Frequency Transformation Branch 
                x_branch1 = self.patch_embed[i](x_branch1)
                if i == 0: x_branch1 = x_branch1 + self.pos_embed

                for blk in self.blocks[i]:
                    if isinstance(blk, BlockLayerScale):
                        blk.enable_spectrum_noise(True)
                    x_branch1 = blk(x_branch1)
                
                # Branch 2: Frequency Domain Filtering Branch 
                x_branch2 = self.patch_embed[i](x_branch2)
                if i == 0: x_branch2 = x_branch2 + self.pos_embed

                for blk in self.blocks[i]:
                    if isinstance(blk, BlockLayerScale):
                        blk.enable_spectrum_noise(False)
                    x_branch2 = blk(x_branch2)

            # Normalization and pooling
            x_aug = self.norm(x_branch1).mean(1)
            x = self.norm(x_branch2).mean(1)
        
            return x_aug, x

    def forward(self, x):
        x_aug, x = self.forward_features(x)
        return x_aug, x
    
    
class NotearsClassifier(nn.Module):
    def __init__(self, dims, num_classes):
        super(NotearsClassifier, self).__init__()
        self.dims = dims
        self.num_classes = num_classes
        self.weight_pos = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.weight_neg = nn.Parameter(torch.zeros(dims + 1, dims + 1))
        self.register_buffer("_I", torch.eye(dims + 1))
        self.register_buffer("_repeats", torch.ones(dims + 1).long())
        self._repeats[-1] *= num_classes

    def _adj(self):
        return self.weight_pos - self.weight_neg

    def _adj_sub(self):
        W = self._adj()
        return torch.matrix_exp(W * W)

    def h_func(self):
        W = self._adj()
        E = torch.matrix_exp(W * W)
        h = torch.trace(E) - self.dims - 1
        return h

    def w_l1_reg(self):
        reg = torch.mean(self.weight_pos + self.weight_neg)
        return reg

    def forward(self, x, y=None):
        W = self._adj()
        W_sub = self._adj_sub()
        if y is not None:
            x_aug = torch.cat((x, y.unsqueeze(1)), dim=1)
            M = x_aug @ W
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0)
            # reconstruct variables, classification logits
            return M[:, :self.dims], masked_x
        else:
            masked_x = x * W_sub[:self.dims, -1].unsqueeze(0).detach()
            return masked_x

    def mask_feature(self, x):
        W_sub = self._adj_sub()
        mask = W_sub[:self.dims, -1].unsqueeze(0).detach()
        return x * mask

    @torch.no_grad()
    def projection(self):
        self.weight_pos.data.clamp_(0, None)
        self.weight_neg.data.clamp_(0, None)
        self.weight_pos.data.fill_diagonal_(0)
        self.weight_neg.data.fill_diagonal_(0)

    @torch.no_grad()
    def masked_ratio(self):
        W = self._adj()
        return torch.norm(W[:self.dims, -1], p=0)
    
class DSPLTnet(nn.Module):

    def __init__(self, model, NotearsClassifier, num_classes, hparams):
        super(DSPLTnet, self).__init__()
        self.num_classes = num_classes
        # self.num_domains = num_domains
        self.hparams = hparams
        self.featurizer = model
        self.dag_mlp = NotearsClassifier(hparams['out_dim'], num_classes)
        self.dag_mlp.weight_pos.data[:-1, -1].fill_(1.0)

        self.inv_classifier = nn.Linear(hparams['out_dim'], num_classes)
        self.rec_classifier = nn.Linear(hparams['out_dim'], num_classes)
        
        self.network = nn.Sequential(self.featurizer, self.dag_mlp, self.inv_classifier)
        
        self.proto_m = self.hparams["ema_ratio"]
        self.lambda1 = self.hparams["lambda1"]
        self.lambda2 = self.hparams["lambda2"]
        self.rho_max = self.hparams["rho_max"]
        self.alpha = self.hparams["alpha"]
        self.rho = self.hparams["rho"]
        self._h_val = np.inf
        
        self.register_buffer(
            "prototypes_y",
            torch.zeros(num_classes, hparams['out_dim']))
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, hparams['out_dim']))
        self.register_buffer(
            "prototypes_label",
            torch.arange(num_classes))

        params = [
            {"params": self.network.parameters()},
            {"params": self.rec_classifier.parameters()},
        ]
        
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        
        self.loss_proto_con = PrototypePLoss(num_classes, hparams['temperature'])
        self.supcon = SupConLoss(temperature = hparams['temperature'], base_temperature = hparams['temperature'], device=hparams['gpu'])
        

    def update(self, x, y, **kwargs):
        
        f_aug, f = self.featurizer(x)
        
        masked_f_aug = self.dag_mlp(f_aug)
        masked_f = self.dag_mlp(f)
        
        normalized_f_aug = F.normalize(f_aug, dim=1)
        normalized_f = F.normalize(f, dim=1)
        z = torch.cat([normalized_f.unsqueeze(1), normalized_f_aug.unsqueeze(1)], dim=1)
        
        normalized_masked_f_aug = F.normalize(masked_f_aug, dim=1)
        
        for single_f, single_masked_f, label_y in zip(normalized_f, 
                                                 normalized_masked_f_aug, 
                                                 y):
            self.prototypes[label_y] = self.prototypes[label_y] * self.proto_m + (1 - self.proto_m) * single_f.detach()
            self.prototypes_y[label_y] = self.prototypes_y[label_y] * self.proto_m + (1 - self.proto_m) * single_masked_f.detach()
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        self.prototypes_y = F.normalize(self.prototypes_y, p=2, dim=1)

        prototypes = self.prototypes.detach().clone()
        prototypes_y = self.prototypes_y.detach().clone()
        
        proto_rec, masked_proto = self.dag_mlp(
            x=prototypes.view(self.num_classes, -1),
            y=self.prototypes_label)

        # reconstruction loss
        loss_rec = F.cosine_embedding_loss(
            proto_rec, 
            prototypes.view(self.num_classes, -1),
            torch.ones(self.num_classes, device=x.device))
        loss_rec += F.cross_entropy(
            self.rec_classifier(masked_proto),
            self.prototypes_label)
        
        loss_rec = self.lambda2 * loss_rec
        h_val = self.dag_mlp.h_func()
        penalty = 0.5 * self.rho * h_val * h_val + self.alpha * h_val
        l1_reg = self.lambda1 * self.dag_mlp.w_l1_reg()
        
        # update the DAG hyper-parameters
        if kwargs['step'] % 100 == 0:
            if self.rho < self.rho_max and h_val > 0.25 * self._h_val:
                self.rho *= 10
                self.alpha += self.rho * h_val.item()
            self._h_val = h_val.item()

        loss_dag = loss_rec + penalty + l1_reg

        loss_inv_ce = F.cross_entropy(self.inv_classifier(masked_f), y) + F.cross_entropy(self.inv_classifier(masked_f_aug), y)

        loss_contr = self.hparams["weight_con"] * (self.loss_proto_con(masked_f_aug, prototypes_y, y) + self.supcon(z, y) + self.loss_proto_con(f, prototypes, y))
        
        if kwargs['step'] == self.hparams["dag_anneal_steps"]:
            # avoid the gradient jump
            params = [
                {"params": self.network.parameters()},
            ]
            self.optimizer = get_optimizer(
                self.hparams["optimizer"],
                params,
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )

        if kwargs['step'] >= self.hparams["dag_anneal_steps"]:
            loss = loss_inv_ce + self.hparams["weight_dag"]*loss_dag + loss_contr
        else:
            loss = loss_inv_ce + self.hparams["weight_con"] * (self.supcon(z, y) + self.loss_proto_con(f, prototypes, y))
        
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # constraint DAG weights
        self.dag_mlp.projection()
        
        
        return {"loss": loss.item(),
                "inv_ce": loss_inv_ce.item(),
                "l2": loss_rec.item(),
                "penalty": penalty.item(),
                "l1": l1_reg.item(),
                "cl": loss_contr.item()}
        
        # return {"loss": loss.item()}


    def predict(self, x):
        f_aug, f = self.featurizer(x)
        masked_f = self.dag_mlp(f)
        return self.inv_classifier(masked_f)

    def forward(self, x):
        return self.predict(x)
    
    def new_optimizer(self, parameters):
        optimizer = get_optimizer(
            self.hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return optimizer

    def clone(self):
        clone = copy.deepcopy(self)
        params = [
            {"params": clone.network.parameters()},
        ]
        clone.optimizer = self.new_optimizer(params)
        clone.optimizer.load_state_dict(self.optimizer.state_dict())

        return clone