import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp
from timm.models.vision_transformer import Attention, Block


__all__ = ['B4Classifier']


HIDDEN_num = 7
BM_num_classes = 1

class C025Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x48x48x48 -> 1x48x48x48
        self.conv = nn.Conv3d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, c025_img):
        """
        c025_img: (N, 3, D, H, W)
        mask: (N, 1, D, H, W)
        """
        c025_img = self.conv(c025_img) # (N, 1, D, H, W)
        return c025_img

    def init_weights(self):
        """
        Initializes weights of the model.
        """
        
        # Initialize weights of the mlp and linear layers with xavier uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class B4Classifier(nn.Module):
    def __init__(self, x_dim, mlp_dim, patch_size):
        super().__init__()
        self.x_dim = x_dim

        self.mask_conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        weight = torch.ones((1, 1, patch_size, patch_size, patch_size)).float() / (patch_size ** 3)
        self.mask_conv3d.weight = nn.Parameter(weight, requires_grad=False)

        self.mask_max3d = nn.MaxPool3d(kernel_size=patch_size, stride=patch_size, padding=0)

        self.mlp = self._build_mlp(num_layers=2, input_dim=x_dim, mlp_dim=mlp_dim, output_dim=HIDDEN_num)
        self.bmbi_head = nn.Linear(HIDDEN_num, BM_num_classes)
        self.bmbi_head2 = nn.Linear(HIDDEN_num, BM_num_classes)
        self.init_weights()

    def forward2(self, x):
        x, mask = x
        return self.forward(x, mask)

    def forward(self, x, mask):
        """
        x: (N, L+1, C), where L = D * H * W, C = x_dim
        mask: (N, 1, OD, OH, OW), where OD = D * patch_size, OH = H * patch_size, OW = W * patch_size
        """
        N, L1, C = x.shape
        x = x.view(N * L1, C)
        x = self.mlp(x) # (N, L+1, C) -> (N, L+1, 7)
        x = x.view(N, L1, -1)

        og_mask = mask.clone()

        mask_pool = self.mask_max3d(og_mask) 
        roi_mask = mask_pool.flatten(1).unsqueeze(-1) 

        bg_mask = 1 - roi_mask 

        x1 = self.bmbi_head(x[:, 0]) # (N, 1)
        x2 = self.bmbi_head2(x[:, 1:]) # (N, L, C)
        x2 = x2 * roi_mask # (N, L, 1)
        x2 = x2.sum(dim=1) / roi_mask.sum(dim=1) # (N, 1)
        roi_bm_logits = (x1 + x2) / 2 # (N, 1)

        x3 = x[:, 0] # (N, 7)
        x4 = x[:, 1:] * roi_mask # (N, L, 7)
        x4 = x4.sum(dim=1) / roi_mask.sum(dim=1) # (N, 7)
        roi_bi_logits = (x3 + x4) / 2 # (N, 7)

        x1 = self.bmbi_head(x[:, 0]) # (N, 1)
        x2 = self.bmbi_head2(x[:, 1:]) # (N, L)
        x2 = x2 * bg_mask # (N, L, 1)
        x2 = x2.sum(dim=1) / bg_mask.sum(dim=1) # (N, 1)
        bg_bm_logits = (x1 + x2) / 2 # (N, 1)

        x3 = x[:, 0] # (N, 7)
        x4 = x[:, 1:] * bg_mask # (N, L, 7)
        x4 = x4.sum(dim=1) / bg_mask.sum(dim=1) # (N, 7)
        bg_bi_logits = (x3 + x4) / 2 # (N, 7)
        if torch.isnan(roi_bm_logits).any() or torch.isnan(roi_bi_logits).any() or torch.isnan(bg_bm_logits).any() or torch.isnan(bg_bi_logits).any():
            import pdb; pdb.set_trace()

        return {
            "bm_logits": roi_bm_logits,
            "bi_logits": roi_bi_logits,
            "bg_bm_logits": bg_bm_logits,
            "bg_bi_logits": bg_bi_logits,
        }

    def forward_without_PLFB(self, x, mask):
        """
        x: (N, L+1, C), where L = D * H * W, C = x_dim
        mask: (N, 1, OD, OH, OW), where OD = D * patch_size, OH = H * patch_size, OW = W * patch_size
        """
        N, L1, C = x.shape
        x = x.view(N * L1, C)
        x = self.mlp(x) # (N, L+1, C) -> (N, L+1, 7)
        x = x.view(N, L1, -1)

        x1 = self.bmbi_head(x[:, 0]) # (N, 1)
        x2 = self.bmbi_head2(x[:, 1:]) # (N, L, C)
        x2 = x2.mean(dim=1) # (N, 1)
        roi_bm_logits = (x1 + x2) / 2 # (N, 1)

        x3 = x[:, 0] # (N, 7)
        x4 = x[:, 1:].mean(dim=1) # (N, 7)
        roi_bi_logits = (x3 + x4) / 2 # (N, 7)

        return {
            "bm_logits": roi_bm_logits,
            "bi_logits": roi_bi_logits,
        } 

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True, dropout_rate=0.0):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
                mlp.append(nn.Dropout(dropout_rate))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def init_weights(self):
        """
        Initializes weights of the model.
        """
        
        # Initialize weights of the mlp and linear layers with xavier uniform initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize the parameters of the LayerNorm layers
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
