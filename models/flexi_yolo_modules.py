"""
Flexi-YOLO Core Modules & Loss Functions
Reference: "Flexi-YOLO: A lightweight method for road crack detection in complex environments"
PLOS ONE (2025) - https://doi.org/10.1371/journal.pone.0325993

Modules:
1. AKConv: Alterable Kernel Convolution with arbitrary kernel size & coordinate offsets
2. GAMAttention: Global Attention Mechanism (3D Channel Attention + 7x7 Spatial Attention)
3. GhostConv & GHead: Lightweight Ghost convolution & coupled detection head
4. DCNv_C2f: C2f module with Deformable Convolution kernels
5. WiseIoULoss: Dynamic non-monotonic focusing loss function
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. AKConv: Alterable Kernel Convolution
# =====================================================================
class AKConv(nn.Module):
    """
    AKConv (Alterable Kernel Convolution)
    Dynamically adjusts kernel size and coordinates to accommodate 
    irregular, elongated road crack patterns.
    """
    def __init__(self, inc: int, outc: int, num_param: int = 9, stride: int = 1):
        super().__init__()
        self.inc = inc
        self.outc = outc
        self.num_param = num_param
        self.stride = stride

        # Offset convolution: predicts (dx, dy) for each sampling point
        self.offset_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        # Feature projection after adaptive sampling
        self.proj_conv = nn.Conv2d(inc * num_param, outc, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.act = nn.SiLU()

        # Initial sampling point distribution (optimized for elongated cracks)
        initial_offsets = []
        side = int(math.isqrt(num_param))
        if side * side == num_param:
            for r in range(side):
                for c in range(side):
                    initial_offsets.append([float(c - side // 2), float(r - side // 2)])
        else:
            for i in range(num_param):
                # Elongated vertical extension for crack detection
                initial_offsets.append([0.0, float(i - num_param // 2)])
        self.register_buffer("base_offsets", torch.tensor(initial_offsets, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        offsets = self.offset_conv(x)  # (B, 2*num_param, H_out, W_out)
        H_out, W_out = offsets.shape[2], offsets.shape[3]

        offsets = offsets.view(B, self.num_param, 2, H_out, W_out)

        # Base normalized grid coordinates [-1, 1]
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, H_out, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, W_out, device=x.device, dtype=x.dtype),
            indexing="ij"
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        scale_x = 2.0 / max(1, W - 1)
        scale_y = 2.0 / max(1, H - 1)

        sampled_features = []
        for i in range(self.num_param):
            base_off = self.base_offsets[i]
            off_i = offsets[:, i].permute(0, 2, 3, 1)  # (B, H_out, W_out, 2)

            sample_grid = base_grid.clone()
            sample_grid[..., 0] += (base_off[0] + off_i[..., 0]) * scale_x
            sample_grid[..., 1] += (base_off[1] + off_i[..., 1]) * scale_y

            sampled = F.grid_sample(
                x, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
            )
            sampled_features.append(sampled)

        cat_features = torch.cat(sampled_features, dim=1)  # (B, C * num_param, H_out, W_out)
        out = self.act(self.bn(self.proj_conv(cat_features)))
        return out


# =====================================================================
# 2. GAM: Global Attention Module
# =====================================================================
class GAMAttention(nn.Module):
    """
    Global Attention Mechanism (GAM)
    Integrates spatial and channel dependencies to preserve fine crack details 
    and suppress noisy road backgrounds.
    """
    def __init__(self, c1: int, c2: Optional[int] = None, rate: int = 4):
        super().__init__()
        in_channels = c1
        mid_channels = max(4, in_channels // rate)

        # Channel Attention Submodule (3D cross-dimensional interaction)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels)
        )

        # Spatial Attention Submodule (Two 7x7 convolutions)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        x_perm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_ch = self.channel_attention(x_perm).permute(0, 3, 1, 2)
        x_ch = torch.sigmoid(x_ch) * x

        # Spatial Attention
        x_sp = torch.sigmoid(self.spatial_attention(x_ch)) * x_ch
        return x_sp


# =====================================================================
# 3. GhostConv & Ghost Bottleneck
# =====================================================================
class GhostConv(nn.Module):
    """
    GhostConv: Generates more features using cheap linear operations 
    to reduce parameters and computational redundancy.
    """
    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, g: int = 1, act: bool = True):
        super().__init__()
        c_ = c2 // 2
        self.primary_conv = nn.Sequential(
            nn.Conv2d(c1, c_, k, s, k // 2, 1, groups=g, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU() if act else nn.Identity()
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(c_, c2 - c_, 3, 1, 1, groups=c_, bias=False),
            nn.BatchNorm2d(c2 - c_),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck module for lightweight representation."""
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        c_ = c2 // 2
        self.conv1 = GhostConv(c1, c_, 1, 1)
        self.conv2 = GhostConv(c_, c2, 1, 1, act=False)
        self.shortcut = nn.Sequential() if (c1 == c2 and s == 1) else nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, bias=False),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shortcut(x) + self.conv2(self.conv1(x))


# =====================================================================
# 4. DCNv-C2f: Deformable Convolution C2f Module
# =====================================================================
class DCNv_Bottleneck(nn.Module):
    """
    Bottleneck using AKConv / Deformable Convolution 
    to handle non-linear curved crack shapes.
    """
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        # Deformable convolution via AKConv
        self.cv2 = AKConv(c_, c2, num_param=9, stride=1)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class DCNv_C2f(nn.Module):
    """
    DCNv-C2f Module
    Replaces standard YOLOv8 C2f with deformable kernels for richer gradient 
    flow and superior crack boundary capturing.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, 2 * self.c, 1, 1, bias=False),
            nn.BatchNorm2d(2 * self.c),
            nn.SiLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d((2 + n) * self.c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        self.m = nn.ModuleList(
            DCNv_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# =====================================================================
# 5. G-Head: Lightweight Ghost Coupled Detection Head
# =====================================================================
class GHead(nn.Module):
    """
    G-Head (Ghost-Head)
    Coupled detection head sharing parameters for bounding box regression 
    and classification using GhostConv to eliminate redundant computations.
    """
    def __init__(self, nc: int = 3, ch: Tuple[int, ...] = (64, 128, 256)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers (P3, P4, P5)
        self.reg_max = 16  # DFL channels
        self.no = nc + self.reg_max * 4  # outputs per anchor

        # Shared Ghost heads for each scale
        self.ghost_layers = nn.ModuleList([
            nn.Sequential(
                GhostConv(c, c, 3, 1),
                GhostConv(c, c, 3, 1),
                nn.Conv2d(c, self.no, 1)
            ) for c in ch
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.ghost_layers[i](feats[i]) for i in range(self.nl)]


# =====================================================================
# 6. Wise-IoU Loss (WIoU v1 & WIoU v3)
# =====================================================================
class WiseIoU:
    """
    Wise-IoU (WIoU) Loss Function
    Mitigates low-quality annotation noise using outlier degree beta and 
    dynamic non-monotonic focusing coefficient r.
    
    Formula (Equation 7 & 8 in Paper):
      R_WIoU = exp( ((x - x_gt)^2 + (y - y_gt)^2) / ((Wg^2 + Hg^2)*) )
      L_WIoU_v1 = R_WIoU * (1 - IoU)
      beta = L_IoU* / E(L_IoU)
      r = beta / (delta * alpha^(beta - delta))
      L_WIoU_v3 = r * L_WIoU_v1
    """
    def __init__(self, alpha: float = 1.9, delta: float = 3.0):
        self.alpha = alpha
        self.delta = delta
        self.iou_mean = 1.0
        self.momentum = 0.9

    def bbox_iou(self, box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates IoU and minimum enclosing box dimensions.
        Boxes format: (x1, y1, x2, y2)
        """
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

        # Center coordinates
        b1_x, b1_y = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        b2_x, b2_y = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

        # Intersection
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        # Enclosing box
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps

        # Center distance squared
        rho2 = (b1_x - b2_x) ** 2 + (b1_y - b2_y) ** 2

        # R_WIoU penalty
        r_wiou = torch.exp(rho2 / c2.detach())

        return iou, r_wiou

    def compute_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor, v3: bool = True) -> torch.Tensor:
        """
        Computes Wise-IoU loss (v1 or v3 with dynamic non-monotonic focusing).
        """
        iou, r_wiou = self.bbox_iou(pred_boxes, target_boxes)
        loss_v1 = r_wiou * (1.0 - iou)

        if not v3:
            return loss_v1

        # Dynamic outlier factor beta
        loss_iou_detach = (1.0 - iou).detach()
        with torch.no_grad():
            self.iou_mean = self.momentum * self.iou_mean + (1 - self.momentum) * loss_iou_detach.mean().item()
            beta = (loss_iou_detach / max(1e-4, self.iou_mean)).clamp(0.01, 10.0)
            # Dynamic non-monotonic focusing coefficient r
            r = beta / (self.delta * (self.alpha ** (beta - self.delta)))

        loss_v3 = r * loss_v1
        return loss_v3


# =====================================================================
# Unit Test
# =====================================================================
if __name__ == "__main__":
    print("🚀 Đang kiểm thử các Module Flexi-YOLO...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Test AKConv
    ak = AKConv(32, 64, num_param=9).to(device)
    dummy_feat = torch.randn(2, 32, 28, 28, device=device)
    out_ak = ak(dummy_feat)
    assert out_ak.shape == (2, 64, 28, 28), f"AKConv shape mismatch: {out_ak.shape}"
    print("✅ AKConv hoạt động chính xác! Output shape:", out_ak.shape)

    # 2. Test GAM Attention
    gam = GAMAttention(64).to(device)
    out_gam = gam(out_ak)
    assert out_gam.shape == (2, 64, 28, 28), f"GAM shape mismatch: {out_gam.shape}"
    print("✅ GAM Attention hoạt động chính xác! Output shape:", out_gam.shape)

    # 3. Test GhostConv & DCNv-C2f
    ghost = GhostConv(64, 128).to(device)
    out_ghost = ghost(out_gam)
    assert out_ghost.shape == (2, 128, 28, 28), f"GhostConv shape mismatch: {out_ghost.shape}"
    print("✅ GhostConv hoạt động chính xác! Output shape:", out_ghost.shape)

    dcnv_c2f = DCNv_C2f(128, 128, n=1).to(device)
    out_dcnv = dcnv_c2f(out_ghost)
    assert out_dcnv.shape == (2, 128, 28, 28), f"DCNv-C2f shape mismatch: {out_dcnv.shape}"
    print("✅ DCNv-C2f hoạt động chính xác! Output shape:", out_dcnv.shape)

    # 4. Test GHead
    ghead = GHead(nc=3, ch=(64, 128, 256)).to(device)
    p3 = torch.randn(2, 64, 28, 28, device=device)
    p4 = torch.randn(2, 128, 14, 14, device=device)
    p5 = torch.randn(2, 256, 7, 7, device=device)
    out_ghead = ghead([p3, p4, p5])
    assert len(out_ghead) == 3
    print("✅ G-Head hoạt động chính xác! Số tầng phát hiện:", len(out_ghead))

    # 5. Test Wise-IoU Loss
    wiou = WiseIoU()
    b1 = torch.tensor([[10.0, 10.0, 50.0, 50.0], [20.0, 20.0, 80.0, 80.0]], device=device)
    b2 = torch.tensor([[12.0, 12.0, 48.0, 48.0], [15.0, 15.0, 85.0, 85.0]], device=device)
    loss = wiou.compute_loss(b1, b2, v3=True)
    print("✅ Wise-IoU Loss tính toán thành công! Loss:", loss.squeeze().tolist())
    print("🎉 Toàn bộ các module Flexi-YOLO đã sẵn sàng tích hợp!")
