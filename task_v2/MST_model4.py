import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset
import os
import re
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as T
import random
import csv
import cv2

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def read_points(file_path):
    points = {}
    with open(file_path, "r") as f:
        for line in f:
            try:
                label, x_str, y_str = line.strip().split(',')
                if label.lower() not in points: points[label.lower()] = []
                points[label.lower()].append((int(x_str), int(y_str)))
            except ValueError: continue
    return points

def plot_point_circle(img_array, x, y, radius, value):
    h, w = img_array.shape
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y-y)**2)
    img_array[dist <= radius] = value

def rotate_point(point, angle_rad, center):
    cx, cy = center
    x, y = point
    new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return int(new_x), int(new_y)

# ガウシアンヒートマップ生成関数
def generate_gaussian_heatmap(img_height, img_width, center_x, center_y, sigma=5.0):
    x = np.arange(0, img_width, 1, np.float32)
    y = np.arange(0, img_height, 1, np.float32)
    y = y[:, np.newaxis]
    heatmap = np.exp(- ((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap

class SoftArgmax2D(nn.Module):
    # beta=50.0 などの大きな値にして、ハードなピーク検出に近づける
    ### 実験7：beta=1.0 にしてみる ###
    ### 実験８：beta=10.0 にしてみる ###
    def __init__(self, normalize=True, beta= 50.0): 
        super().__init__()
        self.normalize = normalize
        self.beta = beta

    def forward(self, heatmaps):
        B, N, H, W = heatmaps.shape
        device = heatmaps.device
        
        heatmaps_flat = heatmaps.view(B, N, -1)
        
        prob_maps = F.softmax(heatmaps_flat * self.beta, dim=-1).view(B, N, H, W)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device), 
            torch.arange(W, dtype=torch.float32, device=device), 
            indexing='ij'
        )
        
        x_coords = torch.sum(prob_maps * x_grid.view(1, 1, H, W), dim=(2, 3))
        y_coords = torch.sum(prob_maps * y_grid.view(1, 1, H, W), dim=(2, 3))
        
        if self.normalize:
            x_coords = x_coords / (W - 1)
            y_coords = y_coords / (H - 1)
            
        coords = torch.stack([x_coords, y_coords], dim=-1)
        return coords

class DifferentiableCatmullRomSpline(nn.Module):
    def __init__(self, output_points=100):
        super().__init__()
        self.output_points = output_points

    def forward(self, points):
        B, N, _ = points.shape
        device = points.device
        
        p_first = points[:, 0:1, :]
        p_last = points[:, -1:, :]
        padded_points = torch.cat([p_first, points, p_last], dim=1)
        
        curve_points = []
        segments = N - 1
        points_per_segment = self.output_points // segments
        remainder = self.output_points % segments
        
        for i in range(segments):
            num_t = points_per_segment + (remainder if i == segments - 1 else 0)
            t = torch.linspace(0, 1, steps=num_t, device=device).view(-1, 1)
            
            P0 = padded_points[:, i]
            P1 = padded_points[:, i + 1]
            P2 = padded_points[:, i + 2]
            P3 = padded_points[:, i + 3]
            
            t2 = t ** 2
            t3 = t ** 3
            
            c0 = 2 * P1
            c1 = -P0 + P2
            c2 = 2 * P0 - 5 * P1 + 4 * P2 - P3
            c3 = -P0 + 3 * P1 - 3 * P2 + P3
            
            segment_curve = 0.5 * (
                c0.unsqueeze(1) + 
                c1.unsqueeze(1) * t + 
                c2.unsqueeze(1) * t2 + 
                c3.unsqueeze(1) * t3
            )
            curve_points.append(segment_curve)
            
        return torch.cat(curve_points, dim=1)

# =====================================================================
# 1. 論文アイデア①：チャネル＆空間アテンション (CBAM)
# 葉のテクスチャを無視し、茎の特徴だけを強調するモジュール
# =====================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x) # 茎らしいチャネルを強調（葉を無視）
        x = x * self.sa(x) # 茎がある空間位置を強調
        return x

# =====================================================================
# 2. 論文アイデア②：マルチスケール拡張畳み込み (Dilation Block)
# クリップや葉による遮蔽（オクルージョン）を飛び越えて茎を繋ぐモジュール
# =====================================================================
class MultiDilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        # 異なる受容野（視野の広さ）を持つ畳み込みを並列に配置
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=8, dilation=8)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 結合後にチャンネルを調整する1x1畳み込み
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d3 = self.conv3(x)
        d4 = self.conv4(x)
        out = torch.cat([d1, d2, d3, d4], dim=1) # 異なる視野の情報を結合
        out = self.bn(out)
        out = self.relu(out)
        return self.fuse(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)


class AdvancedStemUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck (★ ここに Dilation Block を配置して視野を極大化)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), MultiDilatedBlock(512, 1024))
        
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        # 出力層 (中間制御点の数だけのヒートマップを出力)
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck (広い視野でクリップの裏の茎を推論)
        x5 = self.down4(x4)
        
        # Decoder (スキップ接続される特徴マップをCBAMで浄化)
        x = self.up1(x5)
        x = torch.cat([x, self.cbam4(x4)], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, self.cbam3(x3)], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, self.cbam2(x2)], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, self.cbam1(x1)], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits

class TomatoStemDataset(Dataset):
    
    def __init__(self, pt_path, color_path, gt_pointcloud_path, num_control_points=10, augment=False, file_list=None):
        self.pt_path = pt_path 
        self.color_path = color_path
        self.gt_pointcloud_path = gt_pointcloud_path
        self.pt_baselist = sorted([f for f in os.listdir(pt_path) if f.endswith('.txt')], key=natural_key)
        self.augment = augment
        self.num_control_points = num_control_points # 保存
        
        if file_list:
            self.point_files = [os.path.join(self.pt_path, f) for f in file_list if os.path.exists(os.path.join(self.pt_path, f))]
        else:
            self.point_files = sorted([os.path.join(self.pt_path, f) for f in os.listdir(self.pt_path) if f.endswith('.txt')], key=natural_key)

    def __len__(self): 
        return len(self.pt_baselist)

    def __getitem__(self, idx): 
        basename = self.pt_baselist[idx]
        pt_file_path = os.path.join(self.pt_path, basename) 
        color_image_path = os.path.join(self.color_path, basename.replace('_points.txt', '.png'))
        gt_pc_file_path = os.path.join(self.gt_pointcloud_path, basename.replace('_points.txt', '_gt_pointcloud.csv'))
        
        if not all(os.path.exists(p) for p in [pt_file_path, color_image_path, gt_pc_file_path]):
            return None
        
        point_data = read_points(pt_file_path)
        start_point = point_data.get("growth")[0]
        candidate_point = point_data.get("15cm")[0]

        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size

        gt_points = []
        with open(gt_pc_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: gt_points.append([float(row[0]), float(row[1])])

        if self.augment:
            color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            color_image_pil = color_jitter(color_image_pil)

            if random.random() > 0.5:
                color_image_pil = T.functional.hflip(color_image_pil)
                start_point = (img_width - 1 - start_point[0], start_point[1])
                candidate_point = (img_width - 1 - candidate_point[0], candidate_point[1])
                gt_points = [[img_width - 1 - x, y] for x, y in gt_points]

            angle = random.uniform(-15, 15)
            color_image_pil = T.functional.rotate(color_image_pil, angle)
            angle_rad = -math.radians(angle)
            center = (img_width / 2, img_height / 2)
            start_point = rotate_point(start_point, angle_rad, center)
            candidate_point = rotate_point(candidate_point, angle_rad, center)
            gt_points = [list(rotate_point(p, angle_rad, center)) for p in gt_points]

            # 50%の確率で、画像に1〜3個の黒い四角形をランダムに描画する
            if random.random() > 0.5:
                draw = ImageDraw.Draw(color_image_pil)
                num_boxes = random.randint(1, 3) # 1〜3個のブロック
                for _ in range(num_boxes):
                    box_w = random.randint(50, 150)
                    box_h = random.randint(50, 150)
                    x1 = random.randint(0, max(1, img_width - box_w))
                    y1 = random.randint(0, max(1, img_height - box_h))
                    draw.rectangle([x1, y1, x1 + box_w, y1 + box_h], fill=(0, 0, 0))
            # ----------------------------------------------------
        
        points_to_check = [start_point, candidate_point] + gt_points
        is_out_of_bounds = False
        for x, y in points_to_check:
            if not (0 <= x < img_width and 0 <= y < img_height):
                is_out_of_bounds = True
                break
        
        if is_out_of_bounds:
            return None

        
        num_intermediate = self.num_control_points - 2
        gt_heatmaps = np.zeros((num_intermediate, img_height, img_width), dtype=np.float32)
        
        # 100個のGT点群から、両端を除く中間部分から等間隔にインデックスを抽出
        # (例: N=10 なら、中間8個の点を抽出)
        total_points = num_intermediate + 2
        step = (len(gt_points) - 1) / (total_points - 1)
        indices = np.round(np.arange(1, num_intermediate + 1) * step).astype(int)
        
        for i, pt_idx in enumerate(indices):
            pt = gt_points[pt_idx] # オーグメンテーション済みの [x, y]
            heatmap = generate_gaussian_heatmap(img_height, img_width, pt[0], pt[1], sigma=5.0)
            # 実験５：sigma=2.0 に変更してみる
            # heatmap = generate_gaussian_heatmap(img_height, img_width, pt[0], pt[1], sigma=2.0)
            gt_heatmaps[i] = heatmap
            
        gt_heatmaps_tensor = torch.from_numpy(gt_heatmaps) # (N-2, H, W)
        # ----------------------------------------------------
        
        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)


        plot_point_circle(heatmap, start_point[0], start_point[1], 5, 255)
        plot_point_circle(heatmap, candidate_point[0], candidate_point[1], 5, 128)
        heatmap_pil = Image.fromarray(heatmap, 'L')

        color_tensor = T.ToTensor()(color_image_pil)
        heatmap_tensor = T.ToTensor()(heatmap_pil)

        input_tensor = torch.cat([color_tensor, heatmap_tensor], dim=0)
        growth_point_tensor = torch.tensor([start_point[0] / (img_width - 1), start_point[1] / (img_height - 1)], dtype=torch.float32)
        candidate_point_tensor = torch.tensor([candidate_point[0] / (img_width - 1), candidate_point[1] / (img_height - 1)], dtype=torch.float32)
        gt_tensor = torch.tensor(gt_points, dtype=torch.float32)
        norm_factor = torch.tensor([img_width - 1, img_height - 1], dtype=torch.float32)
        gt_pointcloud_tensor = gt_tensor / norm_factor

        return input_tensor, growth_point_tensor, candidate_point_tensor, gt_pointcloud_tensor, gt_heatmaps_tensor, basename

class StemCurveModel(nn.Module):
    def __init__(self, unet_backbone, beta=50.0): 
        super().__init__()
        self.unet_backbone = unet_backbone
        self.soft_argmax = SoftArgmax2D(normalize=True, beta=beta) 
        self.curve_gen = DifferentiableCatmullRomSpline(output_points=100)

    def forward(self, x, g_pts, c_pts):
        # 1. U-Netが直接中間ヒートマップを出力する
        heatmaps = self.unet_backbone(x) # (B, N-2, H, W)
        
        # 2. 中間ヒートマップから中間座標を取得
        intermediate_points = self.soft_argmax(heatmaps) # (B, N-2, 2)
        
        # 3. 【ハード制約】結合
        g_pts_exp = g_pts.unsqueeze(1) 
        c_pts_exp = c_pts.unsqueeze(1) 
        control_points = torch.cat([g_pts_exp, intermediate_points, c_pts_exp], dim=1) # (B, N, 2)
        
        # 4. 100点の曲線を生成
        curve_100 = self.curve_gen(control_points)
        
        return curve_100, control_points, heatmaps
