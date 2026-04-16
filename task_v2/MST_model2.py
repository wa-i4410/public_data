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
import torchvision.models as models


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

# ★追加: ガウシアンヒートマップ生成関数
def generate_gaussian_heatmap(img_height, img_width, center_x, center_y, sigma=5.0):
    x = np.arange(0, img_width, 1, np.float32)
    y = np.arange(0, img_height, 1, np.float32)
    y = y[:, np.newaxis]
    heatmap = np.exp(- ((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma ** 2))
    return heatmap



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

        # --- ★追加: 中間点(N-2個)の正解ヒートマップを生成 ---
        num_intermediate = self.num_control_points - 2
        gt_heatmaps = np.zeros((num_intermediate, img_height, img_width), dtype=np.float32)
        
        # 100個のGT点群から、両端を除く中間部分から等間隔にインデックスを抽出
        # (例: N=10 なら、中間8個の点を抽出)
        indices = np.linspace(1, len(gt_points) - 2, num_intermediate, dtype=int)
        
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

        # ★修正: 戻り値に gt_heatmaps_tensor を追加 (計6個)
        return input_tensor, growth_point_tensor, candidate_point_tensor, gt_pointcloud_tensor, gt_heatmaps_tensor, basename

# SoftArgmax2D, DifferentiableCatmullRomSpline は変更なし

    

# ==========================================
# 1. 位置エンコーディング (Positional Encoding)
# ==========================================
class PositionEmbeddingSine2D(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, h, w = x.shape
        mask = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos # shape: (B, 256, H, W)

# ==========================================
# 2. 座標出力用 スプライン曲線生成モジュール
# ==========================================
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
                c0.unsqueeze(1) + c1.unsqueeze(1) * t + 
                c2.unsqueeze(1) * t2 + c3.unsqueeze(1) * t3
            )
            curve_points.append(segment_curve)
        return torch.cat(curve_points, dim=1)

# ==========================================
# 3. メインモデル：CNN + Transformer
# ==========================================
class StemTransformerModel(nn.Module):
    def __init__(self, num_queries=8, hidden_dim=256, nheads=8, num_encoder_layers=4, num_decoder_layers=4):
        super().__init__()
        
        # --- (A) CNN バックボーン (ResNet18を使用) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # 入力を4チャネル(RGB+ヒートマップ)に変更
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 元のRGBの重みをコピーし、4チャネル目はゼロ初期化(最初は無視させる)
        with torch.no_grad():
            self.conv1.weight[:, :3] = original_conv1.weight
            self.conv1.weight[:, 3] = torch.zeros_like(original_conv1.weight[:, 0])
        
        # ResNetの特徴抽出レイヤー群 (Layer 4まで使用して特徴マップを取得)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # ResNet出力(512次元)をTransformerの次元(256)に変換
        self.input_proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        
        # --- (B) Transformer モジュール ---
        self.pos_embed = PositionEmbeddingSine2D(hidden_dim // 2)
        # 8個の点を予測するためのハコ (Learnable Queries)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim, 
            nhead=nheads, 
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True # 入力を (Batch, Seq, Feature) で扱う
        )
        
        # --- (C) 座標出力ヘッド ---
        # Transformerの出力を [x, y] 座標に変換
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )
        
        self.curve_gen = DifferentiableCatmullRomSpline(output_points=100)

    def forward(self, x, g_pts, c_pts):
        """
        x: [B, 4, H, W] (RGB + Heatmap)
        g_pts: [B, 2] 始点
        c_pts: [B, 2] 終点
        """
        B = x.size(0)
        
        # 1. CNNによる特徴抽出
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h) # shape: (B, 512, H/32, W/32)
        
        # 次元変換
        src = self.input_proj(h) # shape: (B, 256, 15, 20) ※640x480の場合
        
        # 2. 位置エンコーディングの生成と加算
        pos = self.pos_embed(src) # shape: (B, 256, 15, 20)
        
        # Transformer用に平坦化 (Batch, Seq, Feature) に変換
        src_flatten = src.flatten(2).transpose(1, 2) # (B, 300, 256)
        pos_flatten = pos.flatten(2).transpose(1, 2) # (B, 300, 256)
        
        # Queryの準備 (Batchサイズに拡張)
        query_embeds = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, 8, 256)
        # Decoderへの入力 (最初はゼロで初期化し、Query Embeddingsを位置情報として足す)
        tgt = torch.zeros_like(query_embeds)
        
        # 3. Transformer
        # src: 画像特徴, tgt: ハコ
        # メモ: PyTorchのnn.Transformerは src に位置エンコーディングを直接足してはくれないので、自分で足す
        hs = self.transformer(
            src=src_flatten + pos_flatten, 
            tgt=tgt + query_embeds
        ) 
        # hs shape: (B, 8, 256)
        
        # 4. 座標予測
        # MLPに通して Sigmoid で [0, 1] に制限
        pred_intermediate_pts = torch.sigmoid(self.coord_head(hs)) # (B, 8, 2)
        
        # 5. 【ハード制約】両端の結合
        # 今まで通り、始点と終点の間に予測した8点を挟み込む
        g_pts_exp = g_pts.unsqueeze(1) # (B, 1, 2)
        c_pts_exp = c_pts.unsqueeze(1) # (B, 1, 2)
        control_points = torch.cat([g_pts_exp, pred_intermediate_pts, c_pts_exp], dim=1) # (B, 10, 2)
        
        # 6. スプライン曲線の生成
        curve_100 = self.curve_gen(control_points)
        
        # Transformerの場合、ヒートマップMSEは不要になるので、今回はNoneや0を返すようにします
        return curve_100, control_points, None