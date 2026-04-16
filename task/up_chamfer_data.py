import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import re
import csv
import torchvision.transforms as T
import shutil # split_dataで必要
from torchvision import models
import math


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

class PointCloudDataset(Dataset):
    def __init__(self, pt_path, color_path, map_path, gt_pointcloud_path,augment=False):
        self.pt_path = pt_path 
        self.color_path = color_path
        self.map_path = map_path
        self.gt_pointcloud_path = gt_pointcloud_path
        self.pt_baselist = sorted([f for f in os.listdir(pt_path) if f.endswith('.txt')], key=natural_key)
        self.augment = augment
        
    def __len__(self): 
        return len(self.pt_baselist)

    def __getitem__(self, idx): 
        basename = self.pt_baselist[idx]
        # ★ 修正点: self.pt_path を使う
        pt_file_path = os.path.join(self.pt_path, basename) 
        color_image_path = os.path.join(self.color_path, basename.replace('_points.txt', '.png'))
        map_image_path = os.path.join(self.map_path, basename.replace('_points.txt', '.png'))
        gt_pc_file_path = os.path.join(self.gt_pointcloud_path, basename.replace('_points.txt', '_gt_pointcloud.csv'))
        
        if not all(os.path.exists(p) for p in [pt_file_path, color_image_path, gt_pc_file_path]):
            return None
        
        point_data = read_points(pt_file_path)
        start_point = point_data.get("growth")[0]
        candidate_point = point_data.get("15cm")[0]

        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size

        map_image_pil = Image.open(map_image_path).convert('L')

        gt_points = []
        with open(gt_pc_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: gt_points.append([float(row[0]), float(row[1])])

        if self.augment:
            # 1. カラージッター (カラー画像のみ)
            color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
            color_image_pil = color_jitter(color_image_pil)

            # 2. ランダム左右反転
            if random.random() > 0.5:
                color_image_pil = T.functional.hflip(color_image_pil)
                map_image_pil = T.functional.hflip(map_image_pil)
                start_point = (img_width - 1 - start_point[0], start_point[1])
                candidate_point = (img_width - 1 - candidate_point[0], candidate_point[1])
                gt_points = [[img_width - 1 - x, y] for x, y in gt_points]

            # 3. ランダム回転
            angle = random.uniform(-15, 15)
            color_image_pil = T.functional.rotate(color_image_pil, angle)
            map_image_pil = T.functional.rotate(map_image_pil, angle)
            angle_rad = -math.radians(angle)
            center = (img_width / 2, img_height / 2)
            start_point = rotate_point(start_point, angle_rad, center)
            candidate_point = rotate_point(candidate_point, angle_rad, center)
            gt_points = [list(rotate_point(p, angle_rad, center)) for p in gt_points]
        
        points_to_check = [start_point, candidate_point] + gt_points
        is_out_of_bounds = False
        for x, y in points_to_check:
            if not (0 <= x < img_width and 0 <= y < img_height):
                is_out_of_bounds = True
                break  # 1つでも範囲外ならチェックを終了
        
        # もし範囲外の点があれば、このデータを破棄してNoneを返す
        if is_out_of_bounds:
            return None
        
        # <--- データ拡張の処理ここまで ---


        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)
        plot_point_circle(heatmap, start_point[0], start_point[1], 5, 255)
        plot_point_circle(heatmap, candidate_point[0], candidate_point[1], 5, 128)
        heatmap_pil = Image.fromarray(heatmap, 'L')

        color_tensor = T.ToTensor()(color_image_pil)
        map_tensor = T.ToTensor()(map_image_pil)
        heatmap_tensor = T.ToTensor()(heatmap_pil)

        input_tensor = torch.cat([color_tensor, map_tensor, heatmap_tensor], dim=0)  # 5チャンネルにする
        growth_point_tensor = torch.tensor([start_point[0] / img_width, start_point[1] / img_height], dtype=torch.float32)
        candidate_point_tensor = torch.tensor([candidate_point[0] / img_width, candidate_point[1] / img_height], dtype=torch.float32)
        gt_pointcloud_tensor = torch.tensor(gt_points, dtype=torch.float32)


        return input_tensor, growth_point_tensor, candidate_point_tensor, gt_pointcloud_tensor, basename


# ===== 3. モデル & 損失関数 =====

class PositionChamferModel(nn.Module):
    """
    シャンファー距離を用いた位置予測モデル
    """
    def __init__(self, img_width=640, img_height=480, num_points=3):
        super(PositionChamferModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points

        # --- 畳み込み層に padding=1 を追加 ---
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.flattened_size = 256 * (img_width // 16) * (img_height // 16)

        # 全結合層
        self.fc1 = nn.Linear(self.flattened_size + 4, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, self.num_points * 2)

    def forward(self, x, growth_points, candidate_points):
        # (forwardパスの中身は、以前の正常なバージョンから変更ありません)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)
        
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        combined_features = torch.cat([x, coords_input], dim=1)
        
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        
        offsets = self.fc2(x).view(-1, self.num_points, 2) * 0.5
        
        start_points = growth_points.unsqueeze(1)
        end_points = candidate_points.unsqueeze(1)
        t = torch.linspace(0, 1, self.num_points, device=x.device).unsqueeze(0).unsqueeze(2)
        base_path = start_points + t * (end_points - start_points)
        
        pred_points_norm = base_path + offsets
        return pred_points_norm
    
class PositionChamferModel_ResNet(nn.Module):
    """
    ResNet18をベースにしてシャンファー距離を用いた位置予測モデル
    """
    def __init__(self, img_width=640, img_height=480, num_points=3, dropout_rate=0.5):
        super(PositionChamferModel_ResNet, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points

        # --- 1. 学習済みのResNet18をロード ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- 2. 入力層を4チャンネルに対応させる ---
        # 元の重みを保持しつつ、入力チャンネル数を変更する
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 元のRGBの重みをコピー
        self.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        # 4チャンネル目（深度）の重みは、RGBの平均で初期化
        self.conv1.weight.data[:, 3, :, :] = original_conv1.weight.data.mean(dim=1)
        # 5チャンネル目（ヒートマップ）の重みは、RGBの平均で初期化
        self.conv1.weight.data[:, 4, :, :] = original_conv1.weight.data.mean(dim=1)

        # --- 3. ResNetの残りの部分をバックボーンとして使用 ---
        # 最後の全結合層（分類器）は不要なので取り除く
        self.backbone = nn.Sequential(
            self.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool
        )
        
        # ResNet18の出力特徴量は512次元
        resnet_output_size = 512

        # --- 4. 全結合層（予測ヘッド）の定義 ---
        # CNNの出力特徴量に、座標データ4つ分 (始点x,y, 終点x,y) を加える
        self.fc1 = nn.Linear(resnet_output_size + 4, 1024)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, self.num_points * 2) # 10点の(x,y)で20次元

    def forward(self, x, growth_points, candidate_points):
        # 1. ResNetバックボーンで画像特徴量を抽出
        x = self.backbone(x)
        x = x.view(x.size(0), -1) # 平坦化

        # 2. 座標データを連結
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        combined_features = torch.cat([x, coords_input], dim=1)
        
        # 3. 全結合層で最終的なオフセットを予測
        x = F.relu(self.fc1(combined_features))
        x = self.dropout(x)
        
        # 4. 予測された点群を計算（この部分は以前のモデルと同じ）
        offsets = self.fc2(x).view(-1, self.num_points, 2) * 0.5
        
        start_points = growth_points.unsqueeze(1)
        end_points = candidate_points.unsqueeze(1)
        t = torch.linspace(0, 1, self.num_points, device=x.device).unsqueeze(0).unsqueeze(2)
        base_path = start_points + t * (end_points - start_points)
        
        pred_points_norm = base_path + offsets
        return pred_points_norm
    

    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels) # 入力は cat(skip, up)
        
    def forward(self, x1, x2):
        # x1: デコーダからの低解像度特徴
        # x2: エンコーダからのスキップ接続
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ===== ★★★ U-Netベースの新しい点群予測モデル ★★★ =====
class UNetPointCloudModel(nn.Module):
    """ U-Netをベースにした点群予測モデル """
    def __init__(self, img_width=640, img_height=480, num_points=5):
        super(UNetPointCloudModel, self).__init__()
        self.num_points = num_points

        # --- 1. U-Net バックボーン ---
        self.inc = ConvBlock(5, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024) # ボトムネック

        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        pooled_feature_size = 64 * 7 * 7

        
        self.prediction_head = nn.Sequential(
            nn.Linear(pooled_feature_size + 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_points * 2) # 最終的に10点*2座標 = 20個の数値を出力
        )

    def forward(self, x, growth_points, candidate_points):
        # --- U-Netによる特徴抽出 ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5) # ボトムネック

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) # 最終的な特徴マップ (shape: [batch, 64, 480, 640])
        
        x_pooled = self.avgpool(x) # (batch, 64, 7, 7)
        x_flat = x_pooled.view(x_pooled.size(0), -1) # (batch, 64*7*7)
        
        # 座標データを連結
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        combined_features = torch.cat([x_flat, coords_input], dim=1)
        
        # 最終的な3点の座標を予測
        pred_points_offsets = self.prediction_head(combined_features).view(-1, self.num_points, 2) * 0.5
        
        # 始点から終点への線形補間をベースにオフセットを加える
        start_points = growth_points.unsqueeze(1)
        end_points = candidate_points.unsqueeze(1)
        t = torch.linspace(0, 1, self.num_points, device=x.device).unsqueeze(0).unsqueeze(2)
        base_path = start_points + t * (end_points - start_points)
        
        pred_points_norm = base_path + pred_points_offsets
        
        return pred_points_norm
    
# ===== ★★★ U-Netベースの点群予測モデル (自由配置版) ★★★ =====
class UNetPointCloudModelFree(nn.Module):
    """ U-Netをベースにした点群予測モデル """
    def __init__(self, img_width=640, img_height=480, num_points=3):
        super(UNetPointCloudModelFree, self).__init__()
        self.num_points = num_points

        # --- 1. U-Net バックボーン (変更なし) ---
        self.inc = ConvBlock(5, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024) # ボトムネック

        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        pooled_feature_size = 64 * 7 * 7

        # --- 2. 予測ヘッド (変更なし) ---
        self.prediction_head = nn.Sequential(
            nn.Linear(pooled_feature_size + 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.num_points * 2)
        )

    def forward(self, x, growth_points, candidate_points):
        # --- U-Netによる特徴抽出 (変更なし) ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        x_pooled = self.avgpool(x)
        x_flat = x_pooled.view(x_pooled.size(0), -1)
        
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        combined_features = torch.cat([x_flat, coords_input], dim=1)
        
        # --- ★★★ ここからが変更箇所 ★★★ ---
        
        # 1. 予測ヘッドで直接、点の座標(raw値)を出力
        predicted_points_raw = self.prediction_head(combined_features)
        
        # 2. Sigmoid関数を適用し、出力を0.0〜1.0の範囲に正規化
        predicted_points_sigmoid = torch.sigmoid(predicted_points_raw)
        
        # 3. 最終的な点群の形状 (Batch, num_points, 2) に変形
        pred_points_norm = predicted_points_sigmoid.view(-1, self.num_points, 2)
        
        return pred_points_norm

class ConvBlock2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class UpConv2(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock2(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock2(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- ここからが改良版モデル ---
class UNetHeatmapModel(nn.Module):
    """ U-Netをベースにしたヒートマップ予測モデル """
    def __init__(self, in_channels=5, n_heatmaps=3, bilinear=True):
        """
        Args:
            in_channels (int): 入力画像のチャンネル数 (例: RGB+Depth+座標情報=5)
            n_heatmaps (int): 出力するヒートマップの数
        """
        super(UNetHeatmapModel, self).__init__()
        self.n_heatmaps = n_heatmaps

        # --- U-Net バックボーン (エンコーダー・デコーダー) ---
        self.inc = ConvBlock2(in_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock2(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock2(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock2(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock2(512, 1024 // factor) # ボトムネック

        self.up1 = UpConv2(1024, 512 // factor, bilinear)
        self.up2 = UpConv2(512, 256 // factor, bilinear)
        self.up3 = UpConv2(256, 128 // factor, bilinear)
        self.up4 = UpConv2(128, 64, bilinear)

        # --- 出力層: n_heatmapsチャンネルのヒートマップを出力 ---
        self.outc = nn.Conv2d(64, n_heatmaps, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x (Tensor): 入力テンソル (batch, in_channels, height, width)
        Returns:
            Tensor: 予測ヒートマップ (batch, n_heatmaps, height, width)
        """
        # --- U-Netによる特徴抽出 ---
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.conv1(x2)
        x3 = self.down2(x2)
        x3 = self.conv2(x3)
        x4 = self.down3(x3)
        x4 = self.conv3(x4)
        x5 = self.down4(x4)
        x5 = self.conv4(x5) # ボトムネック

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 出力層を通してヒートマップを生成
        heatmaps = self.outc(x)
        
        # Sigmoid関数で出力を0-1の範囲に正規化
        return torch.sigmoid(heatmaps)