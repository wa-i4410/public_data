import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import numpy as np
import torchvision.transforms as T
from PIL import Image
import re
import math
import random
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)]

def rotate_point(point, angle_rad, center):
    cx, cy = center
    x, y = point
    new_x = cx + (x - cx) * math.cos(angle_rad) - (y - cy) * math.sin(angle_rad)
    new_y = cy + (x - cx) * math.sin(angle_rad) + (y - cy) * math.cos(angle_rad)
    return int(new_x), int(new_y)

def read_points(file_path):
    points = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    label, x_str, y_str = line.split(',')
                    points.append((label, int(x_str), int(y_str)))
                except ValueError:
                    continue
    return points

def plot_point_circle(img_array, x, y, radius, value):
    # 描画範囲をクリップ
    h, w = img_array.shape
    center_x, center_y = int(x), int(y)

    # 円が収まる外接矩形の範囲を計算
    min_y = max(0, center_y - radius)
    max_y = min(h, center_y + radius + 1)
    min_x = max(0, center_x - radius)
    max_x = min(w, center_x + radius + 1)

    # 範囲内の各ピクセルが円内にあるかチェックして値を設定
    for r in range(min_y, max_y):
        for c in range(min_x, max_x):
            # 中心からの距離が半径以内かチェック
            if (c - center_x)**2 + (r - center_y)**2 <= radius**2:
                img_array[r, c] = value

def combine_image_with_heatmap(color_image_path, heatmap_array):
    """
    640x480x3のカラー画像とNumPy配列のヒートマップを結合し、
    640x480x4のRGBA形式の画像を生成します。

    Args:
        color_image_path (str): カラー画像ファイルのパス。
        heatmap_array (np.ndarray): NumPy配列形式のヒートマップ (0-255の1チャンネル)。
        output_path (str): 出力する結合済み画像ファイルのパス。
    """
    try:
        # 1. カラー画像を読み込む (RGB形式に変換)
        color_image = Image.open(color_image_path).convert("RGB")
        #print(f"カラー画像のサイズ: {color_image.size}, モード: {color_image.mode}")

        # サイズの確認
        if color_image.size[0] != 640 or color_image.size[1] != 480:
            print("警告: カラー画像のサイズが640x480ではありません。処理は続行しますが、予期せぬ結果になる可能性があります。")
        if heatmap_array.shape[0] != 480 or heatmap_array.shape[1] != 640:
            print("警告: ヒートマップのサイズが480x640ではありません。処理は続行しますが、予期せぬ結果になる可能性があります。")

        # 2. ヒートマップをPIL Imageに変換 (グレースケール)
        heatmap_pil = Image.fromarray(heatmap_array, 'L')
        #print(f"ヒートマップ（PIL変換後）のサイズ: {heatmap_pil.size}, モード: {heatmap_pil.mode}")

        # 3. カラー画像をRGBA形式に変換
        rgba_image = color_image.convert("RGBA")

        # 4. ヒートマップをアルファチャンネルとして設定
        rgba_image.putalpha(heatmap_pil)
        #print(f"結合後の画像のサイズ: {rgba_image.size}, モード: {rgba_image.mode}")

        return rgba_image


    except FileNotFoundError:
        print("エラー: 指定されたカラー画像ファイルが見つかりません。パスを確認してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

class ExpectCurveDataset(Dataset):
    def __init__(self,pt_path,color_path=None,gt_path=None,augment=False):
        self.augment = augment
        self.color_path=color_path
        self.gt_path=gt_path
        self.pt_baselist = []
        for pt in os.listdir(pt_path):
            self.pt_baselist.append(pt)
        self.pt_baselist.sort(key=natural_key)
        self.pt_full_list=[] # ポイント情報のフルパスをリスト状に格納
        self.color_full_list=[] # カラー画像のフルパスをリスト状に格納
        self.gt_full_list=[]
        for basename in self.pt_baselist:
            color_basename = basename.replace('_points.txt','.png')
            gt_basename = basename.replace('_points.txt','_gt.png')
            self.pt_full_list.append(os.path.join(pt_path,basename))
            if color_path is not None:
                self.color_full_list.append(os.path.join(self.color_path,color_basename))
            else:
                self.color_full_list.append(None)

            if gt_path is not None:
                self.gt_full_list.append(os.path.join(self.gt_path, gt_basename))
            else:
                self.gt_full_list.append(None) 
    
    def __len__(self): 
        return len(self.pt_full_list)

    def __getitem__(self, idx): 
        pt_file_path = self.pt_full_list[idx]
        color_image_path = self.color_full_list[idx]
        gt_file_path = self.gt_full_list[idx]
        

        point_data = read_points(pt_file_path)

        start_point = None
        candidates = [] 

        for label, x, y in point_data:
            if label.lower() == "growth":
                start_point = (x,y)
            elif label.lower() == "15cm":
                candidates.append((x,y)) # 候補点をリストに追加

        # PIL形式で画像を開く
        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size

        # GT画像も同様に開く
        gt_image_pil = None
        if gt_file_path and os.path.exists(gt_file_path):
            gt_image_pil = Image.open(gt_file_path).convert("L")
        else:
            # GTがない場合は黒画像で代替
            gt_image_pil = Image.new('L', (img_width, img_height), 0)

        
        if self.augment:
            # --- カラージッター (カラー画像のみに適用) ---
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            color_image_pil = color_jitter(color_image_pil)

            # --- ランダムな左右反転 (全てに同じ処理を適用) ---
            if random.random() > 0.5:
                color_image_pil = T.functional.hflip(color_image_pil)
                gt_image_pil = T.functional.hflip(gt_image_pil) # GT画像も反転
                if start_point: start_point = (img_width - 1 - start_point[0], start_point[1])
                candidates = [(img_width - 1 - x, y) for x, y in candidates]

            # --- ランダムな回転 (全てに同じ処理を適用) ---
            angle = random.uniform(-10, 10)
            color_image_pil = T.functional.rotate(color_image_pil, angle)
            gt_image_pil = T.functional.rotate(gt_image_pil, angle) # GT画像も回転
            angle_rad = -math.radians(angle)
            center = (img_width / 2, img_height / 2)
            if start_point: start_point = rotate_point(start_point, angle_rad, center)
            candidates = [rotate_point(p, angle_rad, center) for p in candidates]


        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)
        if start_point:
            plot_point_circle(heatmap, start_point[0], start_point[1], radius=5, value=255)
        if candidates:
            for cand_x, cand_y in candidates:
                plot_point_circle(heatmap, cand_x, cand_y, radius=5, value=255)

        # 3. 拡張後のデータを使ってヒートマップ生成と結合
        heatmap_pil = Image.fromarray(heatmap, 'L')
        rgba_image_pil = color_image_pil.copy()
        rgba_image_pil.putalpha(heatmap_pil)

        # 4. テンソルに変換
        rgba_tensor = T.ToTensor()(rgba_image_pil)
        gt_tensor = T.ToTensor()(gt_image_pil) # 拡張後のGT画像をテンソル化

        # 5. 拡張後の座標を正規化してテンソル化
        growth_point_tensor = torch.zeros(2, dtype=torch.float32)
        if start_point:
            growth_x_norm = torch.tensor(start_point[0] / img_width, dtype=torch.float32)
            growth_y_norm = torch.tensor(start_point[1] / img_height, dtype=torch.float32)
            growth_point_tensor = torch.stack([growth_x_norm, growth_y_norm])

        candidate_point_tensor = torch.zeros(2, dtype=torch.float32)
        if candidates:
            cand_x_norm = torch.tensor(candidates[0][0] / img_width, dtype=torch.float32)
            cand_y_norm = torch.tensor(candidates[0][1] / img_height, dtype=torch.float32)
            candidate_point_tensor = torch.stack([cand_x_norm, cand_y_norm])

        # deviceへの転送は学習ループ側で行う方が一般的ですが、ここでやっても問題ありません
        return rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_tensor



class UNetDataset(Dataset):
    def __init__(self, pt_path, color_path, gt_path, augment=False):
        self.augment = augment
        self.color_path=color_path
        self.gt_path=gt_path
        self.pt_baselist = []
        for pt in os.listdir(pt_path):
            self.pt_baselist.append(pt)
        self.pt_baselist.sort(key=natural_key)
        self.pt_full_list=[] # ポイント情報のフルパスをリスト状に格納
        self.color_full_list=[] # カラー画像のフルパスをリスト状に格納
        self.gt_full_list=[]
        for basename in self.pt_baselist:
            color_basename = basename.replace('_points.txt','.png')
            gt_basename = basename.replace('_points.txt','_gt.png')
            self.pt_full_list.append(os.path.join(pt_path,basename))
            if color_path is not None:
                self.color_full_list.append(os.path.join(self.color_path,color_basename))
            else:
                self.color_full_list.append(None)

            if gt_path is not None:
                self.gt_full_list.append(os.path.join(self.gt_path, gt_basename))
            else:
                self.gt_full_list.append(None) 

    def __len__(self):
        # ... (変更なし) ...
        return len(self.pt_full_list)

    def __getitem__(self, idx):
        # 1. 元のデータを読み込む
        pt_file_path = self.pt_full_list[idx]
        color_image_path = self.color_full_list[idx]
        gt_file_path = self.gt_full_list[idx]

        point_data = read_points(pt_file_path) # read_pointsは既存の関数と仮定
        start_point, candidates = None, []
        for label, x, y in point_data:
            if label.lower() == "growth": start_point = (x, y)
            elif label.lower() == "15cm": candidates.append((x, y))

        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size
        gt_image_pil = Image.open(gt_file_path).convert('L') if gt_file_path and os.path.exists(gt_file_path) else Image.new('L', (img_width, img_height), 0)

        # 2. データ拡張 (学習時のみ)
        if self.augment:
            # --- カラージッター (カラー画像のみに適用) ---
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            color_image_pil = color_jitter(color_image_pil)

            # --- ランダムな左右反転 (全てに同じ処理を適用) ---
            if random.random() > 0.5:
                color_image_pil = T.functional.hflip(color_image_pil)
                gt_image_pil = T.functional.hflip(gt_image_pil) # GT画像も反転
                if start_point: start_point = (img_width - 1 - start_point[0], start_point[1])
                candidates = [(img_width - 1 - x, y) for x, y in candidates]

            # --- ランダムな回転 (全てに同じ処理を適用) ---
            angle = random.uniform(-10, 10)
            color_image_pil = T.functional.rotate(color_image_pil, angle)
            gt_image_pil = T.functional.rotate(gt_image_pil, angle) # GT画像も回転
            angle_rad = -math.radians(angle)
            center = (img_width / 2, img_height / 2)
            if start_point: start_point = rotate_point(start_point, angle_rad, center)
            candidates = [rotate_point(p, angle_rad, center) for p in candidates]

        # 3. 入力画像(4ch)の作成
        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)
        if start_point: plot_point_circle(heatmap, start_point[0], start_point[1], radius=5, value=255) # plot_point_circleは既存の関数と仮定
        if candidates:
            for cand_x, cand_y in candidates:
                plot_point_circle(heatmap, cand_x, cand_y, radius=5, value=255)
        
        heatmap_pil = Image.fromarray(heatmap, 'L')
        rgba_image_pil = color_image_pil.copy()
        rgba_image_pil.putalpha(heatmap_pil)

        # 4. テンソルに変換
        input_tensor = T.ToTensor()(rgba_image_pil)
        target_tensor = T.ToTensor()(gt_image_pil) # GTも0-1のテンソルに

        return input_tensor, target_tensor

class PlusModel_ResNet(nn.Module):
    def __init__(self, img_width=640, img_height=480, num_points=2, dropout_rate=0.5):
        super(PlusModel_ResNet, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points

        # --- 1. 学習済みのResNet18をロード ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- 2. 入力層を4チャンネルに対応させる ---
        # 元の重みを保持しつつ、入力チャンネル数を変更する
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 元のRGBの重みをコピー
        self.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        # 4チャンネル目（ヒートマップ）の重みは、RGBの平均で初期化
        self.conv1.weight.data[:, 3, :, :] = original_conv1.weight.data.mean(dim=1)

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
        self.fc2 = nn.Linear(1024, self.num_points) # 2点の(x,y)で20次元

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
        normalized_outputs = torch.sigmoid(self.fc2(x)) 

        # 4. 正規化された出力を座標に変換
        start_x_norm, start_y_norm = growth_points[:,0:1], growth_points[:,1:2]
        end_x_norm, end_y_norm = candidate_points[:,0:1], candidate_points[:,1:2]
        
        vec_x_norm, vec_y_norm = end_x_norm - start_x_norm, end_y_norm - start_y_norm
        dist_norm = torch.sqrt(vec_x_norm**2 + vec_y_norm**2)
        safe_dist_norm = torch.where(dist_norm == 0, torch.tensor(1e-6, device=dist_norm.device), dist_norm)
        perp_x_norm, perp_y_norm = -vec_y_norm / safe_dist_norm, vec_x_norm / safe_dist_norm

        ref1_x, ref1_y = start_x_norm + 1/3 * vec_x_norm, start_y_norm + 1/3 * vec_y_norm
        ref2_x, ref2_y = start_x_norm + 2/3 * vec_x_norm, start_y_norm + 2/3 * vec_y_norm

        d1, d2 = (normalized_outputs[:, 0:1] - 0.5) * 2.0, (normalized_outputs[:, 1:2] - 0.5) * 2.0
        offset_scale = 0.9

        p1_x_norm, p1_y_norm = ref1_x + d1 * perp_x_norm * offset_scale, ref1_y + d1 * perp_y_norm * offset_scale
        p2_x_norm, p2_y_norm = ref2_x + d2 * perp_x_norm * offset_scale, ref2_y + d2 * perp_y_norm * offset_scale

        return torch.cat([p1_x_norm, p1_y_norm, p2_x_norm, p2_y_norm], dim=1)
    

class PlusModel(nn.Module):
    def __init__(self, img_width=640, img_height=480):
        super(PlusModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

        # 畳み込み層
        # 入力: 640x480x4 (バッチサイズ, 4, 480, 640)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1) # 640x480x32
        self.pool1 = nn.MaxPool2d(2, 2)                         # 320x240x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 320x240x64
        self.pool2 = nn.MaxPool2d(2, 2)                         # 160x120x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 160x120x128
        self.pool3 = nn.MaxPool2d(2, 2)                         # 80x60x128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 80x60x256
        self.pool4 = nn.MaxPool2d(2, 2)                         # 40x30x256

        # プーリング後の特徴マップのサイズを計算
        # 640 / (2*2*2*2) = 40
        # 480 / (2*2*2*2) = 30
        self.flattened_size = 256 * 40 * 30

        # 全結合層
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.5) # ドロップアウトで過学習を抑制
        self.fc2 = nn.Linear(512, 2) # 4つの出力値 (x1_norm, y1_norm, x2_norm, y2_norm)

    def forward(self, x, growth_points, candidate_points):
        # 入力は (Batch_size, Channels, Height, Width) の形式を想定
        # 640x480x4 -> (Batch, 4, 480, 640)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        # Flatten (平坦化)
        x = x.view(-1, self.flattened_size)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 最終出力層にはシグモイド関数を適用して0-1の範囲に正規化
        # x1_norm, y1_norm, x2_norm, y2_norm が得られる
        
        normalized_offsets = torch.sigmoid(self.fc2(x)) 
        

        start_x_norm = growth_points[:,0].unsqueeze(1)
        start_y_norm = growth_points[:,1].unsqueeze(1)
        end_x_norm = candidate_points[:,0].unsqueeze(1)
        end_y_norm = candidate_points[:,1].unsqueeze(1)

        # start_x_norm, start_y_norm, end_x_norm, end_y_norm はそれぞれ (Batch_size, 1) のテンソル
        # ベクトルの計算
        vec_x_norm = end_x_norm - start_x_norm
        vec_y_norm = end_y_norm - start_y_norm

        # 法線ベクトルの定義
        perp_x_norm = -vec_y_norm
        perp_y_norm = vec_x_norm

        # 単位法線ベクトル化
        dist_norm = torch.sqrt(vec_x_norm**2 + vec_y_norm**2)
        safe_dist_norm = torch.where(dist_norm == 0, torch.tensor(1e-6, device=dist_norm.device), dist_norm)
        perp_x_norm = perp_x_norm / safe_dist_norm
        perp_y_norm = perp_y_norm / safe_dist_norm

        # 参照点の計算
        # 参照点は線分の1/3と2/3の位置
        reference1_x_norm = start_x_norm + 1/3 * vec_x_norm
        reference1_y_norm = start_y_norm + 1/3 * vec_y_norm
        reference2_x_norm = start_x_norm + 2/3 * vec_x_norm
        reference2_y_norm = start_y_norm + 2/3 * vec_y_norm


        # モデルの出力から、垂直方向のオフセット量 d1, d2 を計算 (-1から1の範囲)
        d1 = (normalized_offsets[:, 0:1] - 0.5) * 2.0
        d2 = (normalized_offsets[:, 1:2] - 0.5) * 2.0

        # オフセットの最大量を調整する係数
        offset_scale = 0.1 

        # 制御点P1は、参照点1から法線方向にd1だけオフセットした点
        p1_x_norm = reference1_x_norm + d1 * perp_x_norm * offset_scale
        p1_y_norm = reference1_y_norm + d1 * perp_y_norm * offset_scale

        # 制御点P2は、参照点2から法線方向にd2だけオフセットした点
        p2_x_norm = reference2_x_norm + d2 * perp_x_norm * offset_scale
        p2_y_norm = reference2_y_norm + d2 * perp_y_norm * offset_scale

        # 正規化された座標を元の画像サイズにスケーリング
        x1 = p1_x_norm * self.img_width
        y1 = p1_y_norm * self.img_height
        x2 = p2_x_norm * self.img_width
        y2 = p2_y_norm * self.img_height

        # 座標を結合して (batch_size, 4) のテンソルとして返す
        return torch.cat([x1, y1, x2, y2], dim=1)
    

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Squeeze: Global Average Poolingで各チャネルの空間情報を集約
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation: 全結合層でチャネル間の相関を学習
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # Squeeze
        y = self.fc(y).view(b, c, 1, 1) # Excitation
        return x * y.expand_as(x) # 元の特徴マップに重みを掛ける
    

class DoubleConv(nn.Module):
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
        self.se = SEBlock(out_channels)
    def forward(self, x):
        x = self.double_conv(x)
        x = self.se(x)  # Squeeze-and-Excitationを適用
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # 最終出力にSigmoidを適用し、確率マップにする
        return torch.sigmoid(logits)

class ConvBlock(nn.Module):
    """
    U-Netで繰り返し使われる「畳み込みを2回行うブロック」
    (Conv -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    AC-UNetの心臓部であるアテンション・ゲート
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # Wg: スキップ接続から来る特徴(g)のための畳み込み
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # Wx: デコーダから来る特徴(x)のための畳み込み
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # psi: 2つの特徴を合わせた後、アテンションマップを生成するための畳み込み
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # それぞれを畳み込み
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 足し合わせてReLUを通す
        psi = self.relu(g1 + x1)
        # アテンションマップを生成
        psi = self.psi(psi)
        # 元のスキップ接続の特徴にアテンションを適用して返す
        return x * psi


class AC_UNet(nn.Module):
    """
    Attention U-Netの実装
    """
    def __init__(self, in_channels=4, out_channels=1):
        super(AC_UNet, self).__init__()

        # --- エンコーダ（収縮パス）---
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(in_channels, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024) # ボトムネック

        # --- デコーダ（拡張パス）---
        self.Up5 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.Att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = ConvBlock(1024, 512)

        self.Up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.Att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = ConvBlock(512, 256)
        
        self.Up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.Att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = ConvBlock(256, 128)
        
        self.Up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.Att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = ConvBlock(128, 64)

        # --- 最終出力層 ---
        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        # -- エンコーダ --
        e1 = self.Conv1(x)
        
        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)
        
        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)
        
        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)
        
        e5 = self.Maxpool(e4)
        e5 = self.Conv5(e5) # ボトムネック
        
        # -- デコーダ --
        d5 = self.Up5(e5)
        a4 = self.Att5(g=e4, x=d5)
        d5 = torch.cat((a4, d5), dim=1) # スキップ接続
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        a3 = self.Att4(g=e3, x=d4)
        d4 = torch.cat((a3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        a2 = self.Att3(g=e2, x=d3)
        d3 = torch.cat((a2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        a1 = self.Att2(g=e1, x=d2)
        d2 = torch.cat((a1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        # -- 出力 --
        out = self.Conv_1x1(d2)
        out = self.final_activation(out)
        
        return out