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
import csv

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

class PlotDataset(Dataset):
    def __init__(self,pt_path,color_path=None,pt2_path=None,augment=False):
        self.augment = augment
        self.color_path=color_path
        self.pt2_path=pt2_path
        self.pt_baselist = []
        for pt in os.listdir(pt_path):
            self.pt_baselist.append(pt)
        self.pt_baselist.sort(key=natural_key)
        self.pt_full_list=[] # ポイント情報のフルパスをリスト状に格納
        self.color_full_list=[] # カラー画像のフルパスをリスト状に格納
        self.pt2_full_list=[]
        for basename in self.pt_baselist:
            color_basename = basename.replace('_points.txt','.png')
            pt2_basename = basename.replace('_points.txt','_controls.csv')
            self.pt_full_list.append(os.path.join(pt_path,basename))
            if color_path is not None:
                self.color_full_list.append(os.path.join(self.color_path,color_basename))
            else:
                self.color_full_list.append(None)

            if pt2_path is not None:
                self.pt2_full_list.append(os.path.join(self.pt2_path, pt2_basename))
            else:
                self.pt2_full_list.append(None) 
    
    def __len__(self): 
        return len(self.pt_full_list)

    def __getitem__(self, idx): 
        pt_file_path = self.pt_full_list[idx]
        color_image_path = self.color_full_list[idx]
        pt2_file_path = self.pt2_full_list[idx]
        

        point_data = read_points(pt_file_path)

        start_point = None
        candidates = [] 

        if not all(os.path.exists(p) for p in [pt_file_path, color_image_path, pt2_file_path]):
            print(f"Warning: 関連ファイルが見つかりません。スキップします: {os.path.basename(pt_file_path)}")
            return None
        
        # 2. ポイントデータから座標を抽出
        for label, x, y in point_data:
            if label.lower() == "growth":
                start_point = (x,y)
            elif label.lower() == "15cm":
                candidates.append((x,y)) # 候補点をリストに追加

        # PIL形式で画像を開く
        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size


        gt_controls = {}
        with open(pt2_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダー行をスキップ
            for row in reader:
                label, x_str, y_str = row
                gt_controls[label] = (float(x_str), float(y_str))

        if 'c1' not in gt_controls or 'c2' not in gt_controls:
            print(f"Warning: CSVファイルにc1またはc2がありません: {pt2_file_path}")
            return None
        

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

        p1_gt_x, p1_gt_y = gt_controls['c1']
        p2_gt_x, p2_gt_y = gt_controls['c2']

        p1_gt_x_norm = p1_gt_x / img_width
        p1_gt_y_norm = p1_gt_y / img_height
        p2_gt_x_norm = p2_gt_x / img_width
        p2_gt_y_norm = p2_gt_y / img_height
        target_coords_tensor = torch.tensor([p1_gt_x_norm, p1_gt_y_norm, p2_gt_x_norm, p2_gt_y_norm], dtype=torch.float32)

        return rgba_tensor, growth_point_tensor, candidate_point_tensor, target_coords_tensor
    
class PositionModel(nn.Module):
    def __init__(self, img_width=640, img_height=480):
        super(PositionModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height

        # 畳み込み層 (変更なし)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flattened_size = 256 * (img_width // 16) * (img_height // 16)

        # ===== ★★★ 修正点1: 全結合層の入力サイズを変更 ★★★ =====
        # CNNの出力特徴量に、座標データ4つ分 (始点x,y, 終点x,y) を加える
        self.fc1 = nn.Linear(self.flattened_size + 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x, growth_points, candidate_points):
        # 1. CNNで画像特徴量を抽出
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.flattened_size)

        # ===== ★★★ 修正点2: 座標データを連結 ★★★ =====
        # 始点と終点の正規化座標を1つのテンソルにまとめる
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        # 画像特徴量と座標データを連結
        combined_features = torch.cat([x, coords_input], dim=1)
        
        # 3. 連結した特徴量を全結合層に入力
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
    
class PositionalEncoding(nn.Module):
    """
    Transformerに入力する特徴に「位置情報」を付与するためのクラス
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class StemTransformer(nn.Module):
    """
    アテンション機構（Transformer）を導入した新しい経路予測モデル
    """
    def __init__(self, img_width=640, img_height=480, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3):
        super(StemTransformer, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        
        # --- 1. CNNバックボーン ---
        # 4chの入力画像を、d_model次元の特徴マップに変換する
        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=2, padding=1), # 320x240
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 160x120
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, d_model, kernel_size=3, stride=2, padding=1), # 80x60
            nn.ReLU(),
            nn.BatchNorm2d(d_model),
        )

        # --- 2. Transformer ---
        self.pos_encoder = PositionalEncoding(d_model)
        
        # エンコーダ層
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # デコーダ層
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # --- 3. 予測ヘッド ---
        # 始点・終点の座標(4)を入力するための線形層
        self.query_embed = nn.Linear(4, d_model)
        
        # 最終的な制御点座標(4)を出力するための線形層 (d1, d2の2値を出力)
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, image_tensor, growth_points, candidate_points):
        # 1. CNNで画像から特徴マップを抽出
        # shape: (batch, d_model, H, W)
        features = self.backbone(image_tensor)

        # 2. Transformerに入力するために形状を変更
        # (batch, d_model, H, W) -> (batch, H*W, d_model)
        bs, c, h, w = features.shape
        features = features.flatten(2).permute(0, 2, 1)
        
        # 位置エンコーディングを付与
        # (batch, H*W, d_model) -> (H*W, batch, d_model) -> (batch, H*W, d_model)
        features = self.pos_encoder(features.permute(1, 0, 2)).permute(1, 0, 2)

        # 3. Transformerエンコーダで、文脈を考慮した特徴に変換
        memory = self.transformer_encoder(features)

        # 4. 始点と終点の座標から「質問(Query)」を生成
        # shape: (batch, 4) -> (batch, 1, d_model)
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        query = self.query_embed(coords_input).unsqueeze(1)

        # 5. Transformerデコーダで、質問を使って画像特徴から必要な情報を抽出
        # shape: (batch, 1, d_model)
        hs = self.transformer_decoder(query, memory)
        
        # 6. 予測ヘッドで最終的なオフセット値(d1, d2)を予測
        # shape: (batch, 1, 2) -> (batch, 2)
        normalized_outputs = torch.sigmoid(self.prediction_head(hs.squeeze(1)))

        # --- (ここから先の相対座標の計算は、以前のモデルと同じ) ---
        start_x_norm, start_y_norm = growth_points[:,0:1], growth_points[:,1:2]
        end_x_norm, end_y_norm = candidate_points[:,0:1], candidate_points[:,1:2]
        
        vec_x_norm, vec_y_norm = end_x_norm - start_x_norm, end_y_norm - start_y_norm
        dist_norm = torch.sqrt(vec_x_norm**2 + vec_y_norm**2).clamp(min=1e-6)
        perp_x_norm, perp_y_norm = -vec_y_norm / dist_norm, vec_x_norm / dist_norm

        ref1_x, ref1_y = start_x_norm + 1/3 * vec_x_norm, start_y_norm + 1/3 * vec_y_norm
        ref2_x, ref2_y = start_x_norm + 2/3 * vec_x_norm, start_y_norm + 2/3 * vec_y_norm

        d1, d2 = (normalized_outputs[:, 0:1] - 0.5) * 2.0, (normalized_outputs[:, 1:2] - 0.5) * 2.0
        offset_scale = 0.5 # この値はハイパーパラメータとして調整

        p1_x_norm, p1_y_norm = ref1_x + d1 * perp_x_norm * offset_scale, ref1_y + d1 * perp_y_norm * offset_scale
        p2_x_norm, p2_y_norm = ref2_x + d2 * perp_x_norm * offset_scale, ref2_y + d2 * perp_y_norm * offset_scale

        return torch.cat([p1_x_norm, p1_y_norm, p2_x_norm, p2_y_norm], dim=1)
    

class PointCloudDataset(Dataset):
    """
    シャンファー距離損失のために、正解データとして「点群(Point Cloud)」を
    CSVファイルから読み込むように修正したデータセットクラス。
    """
    def __init__(self, pt_path, color_path=None, gt_pointcloud_path=None, augment=False):
        self.augment = augment
        self.color_path = color_path
        # ★ 引数名を分かりやすく変更
        self.gt_pointcloud_path = gt_pointcloud_path 
        
        self.pt_baselist = sorted([f for f in os.listdir(pt_path) if f.endswith('.txt')], key=natural_key)
        
        self.pt_full_list = []
        self.color_full_list = []
        self.gt_pc_full_list = [] # ★ GT点群ファイルのリスト

        for basename in self.pt_baselist:
            color_basename = basename.replace('_points.txt', '.png')
            # ★ 正解点群のファイル名を生成
            gt_pc_basename = basename.replace('_points.txt', '_gt_pointcloud.csv') 
            
            self.pt_full_list.append(os.path.join(pt_path, basename))
            self.color_full_list.append(os.path.join(self.color_path, color_basename))
            
            if self.gt_pointcloud_path:
                self.gt_pc_full_list.append(os.path.join(self.gt_pointcloud_path, gt_pc_basename))

    def __len__(self): 
        return len(self.pt_full_list)

    def __getitem__(self, idx): 
        pt_file_path = self.pt_full_list[idx]
        color_image_path = self.color_full_list[idx]
        gt_pc_file_path = self.gt_pc_full_list[idx]
        
        # --- 必要なファイルが存在するかチェック ---
        if not all(os.path.exists(p) for p in [pt_file_path, color_image_path, gt_pc_file_path]):
            return None
        
        # --- 1. モデルへの入力データを準備 (ここは変更なし) ---
        point_data = read_points(pt_file_path)
        start_point, candidates = None, []
        for label, x, y in point_data:
            if label.lower() == "growth": start_point = (x, y)
            elif label.lower() == "15cm": candidates.append((x, y))
        
        if start_point is None or not candidates: return None

        color_image_pil = Image.open(color_image_path).convert('RGB')
        img_width, img_height = color_image_pil.size
        
        # (データ拡張が必要な場合はここに記述)
        
        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)
        plot_point_circle(heatmap, start_point[0], start_point[1], radius=5, value=255)
        plot_point_circle(heatmap, candidates[0][0], candidates[0][1], radius=5, value=128)
        
        heatmap_pil = Image.fromarray(heatmap, 'L')
        rgba_image_pil = color_image_pil.copy()
        rgba_image_pil.putalpha(heatmap_pil)
        
        rgba_tensor = T.ToTensor()(rgba_image_pil)
        growth_point_tensor = torch.tensor([start_point[0] / img_width, start_point[1] / img_height], dtype=torch.float32)
        candidate_point_tensor = torch.tensor([candidates[0][0] / img_width, candidates[0][1] / img_height], dtype=torch.float32)

        # --- 2. 正解ラベル（点群）をCSVから読み込む ---
        gt_points = []
        with open(gt_pc_file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader) # ヘッダー行'x','y'をスキップ
            for row in reader:
                gt_points.append([float(row[0]), float(row[1])])
        
        # NumPy配列を経由してPyTorchテンソルに変換
        gt_pointcloud_tensor = torch.tensor(np.array(gt_points), dtype=torch.float32)

        # --- 3. 最終的な戻り値を定義 ---
        # 4番目の戻り値が、正解の「点群」テンソルになる
        return rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_pointcloud_tensor
    

class PositionChamferModel(nn.Module):
    def __init__(self, img_width=640, img_height=480, num_points=10):
        super(PositionChamferModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points

        # --- 畳み込み層 ---
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.flattened_size = 256 * (img_width // 16) * (img_height // 16)

        # --- 全結合層 (訓練時のサイズに修正) ---
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
        
        # 始点から終点への線形補間をベースにオフセットを予測
        offsets = self.fc2(x).view(-1, self.num_points, 2) * 0.5
        
        start_points = growth_points.unsqueeze(1)
        end_points = candidate_points.unsqueeze(1)
        t = torch.linspace(0, 1, self.num_points, device=x.device).unsqueeze(0).unsqueeze(2)
        base_path = start_points + t * (end_points - start_points)
        
        pred_points_norm = base_path + offsets
        return pred_points_norm