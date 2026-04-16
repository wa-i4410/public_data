# train_with_chamfer.py

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
# ===== 1. 設定項目 =====
# --- 入力パス ---
PT_PATH = '/home/onozawa/savepoints'
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
# ★★★ 正解の点群データ(CSV)が保存されているパス ★★★
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'


# --- 読み込むモデル名  ---
MODEL_NAME = 3 # 1: PositionChamferModel, 2: PositionChamferModel_ResNet, 3: UNetPointCloudModel



# --- 学習パラメータ ---
IMG_WIDTH, IMG_HEIGHT = 640, 480
LEARNING_RATE = 1e-4
PATIENCE = 20
NUM_EPOCHS = 200 # 十分なエポック数
BATCH_SIZE = 32
LAMBDA = 0  # 平滑化損失の重み

# --- システム設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. ユーティリティ & データセット =====

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    
    # バッチ内のGT点群のサイズを揃える (パディング)
    max_len = max([b[3].shape[0] for b in batch])
    for i in range(len(batch)):
        rgba, g, c, gt_pc = batch[i]
        if gt_pc.shape[0] < max_len:
            pad = torch.zeros(max_len - gt_pc.shape[0], 2, dtype=gt_pc.dtype)
            batch[i] = (rgba, g, c, torch.cat([gt_pc, pad], dim=0))
            
    return torch.utils.data.dataloader.default_collate(batch)

class PointCloudDataset(Dataset):
    def __init__(self, pt_path, color_path, gt_pointcloud_path):
        self.pt_path = pt_path # ★ 修正点: pt_pathをクラス変数として保持
        self.color_path = color_path
        self.gt_pointcloud_path = gt_pointcloud_path
        self.pt_baselist = sorted([f for f in os.listdir(pt_path) if f.endswith('.txt')], key=natural_key)
        
    def __len__(self): 
        return len(self.pt_baselist)

    def __getitem__(self, idx): 
        basename = self.pt_baselist[idx]
        # ★ 修正点: self.pt_path を使う
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

        heatmap = np.zeros((img_height, img_width), dtype=np.uint8)
        plot_point_circle(heatmap, start_point[0], start_point[1], 5, 255)
        plot_point_circle(heatmap, candidate_point[0], candidate_point[1], 5, 128)
        
        heatmap_pil = Image.fromarray(heatmap, 'L')
        rgba_image_pil = color_image_pil.copy()
        rgba_image_pil.putalpha(heatmap_pil)

        rgba_tensor = T.ToTensor()(rgba_image_pil)
        growth_point_tensor = torch.tensor([start_point[0] / img_width, start_point[1] / img_height], dtype=torch.float32)
        candidate_point_tensor = torch.tensor([candidate_point[0] / img_width, candidate_point[1] / img_height], dtype=torch.float32)

        gt_points = []
        with open(gt_pc_file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader: gt_points.append([float(row[0]), float(row[1])])
        gt_pointcloud_tensor = torch.tensor(gt_points, dtype=torch.float32)

        return rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_pointcloud_tensor


# ===== 3. モデル & 損失関数 =====

class PositionChamferModel(nn.Module):
    """
    シャンファー距離を用いた位置予測モデル
    """
    def __init__(self, img_width=640, img_height=480, num_points=10):
        super(PositionChamferModel, self).__init__()
        self.img_width = img_width
        self.img_height = img_height
        self.num_points = num_points

        # --- 畳み込み層に padding=1 を追加 ---
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # この計算が正しくなります
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
    def __init__(self, img_width=640, img_height=480, num_points=3):
        super(UNetPointCloudModel, self).__init__()
        self.num_points = num_points

        # --- 1. U-Net バックボーン ---
        self.inc = ConvBlock(4, 64)
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

        # --- 2. 予測ヘッド ---
        # U-Netの出力特徴マップのチャンネル数
        unet_output_channels = 64
        # U-Netの出力特徴マップを平坦化したサイズ
        flattened_size = unet_output_channels * img_width * img_height
        
        self.prediction_head = nn.Sequential(
            nn.Linear(flattened_size + 4, 1024),
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
        
        # --- 予測ヘッドによる座標予測 ---
        # 特徴マップを平坦化
        x_flat = x.view(x.size(0), -1)
        
        # 座標データを連結
        coords_input = torch.cat([growth_points, candidate_points], dim=1)
        combined_features = torch.cat([x_flat, coords_input], dim=1)
        
        # 最終的な10点の座標を予測
        pred_points_offsets = self.prediction_head(combined_features).view(-1, self.num_points, 2) * 0.5
        
        # 始点から終点への線形補間をベースにオフセットを加える
        start_points = growth_points.unsqueeze(1)
        end_points = candidate_points.unsqueeze(1)
        t = torch.linspace(0, 1, self.num_points, device=x.device).unsqueeze(0).unsqueeze(2)
        base_path = start_points + t * (end_points - start_points)
        
        pred_points_norm = base_path + pred_points_offsets
        
        return pred_points_norm


def chamfer_distance_loss(pred_points, gt_points):
    pred_expanded = pred_points.unsqueeze(2) # (B, N, 1, 2)
    gt_expanded = gt_points.unsqueeze(1)     # (B, 1, M, 2)
    dist_matrix_sq = torch.sum((pred_expanded - gt_expanded) ** 2, dim=3)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix_sq, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix_sq, dim=1)
    return torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

# ===== 4. 訓練・評価関数 =====
def train_and_evaluate(l_r, pa, lam):
    if MODEL_NAME == 1:
        model = PositionChamferModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
    elif MODEL_NAME == 2:
        model = PositionChamferModel_ResNet(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
    elif MODEL_NAME == 3:
        model = UNetPointCloudModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
    else:
        raise ValueError("無効なモデル名です。1, 2, または 3 を指定してください。")
    train_dataset = PointCloudDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH)
    test_dataset = PointCloudDataset(PT_SPLIT_TEST_PATH, COLOR_PATH, GT_POINTCLOUD_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', patience=pa, verbose=True)

    train_losses, test_losses, best_test_loss, best_epoch = [], [], float('inf'), -1
    print(f"-----シャンファー距離での学習開始 (lr={l_r}, lambda={lam})-----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for rgba, g_pts, c_pts, gt_pc in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            rgba, g_pts, c_pts, gt_pc = rgba.to(device), g_pts.to(device), c_pts.to(device), gt_pc.to(device)
            optim_.zero_grad()
            
            pred_pc_norm = model(rgba, g_pts, c_pts)
            pred_pc_pixel = pred_pc_norm * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device).view(1, 1, 2)
            
            chamfer_loss = chamfer_distance_loss(pred_pc_pixel, gt_pc)
            
            vecs = pred_pc_pixel[:, 1:] - pred_pc_pixel[:, :-1]
            cos_sims = F.cosine_similarity(vecs[:, :-1], vecs[:, 1:], dim=2)
            regularization_loss = torch.mean(1 - cos_sims)
            
            loss = chamfer_loss + lam * regularization_loss
            loss.backward()
            optim_.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 評価ループ
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for rgba, g_pts, c_pts, gt_pc in test_loader:
                if rgba is None: continue
                rgba, g_pts, c_pts, gt_pc = rgba.to(device), g_pts.to(device), c_pts.to(device), gt_pc.to(device)
                pred_pc_norm = model(rgba, g_pts, c_pts)
                pred_pc_pixel = pred_pc_norm * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device).view(1, 1, 2)
                total_test_loss += chamfer_distance_loss(pred_pc_pixel, gt_pc).item()
        
        avg_test_loss = total_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss, best_epoch = avg_test_loss, epoch
            torch.save(model.state_dict(), f"best_chamfer_model.pth")
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}")
        scheduler.step(avg_test_loss)

    print(f"\n-----学習終了-----\nBest model at epoch {best_epoch + 1}")
    
    # グラフ保存
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.title(f'Chamfer Distance Loss (lr={l_r}, lam={lam})')
    plt.savefig(f"loss_graph_chamfer_lr{l_r}_lam{lam}.png")
    plt.close()

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    # seed_everything()
    # split_data() # 必要なら実行
    train_and_evaluate(l_r=LEARNING_RATE, pa=PATIENCE, lam=LAMBDA)