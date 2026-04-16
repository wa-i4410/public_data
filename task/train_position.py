# new_train_model.py (座標回帰モデル 訓練＆損失追跡)

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from coordinate_model import StemTransformer, PlotDataset # coordinate_model.pyからインポート
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
import csv
import shutil
import torch.optim as optim
import cv2


# ===== 1. 設定項目 =====
# パス設定
PT_PATH = ('/home/onozawa/savepoints')
PT_SPLIT_TRAIN_PATH =('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train')
PT_SPLIT_TEST_PATH = ('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test')
COLOR_PATH = ('/home/onozawa/CALORsample')
PT2_PATH = ('/home/onozawa/optimized_coords') # 正解座標CSVファイルのパス

# 基本パラメータ
IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 16

# PyTorchデバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★ 損失を追跡・可視化したいテスト画像のファイル名を指定 ★★★
TRACKED_IMAGES = ["609.png", "1519.png", "1649.png"] 
VISUALIZATION_SAVE_DIR = "training_progress_regression"
os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

# ===== 2. ユーティリティ関数 =====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_data(train_ratio=0.8, seed=42):
    # (変更なし)
    os.makedirs(PT_SPLIT_TRAIN_PATH, exist_ok=True)
    os.makedirs(PT_SPLIT_TEST_PATH, exist_ok=True)
    tmp_list=[f for f in os.listdir(PT_PATH) if f.endswith('.txt')]
    random.seed(seed)
    random.shuffle(tmp_list)
    split_idx = int(len(tmp_list) * train_ratio)
    print(f"全ファイル数: {len(tmp_list)}, 訓練: {split_idx}, テスト: {len(tmp_list) - split_idx}")
    for t in tmp_list[:split_idx]: shutil.copy(os.path.join(PT_PATH,t), os.path.join(PT_SPLIT_TRAIN_PATH,t))
    for t in tmp_list[split_idx:]: shutil.copy(os.path.join(PT_PATH,t), os.path.join(PT_SPLIT_TEST_PATH,t))

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

seed_everything()

# ===== 3. 評価と可視化の関数 =====

def evaluate_l1_loss(model, data_loader):
    """データローダー全体の平均L1損失を計算する"""
    model.eval()
    total_loss = 0.0
    l1_loss_fn = nn.L1Loss()
    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None: continue
            rgba_tensor, g_pts, c_pts, target_coords = batch_data
            rgba_gpu, g_pts_gpu, c_pts_gpu, target_gpu = rgba_tensor.to(device), g_pts.to(device), c_pts.to(device), target_coords.to(device)
            predicted_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)
            total_loss += l1_loss_fn(predicted_norm, target_gpu).item()
    return total_loss / len(data_loader) if data_loader else 0.0

def track_and_visualize_images(model, epoch, tracked_images):
    """指定された画像の損失を計算し、予測を可視化して保存する"""
    model.eval()
    losses = {}
    
    # 可視化用のデータセットとローダーを一時的に作成
    temp_dataset = PlotDataset(pt_path=PT_SPLIT_TRAIN_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH)
    l1_loss_fn = nn.L1Loss()

    for image_name in tracked_images:
        try:
            # データセットから該当のインデックスを探す
            idx = temp_dataset.pt_baselist.index(image_name.replace('.png', '_points.txt'))
            data = temp_dataset[idx]
            if data is None: continue
            
            rgba_tensor, g_pts, c_pts, target_coords = [d.unsqueeze(0).to(device) for d in data]

            with torch.no_grad():
                predicted_norm = model(rgba_tensor, g_pts, c_pts)
                loss = l1_loss_fn(predicted_norm, target_coords)
                losses[image_name] = loss.item()

                # --- オーバーレイ画像の保存 ---
                # 予測された正規化座標をピクセル座標に戻す
                pred_p1 = predicted_norm[0, 0:2] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                pred_p2 = predicted_norm[0, 2:4] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                
                # 正解の制御点もピクセル座標に戻す
                gt_p1 = target_coords[0, 0:2] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                gt_p2 = target_coords[0, 2:4] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)

                # 背景のカラー画像を読み込む
                color_img = Image.open(os.path.join(COLOR_PATH, image_name)).convert("RGB")
                overlay_np = np.array(color_img)
                
                # 点を描画
                cv2.circle(overlay_np, (int(gt_p1[0]), int(gt_p1[1])), 5, (0, 255, 0), -1) # 正解P1:緑
                cv2.circle(overlay_np, (int(gt_p2[0]), int(gt_p2[1])), 5, (0, 255, 0), -1) # 正解P2:緑
                cv2.circle(overlay_np, (int(pred_p1[0]), int(pred_p1[1])), 5, (255, 255, 0), -1) # 予測P1:黄
                cv2.circle(overlay_np, (int(pred_p2[0]), int(pred_p2[1])), 5, (255, 255, 0), -1) # 予測P2:黄

                # 画像を保存
                epoch_save_dir = os.path.join(VISUALIZATION_SAVE_DIR, f"epoch_{epoch+1:03d}")
                os.makedirs(epoch_save_dir, exist_ok=True)
                Image.fromarray(overlay_np).save(os.path.join(epoch_save_dir, image_name))
        
        except (ValueError, IndexError):
            print(f"Warning: 追跡用画像が見つかりません: {image_name}")
            continue
    return losses


# ===== 4. 訓練関数 =====
def train_model(l_r, pa):
    model = StemTransformer(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    train_dataset = PlotDataset(pt_path=PT_SPLIT_TRAIN_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH, augment=True)
    test_dataset = PlotDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', factor=0.2, patience=pa, verbose=True)

    l1_loss_fn = nn.L1Loss()
    train_losses, test_losses = [], []
    tracked_losses = {name: [] for name in TRACKED_IMAGES}
    best_test_loss, best_epoch = float('inf'), -1
    NUM_EPOCHS = 100

    print(f"-----座標回帰学習開始 (lr={l_r}, patience={pa})-----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            if batch_data is None: continue
            rgba_tensor, g_pts, c_pts, target_coords = batch_data
            rgba_gpu, g_pts_gpu, c_pts_gpu, target_gpu = rgba_tensor.to(device), g_pts.to(device), c_pts.to(device), target_coords.to(device)
            
            optim_.zero_grad()
            predicted_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)
            loss = l1_loss_fn(predicted_norm, target_gpu)
            loss.backward()
            optim_.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 評価
        avg_test_loss = evaluate_l1_loss(model, test_loader)
        test_losses.append(avg_test_loss)
        
        # 特定画像の損失と予測画像を記録
        specific_losses = track_and_visualize_images(model, epoch, TRACKED_IMAGES)
        for name, loss_val in specific_losses.items():
            tracked_losses[name].append(loss_val)

        # ログ表示
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")
        for name, loss_val in specific_losses.items():
            print(f"  -> Tracked {name}: {loss_val:.6f}")

        if avg_test_loss < best_test_loss:
            best_test_loss, best_epoch = avg_test_loss, epoch
            torch.save(model.state_dict(), f"best_regression_model_lr{l_r}.pth")
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.6f}")

        scheduler.step(avg_test_loss)

    print(f"\n-----学習終了-----\nBest model saved at epoch {best_epoch + 1}")

    # ===== グラフ描画処理 =====
    plt.figure(figsize=(14, 7))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss (All)', color='blue', linewidth=2)
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss (All)', color='orange', linewidth=2)
    
    colors = ['green', 'red', 'purple', 'brown']
    for i, (name, losses) in enumerate(tracked_losses.items()):
        plt.plot(range(1, len(losses) + 1), losses, label=f'Test Loss ({name})', linestyle='--', color=colors[i % len(colors)])

    plt.title(f'Training & Test Loss (L1 Regression)')
    plt.xlabel('Epoch')
    plt.ylabel('Average L1 Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_graph_lr{l_r}_tracked.png")
    plt.close()
    print("訓練とテストの損失グラフを保存しました。")
    
    return best_test_loss

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    split_data()
    best_loss = train_model(l_r=1e-4, pa=10)
    print(f"\n最終的なベストテストLOSS: {best_loss:.6f}")