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
from train_unet import train_unet
from Save_map import generate_and_save_prob_maps
from up_chamfer_data import PointCloudDataset, PositionChamferModel, PositionChamferModel_ResNet, UNetPointCloudModel

# ===== 1. 設定項目 =====
# --- 入力パス ---
PT_PATH = '/home/onozawa/savepoints'
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
MAP_PATH_TRAIN ='train_prob_maps'
MAP_PATH_TEST = 'test_prob_maps'
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
    
    # テンソルデータと、文字列（ファイル名）を分離
    tensor_items = [item[:-1] for item in batch]
    basenames = [item[-1] for item in batch]
    
    # テンソルだけをデフォルトのcollate関数でバッチ化
    # (パディング処理もここで行う)
    max_len = max([t[3].shape[0] for t in tensor_items])
    for i in range(len(tensor_items)):
        rgba, g, c, gt_pc = tensor_items[i]
        if gt_pc.shape[0] < max_len:
            pad = torch.zeros(max_len - gt_pc.shape[0], 2, dtype=gt_pc.dtype)
            tensor_items[i] = (rgba, g, c, torch.cat([gt_pc, pad], dim=0))

    collated_tensors = torch.utils.data.dataloader.default_collate(tensor_items)
    collated_tensors.append(basenames) # 最後にファイル名のリストを追加
    return collated_tensors

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


def chamfer_distance_loss(pred_points, gt_points):
    pred_expanded = pred_points.unsqueeze(2) # (B, N, 1, 2)
    gt_expanded = gt_points.unsqueeze(1)     # (B, 1, M, 2)
    dist_matrix_sq = torch.sum((pred_expanded - gt_expanded) ** 2, dim=3)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix_sq, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix_sq, dim=1)
    return torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

# ===== 4. 訓練・評価関数 =====
def train_and_evaluate(l_r, pa, lam, number):
    if MODEL_NAME == 1:
        model = PositionChamferModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        BATCH_SIZE = 32
    elif MODEL_NAME == 2:
        model = PositionChamferModel_ResNet(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        BATCH_SIZE = 32  # ResNetはメモリ消費が大きいのでバッチサイズを小さく
    elif MODEL_NAME == 3:
        model = UNetPointCloudModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT,num_points=number).to(device)
        BATCH_SIZE = 4 # UNetはメモリ消費が大きいのでバッチサイズを小さく
    else:
        raise ValueError("無効なモデル名です。1, 2, または 3 を指定してください。")
    train_dataset = PointCloudDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, MAP_PATH_TRAIN, GT_POINTCLOUD_PATH,augment=True)
    test_dataset = PointCloudDataset(PT_SPLIT_TEST_PATH, COLOR_PATH, MAP_PATH_TEST, GT_POINTCLOUD_PATH,augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', patience=pa, verbose=True)

    train_losses, test_losses, best_test_loss, best_epoch = [], [], float('inf'), -1
    print(f"-----シャンファー距離での学習開始 (lr={l_r}, lambda={lam})-----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for rgba, g_pts, c_pts, gt_pc, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
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
            for rgba, g_pts, c_pts, gt_pc, _ in test_loader:
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
            if MODEL_NAME == 1:
                torch.save(model.state_dict(), f"best_up_chamfer_normal_model.pth")
            elif MODEL_NAME == 2:
                torch.save(model.state_dict(), f"best_up_chamfer_resnet_model.pth")
            elif MODEL_NAME == 3:
                torch.save(model.state_dict(), f"best_up_chamfer_unet_model_{number}.pth")
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}")
        scheduler.step(avg_test_loss)

    print(f"\n-----学習終了-----\nBest model at epoch {best_epoch + 1}")
    
    # グラフ保存
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.title(f'Chamfer Distance Loss (lr={l_r}, lam={lam})')
    plt.savefig(f"loss_graph_up_chamfer_lr{l_r}_lam{lam}.png")
    plt.close()

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    seed_everything()
    #split_data() # 必要なら実行



    # print("--- Step 1: Training Base Model (sigma=3.0) ---")
    # base_sigma = 2.0
    # base_lr = 5e-4
    # base_epochs = 150
    # # ベースモデルのファイル名を定義
    # base_model_path = f"model_sig_{base_sigma}.pth"

    # train_unet(l_r=base_lr, NUM_EPOCHS=base_epochs, sig=base_sigma, sigma_decay_epochs=50)

    # fine_tune_lr = 1e-5 
    # fine_tune_epochs = 100
    # train_unet(l_r=fine_tune_lr, 
    #             NUM_EPOCHS=fine_tune_epochs, 
    #             sig=0.3, 
    #             fine_tune_from=base_model_path,
    #             sigma_decay_epochs=10)
    

    # print("--- Step 2: UNet Save Model ---")
    # generate_and_save_prob_maps()

    print("--- Step 3: Training Chamfer Distance Model ---")
    # train_and_evaluate(l_r=LEARNING_RATE, pa=PATIENCE, lam=LAMBDA, number=9)
    Num_list=[3,5,7,9]
    for number in Num_list:
        print(f"### Number of Points: {number} ###")
        train_and_evaluate(l_r=LEARNING_RATE, pa=PATIENCE, lam=LAMBDA, number=number)