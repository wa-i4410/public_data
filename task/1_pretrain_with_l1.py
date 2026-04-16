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
from coordinate_model import  PlotDataset,StemTransformer # coordinate_model.pyからインポート
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
BATCH_SIZE = 32

# PyTorchデバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★ 損失を追跡・可視化したいテスト画像のファイル名を指定 ★★★
TRACKED_IMAGES = ["631.png", "2357.png", "7445.png"] 
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

def pretrain_l1_model(l_r=1e-4, pa=10, num_epochs=300):
    model = StemTransformer(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    train_dataset = PlotDataset(pt_path=PT_SPLIT_TRAIN_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH, augment=True)
    test_dataset = PlotDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    optim_ = optim.Adam(model.parameters(), lr=l_r)
    l1_loss_fn = nn.L1Loss()
    
    print("----- 事前学習開始 (L1 Loss) -----")
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch_data in tqdm(train_loader, f"Epoch {epoch+1}/{num_epochs}"):
            if not batch_data: continue
            rgba, g_pts, c_pts, target_coords = batch_data
            rgba, g_pts, c_pts, target = rgba.to(device), g_pts.to(device), c_pts.to(device), target_coords.to(device)
            optim_.zero_grad()
            predicted = model(rgba, g_pts, c_pts)
            loss = l1_loss_fn(predicted, target)
            loss.backward()
            optim_.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1} | Pre-train Loss (L1): {avg_train_loss:.6f}")
        
    torch.save(model.state_dict(), "pretrained_model.pth")
    print("\n事前学習済みモデルを 'pretrained_model.pth' に保存しました。")

if __name__ == '__main__':
    split_data() # 1回実行すればOK
    # pretrain_l1_model()