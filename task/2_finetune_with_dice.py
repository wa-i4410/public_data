# finetune_final.py (全データでのファインチューニング)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pydiffvg
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import shutil
import re
import cv2

# モデルとデータセットのクラスをインポート
from newmodel import ExpectCurveDataset
from coordinate_model import PositionModel

# ===== 1. 設定項目 =====
# --- 入力パス ---
PRETRAINED_MODEL_PATH = "pretrained_model.pth" # L1学習で得たモデル
PT_PATH = '/home/onozawa/savepoints'
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_PATH = '/home/onozawa/GT2dspline'

# --- 学習パラメータ ---
IMG_WIDTH, IMG_HEIGHT = 640, 480
LEARNING_RATE = 1e-5    # ファインチューニングなので学習率は小さく
PATIENCE = 15           # schedulerの我慢エポック数
NUM_EPOCHS = 100        # 全体学習のエポック数
BATCH_SIZE = 16         # GPUメモリに応じて調整
STROKE_WIDTH = 1.5      # 描画する線の太さ
SIGMA = 5.0             # 学習を安定させるための「ぼかし」の強さ
LAMBDA = 0.0           # 平滑化損失の重み

# --- システム設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)
pydiffvg.set_print_timing(False)
render_func = pydiffvg.RenderFunction.apply

# ===== 2. ユーティリティ関数 =====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
def dice_loss(pred, target, smooth=1e-6):
    pred, target = pred.contiguous().view(-1), target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

seed_everything()

# ===== 3. 評価専用の関数 =====
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    bce_loss_fn = nn.BCELoss()
    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None: continue
            rgba_tensor, g_pts, c_pts, gt_tensor = batch_data
            rgba, g_pts, c_pts, gt = rgba_tensor.to(device), g_pts.to(device), c_pts.to(device), gt_tensor.to(device)
            
            predicted_norm = model(rgba, g_pts, c_pts)
            batch_loss = 0.0
            for i in range(rgba.shape[0]):
                p1 = predicted_norm[i, 0:2] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                p2 = predicted_norm[i, 2:4] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                start_px = g_pts[i] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                end_px = c_pts[i] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                points = torch.stack([start_px, p1, p2, end_px])

                path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(STROKE_WIDTH), is_closed=False)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=torch.tensor([1.0]*4, device=device))
                args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
                rendered_mask = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)[:, :, 3].unsqueeze(0).unsqueeze(0)
                target_mask = gt[i].unsqueeze(0)

                if SIGMA > 0:
                    blurred_pred = TF.gaussian_blur(rendered_mask, kernel_size=5, sigma=SIGMA)
                    blurred_gt = TF.gaussian_blur(target_mask, kernel_size=5, sigma=SIGMA)
                else:
                    blurred_pred, blurred_gt = rendered_mask, target_mask

                loss_dice = dice_loss(blurred_pred, blurred_gt)
                loss_bce = bce_loss_fn(blurred_pred, blurred_gt)
                batch_loss += (loss_dice + loss_bce).item()
            
            total_loss += batch_loss / rgba.shape[0]
            
    return total_loss / len(data_loader) if data_loader else 0.0

# ===== 4. ファインチューニング関数 =====
def finetune_model():
    model = PositionModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    if os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"事前学習済みモデル '{PRETRAINED_MODEL_PATH}' をロードしました。")
    else:
        print(f"警告: 事前学習済みモデルが見つかりません。学習はランダムな初期値から始まります。")

    train_dataset = ExpectCurveDataset(pt_path=PT_SPLIT_TRAIN_PATH, color_path=COLOR_PATH, gt_path=GT_PATH, augment=True)
    test_dataset = ExpectCurveDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, gt_path=GT_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    
    optim_ = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', patience=PATIENCE, verbose=True)
    bce_loss_fn = nn.BCELoss()

    train_losses, test_losses = [], []
    best_test_loss, best_epoch = float('inf'), -1

    print(f"-----ファインチューニング開始 (lr={LEARNING_RATE}, lam={LAMBDA}, sigma={SIGMA})-----")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            if batch_data is None: continue
            rgba_tensor, g_pts, c_pts, gt_tensor = batch_data
            rgba, g_pts, c_pts, gt = rgba_tensor.to(device), g_pts.to(device), c_pts.to(device), gt_tensor.to(device)

            optim_.zero_grad()
            predicted_norm = model(rgba, g_pts, c_pts)
            
            batch_loss = 0.0
            for i in range(rgba.shape[0]):
                p1 = predicted_norm[i, 0:2] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                p2 = predicted_norm[i, 2:4] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                start_px = g_pts[i] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                end_px = c_pts[i] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                points = torch.stack([start_px, p1, p2, end_px])

                path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(STROKE_WIDTH), is_closed=False)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=torch.tensor([1.0]*4, device=device))
                args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
                rendered_mask = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)[:, :, 3].unsqueeze(0).unsqueeze(0)
                target_mask = gt[i].unsqueeze(0)

                if SIGMA > 0:
                    blurred_pred = TF.gaussian_blur(rendered_mask, kernel_size=5, sigma=SIGMA)
                    blurred_gt = TF.gaussian_blur(target_mask, kernel_size=5, sigma=SIGMA)
                else:
                    blurred_pred, blurred_gt = rendered_mask, target_mask

                loss_dice = dice_loss(blurred_pred, blurred_gt)
                loss_bce = bce_loss_fn(blurred_pred, blurred_gt)
                
                v1, v2, v3 = points[1] - points[0], points[2] - points[1], points[3] - points[2]
                cos_sim1 = F.cosine_similarity(v1, v2, dim=0)
                cos_sim2 = F.cosine_similarity(v2, v3, dim=0)
                regularization_loss = (1 + cos_sim1) + (1 + cos_sim2)
                
                total_loss_for_item = (loss_dice + loss_bce) + LAMBDA * regularization_loss
                batch_loss += total_loss_for_item

            avg_batch_loss = batch_loss / rgba.shape[0]
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim_.step()
            total_train_loss += avg_batch_loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        avg_test_loss = evaluate_model(model, test_loader)
        test_losses.append(avg_test_loss)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Test Loss: {avg_test_loss:.6f}")

        if avg_test_loss < best_test_loss:
            best_test_loss, best_epoch = avg_test_loss, epoch
            torch.save(model.state_dict(), f"best_finetuned_model.pth")
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.6f}")
        scheduler.step(avg_test_loss)

    print(f"\n-----ファインチューニング終了-----\nBest model at epoch {best_epoch + 1} with Test Loss: {best_test_loss:.6f}")

    # (グラフ保存処理)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.title('Finetuning Loss (Dice+BCE)')
    plt.savefig("finetuning_loss_graph.png")
    plt.close()

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    # split_data()
    finetune_model()