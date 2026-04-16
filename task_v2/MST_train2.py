import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random

# ★ 新しい Transformer モデルをインポート
from MST_model2 import TomatoStemDataset, StemTransformerModel

# ===== 1. 設定項目 =====
# --- 入力パス ---
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'

# --- 学習パラメータ ---
IMG_WIDTH, IMG_HEIGHT = 640, 480
LEARNING_RATE = 1e-4
PATIENCE = 10
NUM_EPOCHS = 300  
BATCH_SIZE = 4  
VALIDATION_SPLIT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. ユーティリティ =====
def custom_collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    tensor_items = [list(item[:-1]) for item in batch]
    basenames = [item[-1] for item in batch]
    
    max_len = max([t[3].shape[0] for t in tensor_items])
    for i in range(len(tensor_items)):
        gt_pc = tensor_items[i][3]
        if gt_pc.shape[0] < max_len:
            pad = torch.zeros(max_len - gt_pc.shape[0], 2, dtype=gt_pc.dtype)
            tensor_items[i][3] = torch.cat([gt_pc, pad], dim=0)

    collated_tensors = torch.utils.data.dataloader.default_collate(tensor_items)
    collated_tensors.append(basenames)
    return collated_tensors

# ===== 3. Transformer用のLoss関数 =====
def transformer_curve_loss(pred_curve, gt_pc, pred_ctrl,growth_points, w_l1=100.0, w_chamfer=10.0, w_smooth=1.0):
    """
    pred_curve: (B, 100, 2) - 予測されたスプライン曲線
    gt_curve: (B, 100, 2) - 正解の100点群 (正規化済)
    pred_ctrl: (B, 10, 2) - 予測された制御点 (両端含む)
    gt_intermediate: (B, 8, 2) - 正解の中間8点 (正規化済)
    """
    B = pred_curve.size(0)
    
    aligned_gts = []
    for i in range(B):
        dist_gt_start = torch.norm(gt_pc[i, 0] - growth_points[i])
        dist_gt_end = torch.norm(gt_pc[i, -1] - growth_points[i])
        if dist_gt_end < dist_gt_start:
            aligned_gts.append(torch.flip(gt_pc[i], dims=[0])) # 逆向きなら反転
        else:
            aligned_gts.append(gt_pc[i])
    aligned_gt = torch.stack(aligned_gts) # (B, 100, 2)

    # 向きを揃えたGTから、L1 Loss用の中間8点を抽出
    num_intermediate = pred_ctrl.size(1) - 2
    indices = torch.linspace(1, aligned_gt.size(1) - 2, steps=num_intermediate).long()
    gt_intermediate = aligned_gt[:, indices, :] # (B, 8, 2)

    # ① Coordinate L1 Loss
    pred_intermediate = pred_ctrl[:, 1:-1, :]
    coord_l1_loss = F.l1_loss(pred_intermediate, gt_intermediate)

    # ② Chamfer Loss (緑の線への吸い付き) ※ここでは向きは関係ないのでそのまま
    dist_matrix_sq = torch.sum((pred_curve.unsqueeze(2) - gt_pc.unsqueeze(1)) ** 2, dim=3)
    dist_matrix = torch.sqrt(dist_matrix_sq + 1e-8)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)
    chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

    # ③ Smoothness Loss (曲げエネルギー)
    bending = pred_ctrl[:, 2:, :] - 2 * pred_ctrl[:, 1:-1, :] + pred_ctrl[:, :-2, :]
    smoothness_loss = torch.mean(torch.norm(bending, dim=2)**2)

    total_loss = 100.0 * (w_l1 * coord_l1_loss + w_chamfer * chamfer_loss + w_smooth * smoothness_loss)
    
    return total_loss, coord_l1_loss, chamfer_loss, smoothness_loss

# ===== 4. 訓練・評価関数 =====
def train_and_evaluate(l_r, pa, number, train_files, val_files, W_l1, W_chamfer, W_smooth, test_num):
    
    model = StemTransformerModel(num_queries=number - 2).to(device)
    
    # データセット (Cutout等のAugmentあり)
    train_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=True, file_list=train_files)
    val_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=False, file_list=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_ignore_none, drop_last=True)
    val_loder = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_ignore_none)
    
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, mode='min', factor=0.1, patience=pa, verbose=True)
    early_stop_patience = pa * 2 + 5
    epochs_no_improve = 0

    if test_num == 10:
        model_save_name = f"MST_model2_try10_TF_l1_{W_l1}_c{W_chamfer}_s{W_smooth}.pth"
        graph_save_name = f"loss_model2_try10_TF_l1_{W_l1}_c{W_chamfer}_s{W_smooth}.png"

    train_losses, val_losses, best_val_loss, best_epoch = [], [], float('inf'), -1
    print(f"----- 学習開始 (Transformer, lr={l_r}, 制御点数={number}) -----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            if batch is None: continue
            
            input_tensor, g_pts, c_pts, gt_pc, _, _ = batch
            input_tensor = input_tensor.to(device)
            g_pts = g_pts.to(device) 
            c_pts = c_pts.to(device)
            gt_pc = gt_pc.to(device)
            
            if gt_pc.max() > 2.0:
                norm_factor = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device).view(1, 1, 2)
                gt_pc = gt_pc / norm_factor
                
            
            optim_.zero_grad()
            
            pred_curve_100, pred_control_points, _ = model(input_tensor, g_pts, c_pts)
            
            # Transformer用Loss関数の呼び出し
            loss, l1_val, chamf_val, smooth_val = transformer_curve_loss(
                pred_curve_100, gt_pc, pred_control_points, g_pts,
                w_l1=W_l1, w_chamfer=W_chamfer, w_smooth=W_smooth
            )
            
            loss.backward()
            optim_.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- 評価ループ ---
        model.eval()
        total_test_loss = 0.0
        
        val_l1_sum, val_chamf_sum, val_smooth_sum = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for batch in val_loder:
                if batch is None: continue
                
                input_tensor, g_pts, c_pts, gt_pc, _, _ = batch
                input_tensor = input_tensor.to(device)
                g_pts = g_pts.to(device)
                c_pts = c_pts.to(device)
                gt_pc = gt_pc.to(device)
                
                if gt_pc.max() > 2.0:
                    norm_factor = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device).view(1, 1, 2)
                    gt_pc = gt_pc / norm_factor
                
                
                pred_curve_100, pred_control_points, _ = model(input_tensor, g_pts, c_pts)
                
                loss, l1_val, chamf_val, smooth_val = transformer_curve_loss(
                    pred_curve_100, gt_pc, pred_control_points, g_pts,
                    w_l1=W_l1, w_chamfer=W_chamfer, w_smooth=W_smooth
                )
                
                total_test_loss += loss.item()
                val_l1_sum += l1_val.item()
                val_chamf_sum += chamf_val.item()
                val_smooth_sum += smooth_val.item()
        
        avg_test_loss = total_test_loss / len(val_loder) if len(val_loder) > 0 else 0.0
        val_losses.append(avg_test_loss)
        
        # 内訳の平均を計算
        avg_l1 = val_l1_sum / len(val_loder)
        avg_chamf = val_chamf_sum / len(val_loder)
        avg_smooth = val_smooth_sum / len(val_loder)


        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train: {avg_train_loss:.4f} | Test: {avg_test_loss:.4f}")
        print(f"   ➔ [Test Details] L1: {avg_l1:.4f}, Chamfer: {avg_chamf:.4f}, Smooth: {avg_smooth:.4f}")

        if avg_test_loss < best_val_loss:
            best_val_loss, best_epoch = avg_test_loss, epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_name)
            print(f"✨ New best model saved at epoch {epoch + 1}!")
        else:
            epochs_no_improve += 1
            
        scheduler.step(avg_test_loss)

        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}!")
            break

    print(f"\n-----学習終了-----\nBest model at epoch {best_epoch + 1} (Loss: {best_val_loss:.4f})")
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Test Loss')
    plt.title(f'Transformer Curve Prediction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(graph_save_name)
    plt.close()

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    all_files = [f for f in os.listdir(PT_SPLIT_TRAIN_PATH) if f.endswith('.txt')]
    train_files, val_files = train_test_split(
        all_files, test_size=VALIDATION_SPLIT, random_state=42
    )
    print(f"データ分割: 学習用 {len(train_files)}件, 検証用 {len(val_files)}件")

    ##### 実験10：Transformerモデルの学習 #####
    w_l1 = 100.0
    w_chamfer = 10.0
    w_smooth = 1.0
    
    print(f"\n" + "="*50)
    print(f"🚀 実験10 (Transformer) スタート: L1={w_l1}, Chamfer={w_chamfer}, Smooth={w_smooth}")
    print("="*50)
    
    train_and_evaluate(
        l_r=LEARNING_RATE, 
        pa=PATIENCE, 
        number=10, 
        train_files=train_files, 
        val_files=val_files, 
        W_l1=w_l1,
        W_chamfer=w_chamfer, 
        W_smooth=w_smooth, 
        test_num=10
    )