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
import sys

from MST_model4 import TomatoStemDataset, StemCurveModel

# ===== 1. 設定項目 =====
# --- 入力パス ---
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS_v2'

# --- 学習パラメータ ---
IMG_WIDTH, IMG_HEIGHT = 640, 480
LEARNING_RATE = 1e-4
PATIENCE = 10
NUM_EPOCHS = 300
BATCH_SIZE = 4  
VALIDATION_SPLIT = 0.2

# --- システム設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. U-Netバックボーンの定義 (EDI-YOLOの知見を統合した新型) =====

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x) # 茎らしいチャネルを強調（葉を無視）
        x = x * self.sa(x) # 茎がある空間位置を強調
        return x

class MultiDilatedBlock(nn.Module):
    """
    拡張畳み込みを用いて受容野を極大化し、クリップ等の遮蔽を飛び越えて茎を繋ぐ
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=8, dilation=8)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(x)
        d3 = self.conv3(x)
        d4 = self.conv4(x)
        out = torch.cat([d1, d2, d3, d4], dim=1) 
        out = self.bn(out)
        out = self.relu(out)
        return self.fuse(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class AdvancedStemUNet(nn.Module):
    """
    CBAMとMultiDilatedBlockを搭載した高精度U-Netバックボーン
    """
    def __init__(self, in_channels=4, out_channels=8):
        super().__init__()
        
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        # Bottleneck (Dilation Blockを配置して視野を極大化)
        self.down4 = nn.Sequential(nn.MaxPool2d(2), MultiDilatedBlock(512, 1024))
        
        # CBAM (スキップ接続される特徴マップのノイズを浄化)
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Decoder (Upsampling)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x, self.cbam4(x4)], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, self.cbam3(x3)], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, self.cbam2(x2)], dim=1)
        x = self.conv_up3(x)
        
        x = self.up4(x)
        x = torch.cat([x, self.cbam1(x1)], dim=1)
        x = self.conv_up4(x)
        
        logits = self.outc(x)
        return logits


# ===== 3. ユーティリティ & 損失関数 =====

def custom_collate_fn_ignore_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: 
        return None
    
    # リストに変換して要素の書き換えを可能にする
    tensor_items = [list(item[:-1]) for item in batch]
    basenames = [item[-1] for item in batch]
    
    # GT点群(インデックス3)のパディング処理
    max_len = max([t[3].shape[0] for t in tensor_items])
    for i in range(len(tensor_items)):
        gt_pc = tensor_items[i][3]
        if gt_pc.shape[0] < max_len:
            pad = torch.zeros(max_len - gt_pc.shape[0], 2, dtype=gt_pc.dtype)
            tensor_items[i][3] = torch.cat([gt_pc, pad], dim=0)

    collated_tensors = torch.utils.data.dataloader.default_collate(tensor_items)
    collated_tensors.append(basenames)
    return collated_tensors

def simple_hybrid_loss(pred_curve, gt_curve, growth_points, w_chamfer=3.0, w_l1=1.0):
    B = pred_curve.size(0)
    aligned_gts = []
    
    for i in range(B):
        dist_gt_start = torch.norm(gt_curve[i, 0] - growth_points[i])
        dist_gt_end = torch.norm(gt_curve[i, -1] - growth_points[i])
        if dist_gt_end < dist_gt_start:
            aligned_gts.append(torch.flip(gt_curve[i], dims=[0]))
        else:
            aligned_gts.append(gt_curve[i])
    aligned_gt = torch.stack(aligned_gts)

    dist_matrix_sq = torch.sum((pred_curve.unsqueeze(2) - aligned_gt.unsqueeze(1)) ** 2, dim=3)
    dist_matrix = torch.sqrt(dist_matrix_sq + 1e-8)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)
    chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

    l1_loss = F.l1_loss(pred_curve, aligned_gt)

    return 100.0 * (w_chamfer * chamfer_loss + w_l1 * l1_loss)

def curve_loss(pred_curve, gt_curve, control_points, w_chamfer=5.0, w_smooth=1.0, w_uniform=0.0):
    B = pred_curve.size(0)
    
    # ① L2 Chamfer
    dist_matrix_sq = torch.sum((pred_curve.unsqueeze(2) - gt_curve.unsqueeze(1)) ** 2, dim=3)
    dist_matrix = torch.sqrt(dist_matrix_sq + 1e-8)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)
    chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

    # ② Smoothness Loss (曲げエネルギー)
    bending = control_points[:, 2:, :] - 2 * control_points[:, 1:-1, :] + control_points[:, :-2, :]
    smoothness_loss = torch.mean(torch.norm(bending, dim=2)**2)

    # ③ Edge Length Penalty (Uniformity Loss)
    diff = control_points[:, 1:, :] - control_points[:, :-1, :] 
    lengths = torch.norm(diff, dim=2) 
    mean_length = torch.mean(lengths, dim=1, keepdim=True) 
    uniformity_loss = torch.mean((lengths - mean_length)**2)

    return 100.0 * (w_chamfer * chamfer_loss + w_smooth * smoothness_loss + w_uniform * uniformity_loss)

def structured_heatmap_loss(pred_logits, gt_points, epsilon=1.0, alpha=1.0):
    B, K, H, W = pred_logits.shape
    device = pred_logits.device
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                    torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([x_grid, y_grid], dim=-1).float() 
    grid = grid.view(1, 1, H, W, 2) 
    
    gt_exp = gt_points.view(B, K, 1, 1, 2)
    
    grid_expanded = grid.expand(B, K, H, W, 2)
    gt_exp_expanded = gt_exp.expand(B, K, H, W, 2)
    
    margin = alpha * torch.sum(F.smooth_l1_loss(grid_expanded, gt_exp_expanded, reduction='none', beta=1.0), dim=-1)
    
    score = (margin + pred_logits) / epsilon
    
    score_flat = score.view(B, K, -1) 
    log_sum_exp = epsilon * torch.logsumexp(score_flat, dim=-1) 
    
    gt_x = torch.clamp(torch.round(gt_points[..., 0]).long(), 0, W - 1)
    gt_y = torch.clamp(torch.round(gt_points[..., 1]).long(), 0, H - 1)
    
    b_idx = torch.arange(B, device=device).view(B, 1)
    k_idx = torch.arange(K, device=device).view(1, K)
    F_y = pred_logits[b_idx, k_idx, gt_y, gt_x] 
    
    loss = log_sum_exp - F_y
    
    return loss.mean()


# ===== 4. 訓練・評価関数 =====

def train_and_evaluate(l_r, pa, number, train_files, val_files, W_chamfer, W_smooth, W_uniform, test_num, beta=50, W_heatmap=1000.0):
    
    # ★ 新しい AdvancedStemUNet に変更
    unet_backbone = AdvancedStemUNet(in_channels=4, out_channels=number - 2)
    model = StemCurveModel(unet_backbone=unet_backbone, beta=beta).to(device)
    
    train_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=True, file_list=train_files)
    val_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=False, file_list=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_ignore_none, drop_last=True)
    val_loder = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_ignore_none)
    
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, mode='min', factor=0.1, patience=pa)
    early_stop_patience = pa * 2 + 5
    epochs_no_improve = 0

    W_h = 1.0

    model_save_name = f"MST_model4_t_gt{test_num}.pth"
    graph_save_name = f"loss_graph4_t_gt{test_num}.png"

    train_losses, val_losses, best_val_loss, best_epoch = [], [], float('inf'), -1
    print(f"----- 学習開始 (lr={l_r}, beta={beta}, 制御点数={number}, EarlyStop Patience={early_stop_patience}) -----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"):
            if batch is None: continue
            
            input_tensor, g_pts, c_pts, gt_pc, gt_heatmaps, _ = batch
            input_tensor = input_tensor.to(device)
            g_pts = g_pts.to(device) 
            c_pts = c_pts.to(device)
            gt_pc = gt_pc.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            
            if gt_pc.max() > 2.0:
                norm_factor = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device).view(1, 1, 2)
                gt_pc = gt_pc / norm_factor
            
            optim_.zero_grad()
            
            pred_curve_100, pred_control_points, pred_heatmaps = model(input_tensor, g_pts, c_pts)
            
            coord_loss = curve_loss(pred_curve_100, gt_pc, pred_control_points, w_chamfer=W_chamfer, w_uniform=W_uniform, w_smooth=W_smooth) 
            B = input_tensor.size(0)
            
            if test_num <= 2:
                W_h =10.0
                num_intermediate = pred_heatmaps.size(1)
                
                total_points = num_intermediate + 2
                step = (gt_pc.size(1) - 1) / (total_points - 1)
                indices = torch.round(torch.arange(1, num_intermediate + 1, device=device) * step).long()
                gt_intermediate_norm = gt_pc[:, indices, :] 
                
                scale = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device)
                gt_points_px = gt_intermediate_norm * scale
                
                heatmap_loss = structured_heatmap_loss(pred_heatmaps, gt_points_px)


            else:
                W_h =10.0
                num_intermediate = pred_heatmaps.size(1)
                
                total_points = num_intermediate + 2
                step = (gt_pc.size(1) - 1) / (total_points - 1)
                indices = torch.round(torch.arange(1, num_intermediate + 1, device=device) * step).long()
                gt_intermediate_norm = gt_pc[:, indices, :] 
                
                scale = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device)
                gt_points_px = gt_intermediate_norm * scale
                
                heatmap_loss = structured_heatmap_loss(pred_heatmaps, gt_points_px)
                
            loss = coord_loss + (W_h * heatmap_loss)
            
            loss.backward()
            optim_.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- 評価ループ ---
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for batch in val_loder:
                if batch is None: continue
                
                input_tensor, g_pts, c_pts, gt_pc, gt_heatmaps, _ = batch
                input_tensor = input_tensor.to(device)
                g_pts = g_pts.to(device)
                c_pts = c_pts.to(device)
                gt_pc = gt_pc.to(device)
                gt_heatmaps = gt_heatmaps.to(device)
                
                if gt_pc.max() > 2.0:
                    norm_factor = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device).view(1, 1, 2)
                    gt_pc = gt_pc / norm_factor
                
                pred_curve_100, pred_control_points, pred_heatmaps = model(input_tensor, g_pts, c_pts)
                
                coord_loss = curve_loss(pred_curve_100, gt_pc, pred_control_points, w_chamfer=W_chamfer, w_uniform=W_uniform, w_smooth=W_smooth) 
                B = input_tensor.size(0)
                
                if test_num <= 2:
                    W_h =10.0
                    num_intermediate = pred_heatmaps.size(1)
                    # ★修正: 正しい等間隔インデックスの計算
                    total_points = num_intermediate + 2
                    step = (gt_pc.size(1) - 1) / (total_points - 1)
                    indices = torch.round(torch.arange(1, num_intermediate + 1, device=device) * step).long()
                    gt_intermediate_norm = gt_pc[:, indices, :] 
                    
                    scale = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device)
                    gt_points_px = gt_intermediate_norm * scale
                    
                    heatmap_loss = structured_heatmap_loss(pred_heatmaps, gt_points_px)


                else:
                    W_h =10.0
                    num_intermediate = pred_heatmaps.size(1)
                    # ★修正: 正しい等間隔インデックスの計算
                    total_points = num_intermediate + 2
                    step = (gt_pc.size(1) - 1) / (total_points - 1)
                    indices = torch.round(torch.arange(1, num_intermediate + 1, device=device) * step).long()
                    gt_intermediate_norm = gt_pc[:, indices, :] 
                    
                    scale = torch.tensor([IMG_WIDTH - 1, IMG_HEIGHT - 1], device=device)
                    gt_points_px = gt_intermediate_norm * scale
                    
                    heatmap_loss = structured_heatmap_loss(pred_heatmaps, gt_points_px)
                    
                
                loss = coord_loss + (W_h * heatmap_loss)
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(val_loder) if len(val_loder) > 0 else 0.0
        val_losses.append(avg_test_loss)

        current_lr = optim_.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_val_loss:
            best_val_loss, best_epoch = avg_test_loss, epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_name)
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")
            
        scheduler.step(avg_test_loss)

        if epochs_no_improve >= early_stop_patience:
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}!")
            break

    print(f"\n-----学習終了-----\nBest model at epoch {best_epoch + 1} (Loss: {best_val_loss:.4f})")
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Test Loss')
    
    loss_type = "MSE Loss" if test_num <= 3 else "Structured Heatmap Loss"
    plt.title(f'Hybrid Chamfer+Uniform Loss ({loss_type}, Points={number})')
    
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

    train_and_evaluate(
                l_r=LEARNING_RATE, 
                pa=PATIENCE, 
                number=15, 
                train_files=train_files, 
                val_files=val_files, 
                W_chamfer=10.0, 
                W_smooth=2.0, 
                W_uniform=10.0, 
                test_num=1,
                beta=1.0
            )

    
    train_and_evaluate(
                l_r=LEARNING_RATE, 
                pa=PATIENCE, 
                number=15, 
                train_files=train_files, 
                val_files=val_files, 
                W_chamfer=10.0, 
                W_smooth=2.0, 
                W_uniform=15.0, 
                test_num=2,
                beta=1.0
            )


    train_and_evaluate(
                l_r=LEARNING_RATE, 
                pa=PATIENCE, 
                number=15, 
                train_files=train_files, 
                val_files=val_files, 
                W_chamfer=10.0, 
                W_smooth=2.0, 
                W_uniform=20.0, 
                test_num=3,
                beta=1.0
            )
