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

from MST_model3 import TomatoStemDataset, StemHybridModel

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
BATCH_SIZE = 4  # メモリ消費に合わせて調整してください
VALIDATION_SPLIT = 0.2

# --- システム設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. U-Netバックボーンの定義 =====

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
class AC_UNetbackborn(nn.Module):
    """
    Attention U-Netの実装
    """
    def __init__(self, in_channels=4, out_channels=1):
        super(AC_UNetbackborn, self).__init__()

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
        
        return out


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
    """
    両端固定モデル用のシンプルなLoss関数。
    GTの向きを成長点スタートに揃えてから、ChamferとL1を計算。
    """
    B = pred_curve.size(0)
    aligned_gts = []
    
    # GTの向きをチェックし、常に成長点スタートに揃える
    for i in range(B):
        dist_gt_start = torch.norm(gt_curve[i, 0] - growth_points[i])
        dist_gt_end = torch.norm(gt_curve[i, -1] - growth_points[i])
        if dist_gt_end < dist_gt_start:
            aligned_gts.append(torch.flip(gt_curve[i], dims=[0]))
        else:
            aligned_gts.append(gt_curve[i])
    aligned_gt = torch.stack(aligned_gts)

    # ① L2 Chamfer
    dist_matrix_sq = torch.sum((pred_curve.unsqueeze(2) - aligned_gt.unsqueeze(1)) ** 2, dim=3)
    dist_matrix = torch.sqrt(dist_matrix_sq + 1e-8)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)
    chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

    # ② Ordered L1 (向きが揃ったので直接計算可能)
    l1_loss = F.l1_loss(pred_curve, aligned_gt)

    # スケール調整
    return 100.0 * (w_chamfer * chamfer_loss + w_l1 * l1_loss)

### 実験6：Chamfer + Smoothness Loss (曲げエネルギー) ###
def curve_loss(pred_curve, gt_curve, control_points, w_chamfer=5.0, w_smooth=1.0, w_uniform=0.0):
    B = pred_curve.size(0)
    
    # ① L2 Chamfer
    dist_matrix_sq = torch.sum((pred_curve.unsqueeze(2) - gt_curve.unsqueeze(1)) ** 2, dim=3)
    dist_matrix = torch.sqrt(dist_matrix_sq + 1e-8)
    min_dist_pred_to_gt, _ = torch.min(dist_matrix, dim=2)
    min_dist_gt_to_pred, _ = torch.min(dist_matrix, dim=1)
    chamfer_loss = torch.mean(min_dist_pred_to_gt) + torch.mean(min_dist_gt_to_pred)

    # ② 【NEW】 Smoothness Loss (Bending Penalty: 曲げエネルギー)
    # 連続する3点 (P_{i-1}, P_i, P_{i+1}) が直線からどれだけズレているか（二次微分）を計算
    bending = control_points[:, 2:, :] - 2 * control_points[:, 1:-1, :] + control_points[:, :-2, :]
    smoothness_loss = torch.mean(torch.norm(bending, dim=2)**2)

    # ③ Edge Length Penalty (Uniformity Loss): 制御点が等間隔に並ぶようにする
    # 隣り合う制御点間の距離を計算
    diff = control_points[:, 1:, :] - control_points[:, :-1, :] # (B, N-1, 2)
    lengths = torch.norm(diff, dim=2) # (B, N-1)
    # 平均距離との二乗誤差を計算
    mean_length = torch.mean(lengths, dim=1, keepdim=True) # (B, 1)
    uniformity_loss = torch.mean((lengths - mean_length)**2)

    # 総合Loss (uniformity を smoothness に置き換え)
    return 100.0 * (w_chamfer * chamfer_loss + w_smooth * smoothness_loss + w_uniform * uniformity_loss)

def structured_heatmap_loss(pred_logits, gt_points, epsilon=1.0, alpha=1.0):
    """
    論文 "Heatmap Regression without Soft-Argmax" に基づく構造化予測Loss。
    
    引数:
        pred_logits: [B, K, H, W] - U-Netが出力したヒートマップ（※Sigmoidをかける前の生の値！）
        gt_points: [B, K, 2] - 正解の座標 (x, y) ※ピクセル単位 (0〜W-1, 0〜H-1)
        epsilon: 温度パラメータ (論文の推奨値は 1.0)
        alpha: マージン(距離ペナルティ)の重み (論文の推奨値は 1.0)
    """
    B, K, H, W = pred_logits.shape
    device = pred_logits.device
    
    # 1. 画像内の全ピクセル座標グリッドを作成 [H, W, 2]
    y_grid, x_grid = torch.meshgrid(torch.arange(H, device=device), 
                                    torch.arange(W, device=device), indexing='ij')
    grid = torch.stack([x_grid, y_grid], dim=-1).float() # (x, y)の順
    grid = grid.view(1, 1, H, W, 2) # ブロードキャスト用に拡張 [1, 1, H, W, 2]
    
    # 2. 正解座標(GT)をブロードキャスト用に拡張 [B, K, 1, 1, 2]
    gt_exp = gt_points.view(B, K, 1, 1, 2)
    
    # 3. マージン Δ の計算 (論文に沿って Smooth L1 距離を使用)
    grid_expanded = grid.expand(B, K, H, W, 2)
    gt_exp_expanded = gt_exp.expand(B, K, H, W, 2)
    
    # サイズが完全に一致した状態でLossを計算
    margin = alpha * torch.sum(F.smooth_l1_loss(grid_expanded, gt_exp_expanded, reduction='none', beta=1.0), dim=-1)
    # margin shape: [B, K, H, W]
    
    # 4. Log-Sum-Exp の内側を計算: ( Δ + F(y_hat) ) / ε
    score = (margin + pred_logits) / epsilon
    
    # 空間次元(H, W)を平坦化して Log-Sum-Exp を計算
    score_flat = score.view(B, K, -1) # [B, K, H*W]
    log_sum_exp = epsilon * torch.logsumexp(score_flat, dim=-1) # [B, K]
    
    # 5. 正解座標 y での予測値 F(y) を抽出
    # GT座標を四捨五入して整数インデックスに変換
    gt_x = torch.clamp(torch.round(gt_points[..., 0]).long(), 0, W - 1)
    gt_y = torch.clamp(torch.round(gt_points[..., 1]).long(), 0, H - 1)
    
    # バッチとキーポイントのインデックスを生成して値をピンポイントで抜く
    b_idx = torch.arange(B, device=device).view(B, 1)
    k_idx = torch.arange(K, device=device).view(1, K)
    F_y = pred_logits[b_idx, k_idx, gt_y, gt_x] # [B, K]
    
    # 6. 最終的な Loss の計算
    loss = log_sum_exp - F_y
    
    # バッチ全体と全ポイント(K個)の平均をとる
    return loss.mean()


# ===== 4. 訓練・評価関数 (MST_train3.py 用) =====
def train_and_evaluate(l_r, pa, number, train_files, val_files, W_chamfer, W_smooth, W_uniform, test_num, beta=50, W_heatmap=1.0):
    unet_backbone = AC_UNetbackborn(in_channels=4, out_channels=number - 2)
    # ★ StemHybridModel (LSTM付き) を使用
    model = StemHybridModel(unet_backbone=unet_backbone, beta=beta).to(device)
    
    train_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=True, file_list=train_files)
    val_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=number, augment=False, file_list=val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn_ignore_none, drop_last=True)
    val_loder = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn_ignore_none)
    
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optim_, mode='min', factor=0.1, patience=pa, verbose=True)
    early_stop_patience = pa * 2 + 5
    epochs_no_improve = 0

    model_save_name = f"MST_model3_t_gt{test_num}.pth"
    graph_save_name = f"loss_graph_t_gt{test_num}.png"

    train_losses, val_losses, best_val_loss, best_epoch = [], [], float('inf'), -1
    print(f"----- 学習開始 (Hybrid Model + MSE Loss) (lr={l_r}, beta={beta}, 制御点数={number}) -----")

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
            
            heatmap_loss = F.mse_loss(torch.sigmoid(pred_heatmaps), gt_heatmaps, reduction='sum') / B
            
            # 総合Loss
            loss = coord_loss + (W_heatmap * heatmap_loss)
            
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
                
                heatmap_loss = F.mse_loss(torch.sigmoid(pred_heatmaps), gt_heatmaps, reduction='sum') / B
                loss = coord_loss + (W_heatmap * heatmap_loss)
                
                total_test_loss += loss.item()
        
        avg_test_loss = total_test_loss / len(val_loder) if len(val_loder) > 0 else 0.0
        val_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_val_loss:
            best_val_loss, best_test_loss = avg_test_loss, avg_test_loss
            best_epoch = epoch
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
    plt.title(f'Hybrid Model (LSTM) + MSE Loss (Points={number})')
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

    
    ##### 実験9：Cutout追加 + Betaの違いによる比較 #####
    w_chamfer = 10.0
    w_smooth = 2.0
    w_uniform = 1.0
    beta_val = 50.0  
    
    print(f"\n" + "="*50)
    print(f"🚀 実験1 スタート: beta={beta_val}, Chamfer={w_chamfer}, Smooth={w_smooth}, Uniform={w_uniform}")
    print("="*50)
    
    train_and_evaluate(
        l_r=LEARNING_RATE, 
        pa=PATIENCE, 
        number=8, 
        train_files=train_files, 
        val_files=val_files, 
        W_chamfer=w_chamfer, 
        W_smooth=w_smooth, 
        W_uniform=w_uniform, 
        test_num=1,
        beta=beta_val # ★ 追加した引数
    )

    train_and_evaluate(
        l_r=LEARNING_RATE, 
        pa=PATIENCE, 
        number=10, 
        train_files=train_files, 
        val_files=val_files, 
        W_chamfer=w_chamfer, 
        W_smooth=w_smooth, 
        W_uniform=w_uniform, 
        test_num=2,
        beta=beta_val # ★ 追加した引数
    )