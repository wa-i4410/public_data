import os
import matplotlib.pyplot as plt 
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
# newmodel.py から UNet と UNetDataset をインポート
from newmodel import AC_UNet, UNetDataset 
from tqdm import tqdm
import torchvision.transforms.functional as TF
import time
import random
import numpy as np
import torch.nn.functional as F

# ===== 初期設定 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PT_SPLIT_TRAIN_PATH =('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train')
PT_SPLIT_TEST_PATH = ('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test')
COLOR_PATH = ('/home/onozawa/CALORsample')
GT_PATH = ('/home/onozawa/GT2dspline')
BATCH_SIZE = 4 # GPUメモリに応じて調整


PT_PATH = ('/home/onozawa/savepoints')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

seed_everything()
def train_unet(l_r, NUM_EPOCHS, sig, fine_tune_from=None, sigma_decay_epochs=50):
    # 1. モデル、データセット、データローダーの準備
    model = AC_UNet(in_channels=4, out_channels=1).to(device)
    if fine_tune_from and os.path.exists(fine_tune_from):
        model.load_state_dict(torch.load(fine_tune_from, weights_only=True))
        print(f"ファインチューニングを開始します。ロードしたモデル: {fine_tune_from}")
    else:
        print("新規モデルの訓練を開始します。")
    train_dataset = UNetDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_PATH, augment=True)
    test_dataset = UNetDataset(PT_SPLIT_TEST_PATH, COLOR_PATH, GT_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    
    train_losses, test_losses = [], []
    best_test_loss = float('inf')

    initial_sigma = sig       # 初期ぼかしの強さ
    alternative = initial_sigma     # 現在のsigma値
    sigma_decay_rate = 0.9 # 減衰率 (0.9倍にしていく)

    epochs = NUM_EPOCHS 

    print(f"----- U-Net訓練開始 (lr={l_r}) -----")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        if (epoch + 1) % sigma_decay_epochs == 0 and alternative > 0.01: # 0.01より小さくならないように
            alternative *= sigma_decay_rate
            print(f"Epoch {epoch + 1}: Sigma updated to {alternative:.3f}")
        # --- 訓練ループ ---
        model.train()
        total_train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(inputs)

            if alternative > 0.0:
                # ぼかしを適用
                blurred_predictions = TF.gaussian_blur(predictions, kernel_size=5, sigma=alternative)
                blurred_targets = TF.gaussian_blur(targets, kernel_size=5, sigma=alternative)
                
                blurred_predictions_clamped = torch.clamp(blurred_predictions, 0, 1)
                blurred_targets_clamped = torch.clamp(blurred_targets, 0, 1)
            else:
                blurred_predictions_clamped = torch.clamp(predictions, 0, 1)
                blurred_targets_clamped = torch.clamp(targets, 0, 1)
                

            diceloss = dice_loss(blurred_predictions_clamped, blurred_targets_clamped)
            loss_bce = F.binary_cross_entropy(blurred_predictions_clamped, blurred_targets_clamped)
            loss = diceloss + loss_bce
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- 評価ループ ---
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                if alternative > 0.0:
                # ぼかしを適用
                    blurred_predictions = TF.gaussian_blur(predictions, kernel_size=5, sigma=alternative)
                    blurred_targets = TF.gaussian_blur(targets, kernel_size=5, sigma=alternative)
                    
                    blurred_predictions_clamped = torch.clamp(blurred_predictions, 0, 1)
                    blurred_targets_clamped = torch.clamp(blurred_targets, 0, 1)
                else:
                    blurred_predictions_clamped = torch.clamp(predictions, 0, 1)
                    blurred_targets_clamped = torch.clamp(targets, 0, 1)

                # ぼかした後のマスクで損失を計算
                diceloss = dice_loss(blurred_predictions_clamped, blurred_targets_clamped)
                loss_bce = F.binary_cross_entropy(blurred_predictions_clamped, blurred_targets_clamped)
                loss = diceloss + loss_bce
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f} | Time: {epoch_duration:.2f}s | Sigma: {alternative:.3f}")
        
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            
            save_filename = f"model_sig_{sig}.pth"
            if fine_tune_from:
                # 例: model_trained_on_3.0_finetuned_to_1.5.pth のような名前
                base_sigma = os.path.basename(fine_tune_from).split('sig_')[1].replace('.pth','')
                save_filename = f"model_base_{base_sigma}_tuned_{sig}.pth"
            # save_filename = f"model_sig_{sig}.pth"
            torch.save(model.state_dict(), save_filename)
            print(f"✨ New best model saved as '{save_filename}' with test loss: {best_test_loss:.4f}")
        
        scheduler.step(avg_test_loss)

        if epoch == 50 and avg_train_loss >= 1.1:
            print(f"損失が改善しないため、このパラメータでの学習を打ち切ります。")
            return avg_train_loss

    # (ここにグラフ描画・保存のコードを追加)
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss', marker='o', linestyle='-')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss', marker='o', linestyle='-')
    plt.title(f'Training & Test Loss (lr={l_r}, sigma={sig})')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    
    plot_output_dir = "training_plots_with_test_UNet"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filename = os.path.join(plot_output_dir, f"loss_lr_{l_r}_sigma_{sig}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"訓練とテストの損失グラフを '{plot_filename}' に保存しました。")

if __name__ == "__main__":
    
    # ===== ステップ1: ベースモデルの学習 =====
    print("--- Step 1: Training Base Model (sigma=3.0) ---")
    base_sigma = 2.0
    base_lr = 5e-4
    base_epochs = 150
    # ベースモデルのファイル名を定義
    base_model_path = f"model_sig_{base_sigma}.pth"

    train_unet(l_r=base_lr, NUM_EPOCHS=base_epochs, sig=base_sigma, sigma_decay_epochs=50)
    
    fine_tune_lr = 1e-5 
    fine_tune_epochs = 200
    train_unet(l_r=fine_tune_lr, 
                NUM_EPOCHS=fine_tune_epochs, 
                sig=1.0, 
                fine_tune_from=base_model_path,
                sigma_decay_epochs=10)
    """
    # ===== ステップ2: ファインチューニングの逐次実行 =====
    fine_tune_sigma_list = [1.0, 0.5]
    last_model_path = base_model_path # ステップ1で保存したモデルから開始

    for i, current_sigma in enumerate(fine_tune_sigma_list):
        print(f"\n--- Step {i + 2}: Fine-tuning (sigma={current_sigma}) ---")
        
        fine_tune_lr = 1e-5 
        fine_tune_epochs = 100
        
        # 前のステップで保存した最良のモデルを読み込んで、追加学習を実行
        train_unet(l_r=fine_tune_lr, 
                NUM_EPOCHS=fine_tune_epochs, 
                sig=current_sigma, 
                fine_tune_from=base_model_path,
                sigma_decay_epochs=10) # ファインチューニング時は更新頻度を早める
    """
    
    print("\nすべてのファインチューニングが完了しました。")