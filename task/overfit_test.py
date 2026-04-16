# overfit_test.py (可視化機能付き)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from coordinate_model import PositionModel, PlotDataset # coordinate_model.pyからインポート
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 # 描画のためにOpenCVをインポート

# ===== 1. 設定項目 =====
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
PT2_PATH = '/home/onozawa/optimized_coords'

# ★★★ テストに使用する画像（どれか一つ、存在するものを指定） ★★★
TEST_IMAGE_NAME = "10079.png"

# --- パラメータ ---
IMG_WIDTH = 640
IMG_HEIGHT = 480
LEARNING_RATE = 1e-4
NUM_STEPS = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. ヘルパー関数 =====

def calculate_bezier_points_numpy(p0, p1, p2, p3, num_points=100):
    """ 4点からベジェ曲線を構成する点のNumPy配列を計算する """
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    points = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return points.astype(np.int32)

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# ===== 3. メイン処理 =====

def run_overfit_test():
    """たった1つのデータでモデルが学習できるか（過学習できるか）をテストする"""
    
    # --- 1. モデルと損失関数の準備 ---
    model = PositionModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    optim_ = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    l1_loss_fn = nn.L1Loss()
    
    # --- 2. たった1つのデータを読み込む ---
    try:
        # データセットから指定したデータを探す
        temp_dataset = PlotDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, pt2_path=PT2_PATH)
        idx = temp_dataset.pt_baselist.index(TEST_IMAGE_NAME.replace('.png', '_points.txt'))
        single_data = temp_dataset[idx]
        if single_data is None:
            print(f"エラー: {TEST_IMAGE_NAME} のデータが読み込めませんでした。")
            return
        
        rgba_tensor, g_pts, c_pts, target_coords = [d.unsqueeze(0).to(device) for d in single_data]
        
        # ★★★ 可視化のために背景画像を読み込んでおく ★★★
        background_image_path = os.path.join(COLOR_PATH, TEST_IMAGE_NAME)
        background_image = Image.open(background_image_path).convert("RGB")
        
        print(f"テストデータ '{TEST_IMAGE_NAME}' を正常に読み込みました。")
    except (ValueError, IndexError):
        print(f"エラー: {TEST_IMAGE_NAME} がテストセットに見つかりません。")
        return

    # --- 3. 過学習ループ ---
    model.train()
    losses = []
    print("\n過学習テストを開始します...")
    for step in tqdm(range(NUM_STEPS), desc="Overfitting Test"):
        optim_.zero_grad()
        predicted_norm = model(rgba_tensor, g_pts, c_pts)
        loss = l1_loss_fn(predicted_norm, target_coords)
        loss.backward()
        optim_.step()
        losses.append(loss.item())

    print("過学習テストが完了しました。")

    # --- 4. 結果のプロット ---
    final_loss = losses[-1]
    print(f"最終的な損失: {final_loss:.8f}")

    if final_loss < 0.01:
        print("\n✅ 成功: モデルは正常に学習できています！")
    else:
        print("\n❌ 失敗: モデルが学習していません。コードに根本的なバグが存在します。")

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Overfitting Test Loss")
    plt.xlabel("Step")
    plt.ylabel("L1 Loss")
    plt.grid(True)
    plt.savefig("overfit_test_loss.png")
    plt.close()
    print("損失グラフを 'overfit_test_loss.png' に保存しました。")
    
    # ===== ★★★ 5. 最終的な予測結果の可視化 ★★★ =====
    print("最終的な予測結果をオーバーレイ画像として保存します...")
    model.eval()
    with torch.no_grad():
        # 最終的な予測座標を取得
        final_prediction_norm = model(rgba_tensor, g_pts, c_pts)

        # 全ての座標をピクセル単位に変換
        start_px = g_pts[0].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        end_px = c_pts[0].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        
        pred_p1_px = final_prediction_norm[0, 0:2].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        pred_p2_px = final_prediction_norm[0, 2:4].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        
        gt_p1_px = target_coords[0, 0:2].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        gt_p2_px = target_coords[0, 2:4].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]

        # オーバーレイ用の画像を作成
        overlay_np = np.array(background_image)

        # 正解のベジェ曲線（赤）を描画
        gt_bezier_points = calculate_bezier_points_numpy(start_px, gt_p1_px, gt_p2_px, end_px)
        cv2.polylines(overlay_np, [gt_bezier_points], isClosed=False, color=(255, 0, 0), thickness=2)

        # 予測のベジェ曲線（白）を描画
        pred_bezier_points = calculate_bezier_points_numpy(start_px, pred_p1_px, pred_p2_px, end_px)
        cv2.polylines(overlay_np, [pred_bezier_points], isClosed=False, color=(255, 255, 255), thickness=2)
        
        # 画像を保存
        Image.fromarray(overlay_np).save("overfit_test_result.png")
        print("オーバーレイ画像を 'overfit_test_result.png' に保存しました。")


if __name__ == '__main__':
    run_overfit_test()