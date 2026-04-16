# visualize_chamfer_model.py (始点・終点の一致を保証する最終版)

import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
from scipy.interpolate import splprep, splev

# 以前のファイルからモデルとデータセットのクラスをインポート
from train_chamfer import PositionChamferModel,PositionChamferModel_ResNet,UNetPointCloudModel, PointCloudDataset 

# ===== 1. 設定項目 =====
MODEL_WEIGHTS_PATH = "best_chamfer_model.pth" 
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'
OUTPUT_DIR1 = "chamfer_final_predictions_normal"
OUTPUT_DIR2 = "chamfer_final_predictions_resnet"
OUTPUT_DIR3 = "chamfer_final_predictions_unet"
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
os.makedirs(OUTPUT_DIR3, exist_ok=True)
IMG_WIDTH, IMG_HEIGHT = 640, 480
BATCH_SIZE = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 読み込むモデル名  ---
MODEL_NAME = 3 # 1: PositionChamferModel, 2: PositionChamferModel_ResNet, 3: UNetPointCloudModel

# ===== 2. ユーティリティ関数 =====
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# ===== ★★★ ここからが修正箇所 ★★★ =====
def draw_spline_with_fixed_endpoints(image_np, start_point, end_point, intermediate_points, color, thickness=2):
    """
    始点と終点を固定し、その間の点を通るスプライン曲線を描画する
    """
    # 描画に使う点のリストを [始点] + [中間の点] + [終点] の順で作成
    all_points_np = np.vstack([start_point, intermediate_points, end_point])
    
    # 重複する点を削除（安全のため）
    unique_points = np.unique(all_points_np, axis=0)
    
    # 点が少ない場合は直線で結ぶ
    if len(unique_points) < 4:
        cv2.polylines(image_np, [unique_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
        return image_np
    
    try:
        # スプライン曲線を計算
        tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=0, k=3)
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck, der=0)
        
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        cv2.polylines(image_np, [spline_points], isClosed=False, color=color, thickness=thickness)
    except Exception as e:
        print(f"  - スプライン描画エラー: {e}")
        # エラー時は点を結ぶだけでも、始点と終点は正しい
        cv2.polylines(image_np, [unique_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
        
    return image_np

# ===== 3. メイン処理 =====
def run_visualization():
    if MODEL_NAME == 1:
        model = PositionChamferModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        OUTPUT_DIR = OUTPUT_DIR1
    elif MODEL_NAME == 2:
        model = PositionChamferModel_ResNet(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        OUTPUT_DIR = OUTPUT_DIR2
    elif MODEL_NAME == 3:
        model = UNetPointCloudModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        OUTPUT_DIR = OUTPUT_DIR3
    else:
        raise ValueError("無効なモデル名です。1, 2, または 3 を指定してください。")
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_WEIGHTS_PATH}")
        return
    model.eval()

    test_dataset = PointCloudDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, gt_pointcloud_path=GT_POINTCLOUD_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    print(f"予測を開始します。結果は '{OUTPUT_DIR}' ディレクトリに保存されます。")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="予測中")):
            if batch_data is None: continue
            
            rgba_tensor, g_pts_norm, c_pts_norm, gt_pointcloud = batch_data
            
            rgba_gpu, g_pts_gpu, c_pts_gpu = rgba_tensor.to(device), g_pts_norm.to(device), c_pts_norm.to(device)
            predicted_pointcloud_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)

            base_filename = test_dataset.pt_baselist[i].replace('_points.txt', '.png')
            color_image_path = os.path.join(COLOR_PATH, base_filename)
            if not os.path.exists(color_image_path): continue
            
            color_img = Image.open(color_image_path).convert("RGB")
            overlay_bgr = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

            # --- 座標をNumPy配列に変換 ---
            start_px = g_pts_norm.squeeze(0).numpy() * [IMG_WIDTH, IMG_HEIGHT]
            end_px = c_pts_norm.squeeze(0).numpy() * [IMG_WIDTH, IMG_HEIGHT]
            pred_pc_pixel = predicted_pointcloud_norm.squeeze(0).cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
            gt_pc_pixel = gt_pointcloud.squeeze(0).numpy()
            gt_pc_pixel = gt_pc_pixel[gt_pc_pixel.sum(axis=1) != 0]

            # --- 曲線を描画 ---
            # 正解(GT)のスプラインを赤色で描画
            overlay_bgr = draw_spline_with_fixed_endpoints(overlay_bgr, start_px, end_px, gt_pc_pixel, color=(0, 0, 255))
            # 予測したスプラインを白色で描画
            overlay_bgr = draw_spline_with_fixed_endpoints(overlay_bgr, start_px, end_px, pred_pc_pixel, color=(255, 255, 255))
            
            save_path = os.path.join(OUTPUT_DIR, base_filename)
            cv2.imwrite(save_path, overlay_bgr)

    print("完了しました。")

if __name__ == '__main__':
    run_visualization()