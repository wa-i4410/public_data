# visualize_chamfer_model.py (元の画像番号表示機能付き)

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
from up_chamfer_data import PositionChamferModel,PositionChamferModel_ResNet,UNetPointCloudModel, PointCloudDataset

# ===== 1. 設定項目 =====

PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'
MAP_PATH_TEST = 'test_prob_maps'
MAP_PATH_TRAIN = 'train_prob_maps'
OUTPUT_DIR1 = "up_chamfer_final_predictions_normal"
OUTPUT_DIR2 = "up_chamfer_final_predictions_resnet"
OUTPUT_DIR3 = "up_chamfer_final_predictions_unet"
OUTPUT_DIR3_TRAIN = "up_chamfer_final_predictions_unet_train"
os.makedirs(OUTPUT_DIR3_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
os.makedirs(OUTPUT_DIR3, exist_ok=True)
IMG_WIDTH, IMG_HEIGHT = 640, 480
BATCH_SIZE = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 読み込むモデル名 ---
MODEL_NAME = 3 # 1: PositionChamferModel, 2: PositionChamferModel_ResNet, 3: UNetPointCloudModel
# --- 予測対象のデータセットを選択 ---
MODE_SELECT = 'test' # 'test' or 'train'

# --- 設定に基づいてパスを自動選択 ---
PT_SPLIT_PATH = PT_SPLIT_TEST_PATH if MODE_SELECT == 'test' else PT_SPLIT_TRAIN_PATH
MAP_PATH = MAP_PATH_TEST if MODE_SELECT == 'test' else MAP_PATH_TRAIN


# ===== 2. ユーティリティ関数 =====
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def draw_spline_with_fixed_endpoints(image_np, start_point, end_point, intermediate_points, color, thickness=2):
    """
    始点と終点を固定し、その間の点を通るスプライン曲線を描画する。
    中間の点は、始点からの距離が近い順に並べ替えられる。
    """
    if len(intermediate_points) > 0:
        dists = np.linalg.norm(intermediate_points - start_point, axis=1)
        sorted_indices = np.argsort(dists)
        sorted_intermediate_points = intermediate_points[sorted_indices]
    else:
        sorted_intermediate_points = intermediate_points
    all_points_np = np.vstack([start_point, sorted_intermediate_points, end_point])
    _, idx = np.unique(all_points_np, axis=0, return_index=True)
    unique_points = all_points_np[np.sort(idx)]
    if len(unique_points) < 3:
        if len(unique_points) >= 2:
            cv2.polylines(image_np, [unique_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
        return image_np
    try:
        k = min(3, len(unique_points) - 1)
        tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=0, k=k)
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck, der=0)
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        cv2.polylines(image_np, [spline_points], isClosed=False, color=color, thickness=thickness)
    except Exception as e:
        print(f"  - スプライン描画エラー: {e}")
        cv2.polylines(image_np, [unique_points.astype(np.int32)], isClosed=False, color=color, thickness=thickness)
    return image_np

# ===== 3. メイン処理 =====
def run_visualization(num=5):
    if MODEL_NAME == 1:
        model = PositionChamferModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        OUTPUT_DIR = OUTPUT_DIR1
        MODEL_WEIGHTS_PATH = "best_up_chamfer_normal_model.pth"
    elif MODEL_NAME == 2:
        model = PositionChamferModel_ResNet(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        OUTPUT_DIR = OUTPUT_DIR2
        MODEL_WEIGHTS_PATH = "best_up_chamfer_resnet_model.pth"
    elif MODEL_NAME == 3:
        model = UNetPointCloudModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT,num_points=num).to(device)
        OUTPUT_DIR = OUTPUT_DIR3_TRAIN if MODE_SELECT == 'train' else OUTPUT_DIR3
        MODEL_WEIGHTS_PATH = f"best_up_chamfer_unet_model_{num}.pth"
    else:
        raise ValueError("無効なモデル名です。1, 2, または 3 を指定してください。")
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_WEIGHTS_PATH}")
        return
    model.eval()

    dataset = PointCloudDataset(PT_SPLIT_PATH, COLOR_PATH, MAP_PATH, GT_POINTCLOUD_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    print(f"予測を開始します。データセット: '{MODE_SELECT}', 結果は '{OUTPUT_DIR}' ディレクトリに保存されます。")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(data_loader, desc="予測中")):
            if batch_data is None: continue
            
            rgba_tensor, g_pts_norm, c_pts_norm, gt_pointcloud, basenames = batch_data
            
            rgba_gpu, g_pts_gpu, c_pts_gpu = rgba_tensor.to(device), g_pts_norm.to(device), c_pts_norm.to(device)
            predicted_pointcloud_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)

            for j in range(rgba_gpu.shape[0]):
                base_filename_txt = basenames[j]
                base_filename_png = base_filename_txt.replace('_points.txt', '.png')
                
                color_image_path = os.path.join(COLOR_PATH, base_filename_png)
                if not os.path.exists(color_image_path): continue
                
                color_img = Image.open(color_image_path).convert("RGB")
                overlay_bgr = cv2.cvtColor(np.array(color_img), cv2.COLOR_RGB2BGR)

                start_px = g_pts_norm[j].squeeze(0).numpy() * [IMG_WIDTH, IMG_HEIGHT]
                end_px = c_pts_norm[j].squeeze(0).numpy() * [IMG_WIDTH, IMG_HEIGHT]
                pred_pc_pixel = predicted_pointcloud_norm[j].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
                gt_pc_pixel = gt_pointcloud[j].numpy()
                gt_pc_pixel = gt_pc_pixel[gt_pc_pixel.sum(axis=1) != 0]

                # 曲線を描画
                overlay_bgr = draw_spline_with_fixed_endpoints(overlay_bgr, start_px, end_px, gt_pc_pixel, color=(0, 0, 255), thickness=3)
                overlay_bgr = draw_spline_with_fixed_endpoints(overlay_bgr, start_px, end_px, pred_pc_pixel, color=(255, 255, 255), thickness=3)
                
                # --- ★★★ ここからが修正箇所 ★★★ ---
                # 1. ファイル名から正規表現で数字を抽出
                image_number = ""
                match = re.search(r'\d+', base_filename_png)
                if match:
                    image_number = match.group(0)

                # 2. 表示するテキストを定義
                display_text = f"No. {image_number}" if image_number else base_filename_png
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_color = (255, 255, 255)  # 白色
                thickness = 2
                padding = 5

                # 3. テキストの大きさを計算し、背景の黒い四角形を描画
                (text_w, text_h), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
                top_left = (padding, padding)
                bottom_right = (padding + text_w + padding, padding + text_h + padding)
                cv2.rectangle(overlay_bgr, top_left, bottom_right, (0, 0, 0), -1)

                # 4. テキストを描画
                text_origin = (padding, padding + text_h)
                cv2.putText(overlay_bgr, display_text, text_origin, font, font_scale, font_color, thickness)
                # --- ★★★ 修正ここまで ★★★ ---

                save_path = os.path.join(OUTPUT_DIR, base_filename_png)
                cv2.imwrite(save_path, overlay_bgr)

    print("完了しました。")

if __name__ == '__main__':
    run_visualization(num=9)