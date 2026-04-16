# evaluate_ade.py (for train_up_chamfer.py)

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv
from scipy.interpolate import splprep, splev

# ★★★ 座標回帰モデルとデータセットをインポート ★★★
from up_chamfer_data import (
    PointCloudDataset, 
    PositionChamferModel, 
    PositionChamferModel_ResNet, 
    UNetPointCloudModel
)

# ===== 1. 設定項目 =====
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'
MAP_PATH_TEST = 'test_prob_maps'
OUTPUT_CSV_PATH = "evaluation_ade_results.csv"

# ★ 出力ディレクトリ名をモデルごとに変更
OUTPUT_DIR1 = "up_chamfer_final_predictions_normal"
OUTPUT_DIR2 = "up_chamfer_final_predictions_resnet"
OUTPUT_DIR3 = "up_chamfer_final_predictions_unet"
os.makedirs(OUTPUT_DIR1, exist_ok=True)
os.makedirs(OUTPUT_DIR2, exist_ok=True)
os.makedirs(OUTPUT_DIR3, exist_ok=True)

IMG_WIDTH, IMG_HEIGHT = 640, 480
BATCH_SIZE = 1
NUM_SAMPLING_POINTS = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★ 評価したいモデルの番号を指定 (1, 2, or 3) ★★★
MODEL_NAME = 2 # 1: PositionChamferModel, 2: PositionChamferModel_ResNet, 3: UNetPointCloudModel 


# ===== 2. ユーティリティ関数 =====
def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    # バッチ内のgt_pointcloudの長さを揃えるパディング処理
    if not batch: return None
    tensor_items = [item[:-1] for item in batch]
    basenames = [item[-1] for item in batch]
    max_len = max([t[3].shape[0] for t in tensor_items if t[3].shape[0] > 0] or [0])
    if max_len == 0: return None # 全て空の場合
    for i in range(len(tensor_items)):
        rgba, g, c, gt_pc = tensor_items[i]
        if gt_pc.shape[0] < max_len:
            pad = torch.zeros(max_len - gt_pc.shape[0], 2, dtype=gt_pc.dtype)
            tensor_items[i] = (rgba, g, c, torch.cat([gt_pc, pad], dim=0))
    collated_tensors = torch.utils.data.dataloader.default_collate(tensor_items)
    collated_tensors.append(basenames)
    return collated_tensors

def generate_spline_points(start_point, end_point, intermediate_points, num_points):
    """
    始点・終点と中間点からスプライン曲線を作成し、指定された数の等間隔な点をサンプリングする
    """
    all_points_np = np.vstack([start_point, intermediate_points, end_point])
    unique_points = np.unique(all_points_np, axis=0)
    
    if len(unique_points) < 2: return np.array([start_point] * num_points)
    if len(unique_points) < 4:
        x = np.linspace(unique_points[0, 0], unique_points[-1, 0], num_points)
        y = np.linspace(unique_points[0, 1], unique_points[-1, 1], num_points)
        return np.vstack((x, y)).T
    try:
        tck, u = splprep([unique_points[:, 0], unique_points[:, 1]], s=0, k=min(3, len(unique_points)-1))
        u_new = np.linspace(u.min(), u.max(), num_points)
        x_new, y_new = splev(u_new, tck, der=0)
        return np.vstack((x_new, y_new)).T
    except Exception:
        x = np.linspace(unique_points[0, 0], unique_points[-1, 0], num_points)
        y = np.linspace(unique_points[0, 1], unique_points[-1, 1], num_points)
        return np.vstack((x, y)).T


# ===== 3. メイン処理 (修正版) =====
def run_evaluation():
    # ★★★ モデル選択ロジックを復活 ★★★
    if MODEL_NAME == 1:
        model = PositionChamferModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        MODEL_WEIGHTS_PATH = "best_up_chamfer_normal_model.pth"
    elif MODEL_NAME == 2:
        model = PositionChamferModel_ResNet(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        MODEL_WEIGHTS_PATH = "best_up_chamfer_resnet_model.pth"
    elif MODEL_NAME == 3:
        model = UNetPointCloudModel(img_width=IMG_WIDTH, img_height=IMG_HEIGHT).to(device)
        MODEL_WEIGHTS_PATH = "best_up_chamfer_unet_model.pth"
    else:
        raise ValueError("無効なモデル名です。1, 2, または 3 を指定してください。")

    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_WEIGHTS_PATH}")
        return
    model.eval()

    test_dataset = PointCloudDataset(PT_SPLIT_TEST_PATH, COLOR_PATH, MAP_PATH_TEST, GT_POINTCLOUD_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

    results = []
    print(f"評価を開始します (モデル: {model.__class__.__name__})。結果は '{OUTPUT_CSV_PATH}' に保存されます。")
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="評価中")):
            if batch_data is None: continue
            
            rgba_tensor, g_pts_norm, c_pts_norm, gt_pointcloud, basenames = batch_data
            
            # ★★★ ここからが修正箇所 ★★★
            rgba_gpu = rgba_tensor.to(device)
            g_pts_gpu = g_pts_norm.to(device)
            c_pts_gpu = c_pts_norm.to(device)
            
            # 1. 座標回帰モデルのフォワードパスを実行
            predicted_pointcloud_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)
            # ★★★ 修正箇所ここまで ★★★

            for j in range(rgba_gpu.shape[0]):
                base_filename_txt = basenames[j]
                base_filename_png = base_filename_txt.replace('_points.txt', '.png')
                
                start_px = g_pts_norm[j].numpy() * [IMG_WIDTH, IMG_HEIGHT]
                end_px = c_pts_norm[j].numpy() * [IMG_WIDTH, IMG_HEIGHT]
                
                # ★★★ 予測された正規化座標をピクセル座標に変換 ★★★
                pred_pc_pixel = predicted_pointcloud_norm[j].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
                
                gt_pc_pixel = gt_pointcloud[j].numpy()
                gt_pc_pixel = gt_pc_pixel[gt_pc_pixel.sum(axis=1) != 0]

                gt_spline_points = generate_spline_points(start_px, end_px, gt_pc_pixel, num_points=NUM_SAMPLING_POINTS)
                pred_spline_points = generate_spline_points(start_px, end_px, pred_pc_pixel, num_points=NUM_SAMPLING_POINTS)

                distances = np.linalg.norm(gt_spline_points - pred_spline_points, axis=1)
                ade = np.mean(distances)

                results.append({'image_name': base_filename_png, 'ade': ade})

    if results:
        with open(OUTPUT_CSV_PATH, 'w', newline='') as csvfile:
            fieldnames = ['image_name', 'ade']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"評価が完了し、結果を {OUTPUT_CSV_PATH} に保存しました。")
        # ADEの平均値を計算して表示
        average_ade = np.mean([res['ade'] for res in results])
        print(f"全テスト画像の平均ADE: {average_ade:.4f} ピクセル")
    else:
        print("評価対象のデータが見つかりませんでした。")

if __name__ == '__main__':
    run_evaluation()