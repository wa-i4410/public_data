import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import csv
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist # 追加: 距離計算用

from MST_model3 import TomatoStemDataset, StemHybridModel
from MST_train1 import AC_UNetbackborn, custom_collate_fn_ignore_none

# ===== 1. 設定項目 =====

PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
PT_SPLIT_TRAIN_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_POINTCLOUD_PATH = '/home/onozawa/GT_POINTCLOUDS'

IMG_WIDTH, IMG_HEIGHT = 640, 480

W_chamfer = 10.0
W_uniform = 1.0
CONTROL_POINTS = 10
DATA="test"
# # 実験３
# MODEL_WEIGHTS_PATH = f"MST_model1_try3_c{W_chamfer}.pth"
# SAVE_DIR = f"MST_model1_try3_c{W_chamfer}" 

# 実験５
# MODEL_WEIGHTS_PATH = f"MST_model1_try5_c{W_chamfer}.pth"
# SAVE_DIR = f"MST_model1_try5_c{W_chamfer}"    
# SAVE_DIR_TRAIN = f"MST_check_{DATA}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===== 2. 評価指標・補間関数 =====

def sort_points_along_curve(pts):
    """
    【追加関数】
    始点(インデックス0)から終点(最後のインデックス)に向かって、
    間の点群を最も滑らかに繋がる順番に並べ替えます（巡回セールスマン問題の貪欲法的なアプローチ）。
    """
    # 始点と終点は固定されているので取り出しておく
    start_pt = pts[0]
    end_pt = pts[-1]
    
    # 並べ替え対象の中間点
    mid_pts = pts[1:-1]
    
    if len(mid_pts) == 0:
        return pts
        
    sorted_pts = [start_pt]
    remaining_pts = list(mid_pts)
    
    # 現在の点から一番近い点を探して繋いでいく
    current_pt = start_pt
    while remaining_pts:
        # 現在の点と、残っている点すべての距離を計算
        dists = cdist([current_pt], remaining_pts)[0]
        nearest_idx = np.argmin(dists)
        
        # 最も近い点を追加し、残りリストから削除
        nearest_pt = remaining_pts.pop(nearest_idx)
        sorted_pts.append(nearest_pt)
        current_pt = nearest_pt
        
    # 最後に終点を追加
    sorted_pts.append(end_pt)
    
    return np.array(sorted_pts)

def generate_spline_curve(pts, num_evals=500):
    keep_idx = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep_idx[-1]]) > 1e-1:
            keep_idx.append(i)
    
    pts_clean = pts[keep_idx]
    
    if len(pts_clean) < 4:
        return pts
        
    try:
        tck, u = splprep([pts_clean[:, 0], pts_clean[:, 1]], s=0, k=3)
        u_new = np.linspace(0, 1, num_evals)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack((x_new, y_new))
    except Exception as e:
        print(f"Spline generation failed: {e}. Returning linear path.")
        return pts

def calculate_pixel_chamfer_distance(pred_pts, gt_pts):
    diff_pred_to_gt = pred_pts[:, np.newaxis, :] - gt_pts[np.newaxis, :, :]
    dist_pred_to_gt = np.sqrt(np.sum(diff_pred_to_gt ** 2, axis=2))
    min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)
    min_dist_gt_to_pred = np.min(dist_pred_to_gt, axis=0)
    cd = (np.mean(min_dist_pred_to_gt) + np.mean(min_dist_gt_to_pred)) / 2.0
    return cd

def calculate_line_iou(pred_pts, gt_pts, img_w, img_h, thickness=5):
    pred_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    pred_pts_int = np.int32(np.round(pred_pts)).reshape((-1, 1, 2))
    gt_pts_int = np.int32(np.round(gt_pts)).reshape((-1, 1, 2))
    cv2.polylines(pred_mask, [pred_pts_int], isClosed=False, color=1, thickness=thickness)
    cv2.polylines(gt_mask, [gt_pts_int], isClosed=False, color=1, thickness=thickness)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0: return 0.0
    return intersection / union

# ===== 3. メイン推論・可視化処理 =====

# --- 実験６ ---
# def evaluate_and_visualize(W_chamfer=W_chamfer, DATA=DATA):
# --- 実験７ ---
# def evaluate_and_visualize(W_smooth=1.5, DATA=DATA):
# --- 実験８ ---
# def evaluate_and_visualize(W_uniform=0.0, DATA=DATA):
# --- 実験９ ---
def evaluate_and_visualize(beta=50,W_chamfer=W_chamfer, W_smooth=1.0, DATA=DATA):
    unet_backbone = AC_UNetbackborn(in_channels=4, out_channels=CONTROL_POINTS - 2)
    model = StemHybridModel(unet_backbone=unet_backbone).to(device)

    # 実験1
    MODEL_WEIGHTS_PATH = f"MST_model3_tn1.pth"
    SAVE_DIR = f"MST_model3_try1"
    SAVE_HIST_DIR = f"MST_model3_histgram1"
    SAVE_CSV_DIR = f"MST_model3_csv1"
    SAVE_DIR_TRAIN = f"MST_check_{DATA}"

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR_TRAIN, exist_ok=True)
    os.makedirs(SAVE_HIST_DIR, exist_ok=True)
    os.makedirs(SAVE_CSV_DIR, exist_ok=True)
    
    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device, weights_only=True))
        print(f"✅ 重み {MODEL_WEIGHTS_PATH} を読み込みました。")
    else:
        print(f"❌ 重みが見つかりません: {MODEL_WEIGHTS_PATH}")
        return

    model.eval()
    
    if DATA == "train":
        test_dataset = TomatoStemDataset(PT_SPLIT_TRAIN_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=CONTROL_POINTS, augment=False)
    else:
        test_dataset = TomatoStemDataset(PT_SPLIT_TEST_PATH, COLOR_PATH, GT_POINTCLOUD_PATH, num_control_points=CONTROL_POINTS, augment=False)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn_ignore_none)
    
    total_iou = 0.0
    total_cd = 0.0
    count = 0

    results_data = [] 
    iou_list = []

    print("推論と画像の保存を開始します...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if batch is None: continue
            
            input_tensor, g_pts, c_pts, gt_pc, _, basenames = batch
            
            input_tensor = input_tensor.to(device)
            g_pts = g_pts.to(device)
            c_pts = c_pts.to(device)
            basename = basenames[0]
            
            _, pred_control_points, _ = model(input_tensor, g_pts, c_pts)
            
            pred_ctrl_px = pred_control_points[0].cpu().numpy() * np.array([IMG_WIDTH - 1, IMG_HEIGHT - 1])
            
            sorted_pred_ctrl_px = sort_points_along_curve(pred_ctrl_px)
            # ----------------------------------------------------

            gt_pc_np = gt_pc[0].cpu().numpy()
            gt_pc_px = gt_pc_np * np.array([IMG_WIDTH - 1, IMG_HEIGHT - 1]) if gt_pc_np.max() <= 1.0 else gt_pc_np
            
            # 並べ替えた点(sorted_pred_ctrl_px)を使ってスプラインを引く
            pred_curve_smooth = generate_spline_curve(sorted_pred_ctrl_px, num_evals=500)
            gt_pc_smooth = generate_spline_curve(gt_pc_px, num_evals=500)

            cd_val = calculate_pixel_chamfer_distance(pred_curve_smooth, gt_pc_smooth)
            iou_val = calculate_line_iou(pred_curve_smooth, gt_pc_smooth, IMG_WIDTH, IMG_HEIGHT, thickness=5)
            
            total_cd += cd_val
            total_iou += iou_val
            count += 1
            
            # ★ 追加: 計測した値をリストに格納
            results_data.append([basename, iou_val, cd_val])
            iou_list.append(iou_val)

            rgb_tensor = input_tensor[0, :3, :, :].cpu()
            rgb_img = rgb_tensor.permute(1, 2, 0).numpy()
            rgb_img = np.clip(rgb_img, 0, 1)

            plt.figure(figsize=(12, 10))
            plt.imshow(rgb_img)
            
            plt.plot(gt_pc_smooth[:, 0], gt_pc_smooth[:, 1], color='lime', linewidth=2, label='Ground Truth Spline', alpha=0.6)
            plt.scatter(gt_pc_px[:, 0], gt_pc_px[:, 1], color='white', s=15, zorder=3, alpha=0.8, label='GT 100 Points')
            
            plt.plot(pred_curve_smooth[:, 0], pred_curve_smooth[:, 1], color='red', linewidth=2, label='Prediction Spline (from N points)', alpha=0.7)
            
            # 描画する点も並べ替え済みのものを使用
            plt.scatter(sorted_pred_ctrl_px[:, 0], sorted_pred_ctrl_px[:, 1], color='cyan', s=100, zorder=5, edgecolors='black', label=f'Control Points (N={CONTROL_POINTS})')

            plt.title(f"File: {basename}\nIoU: {iou_val:.3f} | Chamfer Dist: {cd_val:.1f} px")
            plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
            plt.axis('off')
            
            if DATA == "train":
                save_path = os.path.join(SAVE_DIR_TRAIN, basename.replace('.txt', '_result.png'))
            else:
                save_path = os.path.join(SAVE_DIR, basename.replace('.txt', '_result.png'))
                
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()

            if DATA == "train" and count >= 10:
                print("\n💡 trainデータの確認のため、最初の10枚で推論を早期終了します。")
                break

    if count > 0:
        # --- ★ 追加: CSVファイルの保存処理 ---
        csv_filename = os.path.join(SAVE_CSV_DIR, f"evaluation_results_u{W_uniform}.csv")
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['File Name', 'IoU', 'Chamfer Distance (px)']) # ヘッダー
            writer.writerows(results_data)
        
        plt.figure(figsize=(10, 6))

        plt.hist(iou_list, bins=20, range=(0.0, 1.0), color='skyblue', edgecolor='black')
        plt.title(f'IoU Distribution (W_uniform={W_uniform})')
        plt.xlabel('IoU')
        plt.ylabel('Frequency (Number of Images)')
        plt.grid(axis='y', alpha=0.75)
        
        hist_filename = os.path.join(SAVE_HIST_DIR, f"iou_histogram_u{W_uniform}.png")
        plt.savefig(hist_filename, bbox_inches='tight', dpi=150)
        plt.close()

        avg_iou = total_iou / count
        avg_cd = total_cd / count
        print("\n" + "="*40)
        print(f"📝 評価完了! (テストデータ {count} 件)")
        print(f"   Average Line IoU    : {avg_iou:.4f} (1.0に近いほど良い)")
        print(f"   Average Chamfer Dist: {avg_cd:.2f} px (0に近いほど良い)")
        print("="*40)
        print(f"画像を '{SAVE_DIR}' フォルダに保存しました。")
        print(f"📊 CSVを '{csv_filename}' に保存しました。")
        print(f"📈 ヒストグラムを '{hist_filename}' に保存しました。")

if __name__ == "__main__":

    # --- 実験1 ---   
    evaluate_and_visualize(beta=50, W_chamfer=10.0, W_smooth=1.0, DATA="test")  