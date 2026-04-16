# visualize_final_model.py

import os
import torch
import numpy as np
from PIL import Image
import cv2 # 描画のためにOpenCVをインポート
from tqdm import tqdm
from torch.utils.data import DataLoader

# モデルとデータセットのクラスをインポート
from newmodel import ExpectCurveDataset
from coordinate_model import PositionModel

# ===== 1. 設定項目 =====
# ★★★ 評価に使用する、ファインチューニング済みのモデルパス ★★★
MODEL_WEIGHTS_PATH = "best_finetuned_model.pth" 

# --- 入力パス ---
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_PATH = '/home/onozawa/GT2dspline'

# --- 出力パス ---
OUTPUT_DIR = "final_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 基本パラメータ ---
IMG_WIDTH = 640
IMG_HEIGHT = 480
BATCH_SIZE = 16 # メモリに応じて調整
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 2. ユーティリティ関数 =====

def custom_collate_fn(batch):
    """データローダーがNoneを返す場合にバッチから除外する"""
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

def calculate_bezier_points_numpy(p0, p1, p2, p3, num_points=100):
    """ 4点からベジェ曲線を構成する点のNumPy配列を計算する """
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    points = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return points.astype(np.int32)

# ===== 3. メイン処理 =====

def run_visualization():
    """
    モデルをロードし、テストデータで予測を実行してオーバーレイ画像を保存する
    """
    # モデルをロード
    print(f"モデルをロードしています: {MODEL_WEIGHTS_PATH}")
    model = PositionModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_WEIGHTS_PATH}")
        return
    model.eval()

    # テストデータセットを準備
    print("テストデータを読み込んでいます...")
    test_dataset = ExpectCurveDataset(
        pt_path=PT_SPLIT_TEST_PATH,
        color_path=COLOR_PATH,
        gt_path=GT_PATH,
        augment=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print(f"予測を開始します。結果は '{OUTPUT_DIR}' ディレクトリに保存されます。")
    with torch.no_grad():
        processed_samples = 0
        for batch_data in tqdm(test_loader, desc="予測中"):
            if batch_data is None:
                continue

            rgba_tensor, g_pts_norm, c_pts_norm, gt_tensor = batch_data
            
            # データをGPUに送る
            rgba_gpu = rgba_tensor.to(device)
            g_pts_gpu = g_pts_norm.to(device)
            c_pts_gpu = c_pts_norm.to(device)
            
            # モデルで予測を実行（出力は正規化された座標）
            predicted_norm = model(rgba_gpu, g_pts_gpu, c_pts_gpu)

            # バッチ内の各データに対して処理
            for j in range(rgba_gpu.shape[0]):
                global_idx = processed_samples + j
                
                # --- 1. 土台となるカラー画像を準備 ---
                color_image_path = test_dataset.color_full_list[global_idx]
                if not os.path.exists(color_image_path): continue
                color_img = Image.open(color_image_path).convert("RGB")
                overlay_np = np.array(color_img)
                
                # --- 2. 必要な座標をピクセル単位に変換 ---
                start_px = g_pts_norm[j].numpy() * [IMG_WIDTH, IMG_HEIGHT]
                end_px = c_pts_norm[j].numpy() * [IMG_WIDTH, IMG_HEIGHT]
                
                pred_p1_px = predicted_norm[j, 0:2].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
                pred_p2_px = predicted_norm[j, 2:4].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
                
                # --- 3. 曲線を描画 ---
                # 正解(GT)の線を赤色で描画
                gt_mask = gt_tensor[j].squeeze().cpu().numpy() > 0.5
                overlay_np[gt_mask] = [255, 0, 0]

                # 予測したベジェ曲線を計算して白色で描画
                pred_bezier_points = calculate_bezier_points_numpy(start_px, pred_p1_px, pred_p2_px, end_px)
                # OpenCVはBGR形式を標準とするため、一度色空間を変換
                overlay_bgr = cv2.cvtColor(overlay_np, cv2.COLOR_RGB2BGR)
                cv2.polylines(overlay_bgr, [pred_bezier_points], isClosed=False, color=(255, 255, 255), thickness=2)
                # PILで保存するためにRGB形式に戻す
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                
                # --- 4. 画像を保存 ---
                save_filename = os.path.basename(color_image_path)
                Image.fromarray(overlay_rgb).save(os.path.join(OUTPUT_DIR, save_filename))
            
            processed_samples += rgba_gpu.shape[0]

    print("完了しました。")

if __name__ == '__main__':
    run_visualization()