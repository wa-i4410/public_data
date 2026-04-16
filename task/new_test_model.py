# new_test_model.py (予測結果のオーバーレイ画像生成)

import os
import re
import torch
import numpy as np
from PIL import Image
import pydiffvg
# newmodel.pyから、ご提示いただいたモデルとデータセットをインポート
from newmodel import ExpectCurveDataset, PlusModel 
from torch.utils.data import DataLoader
from tqdm import tqdm

# ----- 1. 設定項目 -----

# ★★★ 使用する学習済みモデルのパスをここに指定してください ★★★
MODEL_PATH = "finetuned_model_lr0.0005_lam0.01_w1.5_s3.0.pth" 

# テストデータ関連のパス
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_PATH = '/home/onozawa/GT2dspline'

# 結果を保存するディレクトリ
SAVE_DIR = "test_overlay_predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

# 画像サイズ
IMG_WIDTH = 640
IMG_HEIGHT = 480

# PyDiffVGの初期設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)
pydiffvg.set_print_timing(False)
render_func = pydiffvg.RenderFunction.apply

# ----- 2. ユーティリティ関数 -----

def custom_collate_fn(batch):
    """データローダーがNoneを返した場合にバッチから除外する"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def draw_curve_on_image(image_np, start_point_px, end_point_px, control_points_px, color):
    """
    指定された点を使ってpydiffvgで曲線を描画し、NumPy配列として返す
    """
    # 制御点と始点・終点を結合してベジェ曲線のための点リストを作成
    points = torch.cat([start_point_px.unsqueeze(0), control_points_px, end_point_px.unsqueeze(0)], dim=0)
    
    path = pydiffvg.Path(
        num_control_points=torch.tensor([2]), 
        points=points, 
        stroke_width=torch.tensor(1.5), 
        is_closed=False
    )
    group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]), 
        fill_color=None, 
        stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0]) # RGBA
    )
    
    # pydiffvgでレンダリング
    scene_args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
    rendered_img = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *scene_args)
    
    # 描画されたマスクを取得 (alphaチャンネル)
    mask = rendered_img[:, :, 3].cpu().numpy() > 0.5
    
    # 元の画像に色を付けて上書き
    image_np[mask] = color
    return image_np

# ----- 3. メイン処理 -----

def run_prediction_and_save():
    """
    モデルをロードし、テストデータで予測を実行して結果を画像として保存する
    """
    # モデルをロード
    print(f"モデルをロードしています: {MODEL_PATH}")
    # ご提示のモデルクラスをインスタンス化
    model = PlusModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        return
    model.eval() # 評価モードに設定

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
        batch_size=16, # メモリに応じて調整
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    print(f"予測を開始します。結果は '{SAVE_DIR}' ディレクトリに保存されます。")
    with torch.no_grad():
        processed_samples = 0
        for batch_data in tqdm(test_loader, desc="予測中"):
            if batch_data is None:
                continue

            rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_tensor = batch_data
            
            # データをGPUに送る
            g_pts = growth_point_tensor.to(device)
            c_pts = candidate_point_tensor.to(device)
            rgba_tensor = rgba_tensor.to(device)
            
            # ★ ご提示のモデルの入力形式に合わせて呼び出し
            predicted_control_points = model(rgba_tensor, g_pts, c_pts)

            # バッチ内の各データに対して処理
            for j in range(rgba_tensor.shape[0]):
                global_idx = processed_samples + j
                
                # --- 1. 土台となるカラー画像を準備 ---
                color_image_path = test_dataset.color_full_list[global_idx]
                if not os.path.exists(color_image_path): continue
                color_img = Image.open(color_image_path).resize((IMG_WIDTH, IMG_HEIGHT)).convert("RGB")
                overlay_np = np.array(color_img)
                
                # --- 2. 正解(GT)の線を描画 (赤色) ---
                gt_mask = gt_tensor[j].squeeze().cpu().numpy() > 0.5
                overlay_np[gt_mask] = [255, 0, 0]
                
                # --- 3. 予測した線を描画 (白色) ---
                start_px = g_pts[j] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                end_px = c_pts[j] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
                pred_ctrl_px = predicted_control_points[j].view(2, 2)
                
                overlay_np = draw_curve_on_image(overlay_np, start_px, end_px, pred_ctrl_px, color=[255, 255, 255])

                # --- 4. 画像を保存 ---
                save_filename = os.path.basename(color_image_path)
                Image.fromarray(overlay_np).save(os.path.join(SAVE_DIR, save_filename))
            
            processed_samples += rgba_tensor.shape[0]

    print("完了しました。")

if __name__ == '__main__':
    run_prediction_and_save()