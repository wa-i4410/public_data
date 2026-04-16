# finetune_overfit_test.py (安定化・最終版)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pydiffvg
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from newmodel import  ExpectCurveDataset # newmodel.pyからインポート
from coordinate_model import PositionModel # coordinate_model.pyからインポート
import cv2

# ===== 1. 設定項目 =====
# --- 入力パス ---
PRETRAINED_MODEL_PATH = "pretrained_model.pth"
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_PATH = '/home/onozawa/GT2dspline'
TEST_IMAGE_NAME = "7445.png"

# --- パラメータ ---
IMG_WIDTH = 640
IMG_HEIGHT = 480
# ===== ★★★ 修正点1: 学習率をさらに下げる ★★★ =====
LEARNING_RATE = 1e-5 
NUM_STEPS = 500      # ステップ数を増やして緩やかな変化を観察
STROKE_WIDTH = 1.5
SIGMA = 5.0

# --- システム設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)
pydiffvg.set_print_timing(False)
render_func = pydiffvg.RenderFunction.apply

# ===== 2. ヘルパー関数 =====
def dice_loss(pred, target, smooth=1e-6):
    pred, target = pred.contiguous().view(-1), target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2. * intersection + smooth) / (union + smooth)

def calculate_bezier_points_numpy(p0, p1, p2, p3, num_points=100):
    t = np.linspace(0, 1, num_points)[:, np.newaxis]
    points = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return points.astype(np.int32)

# ===== 3. メイン処理 =====
def run_finetune_overfit_test():
    """ファインチューニングが単一データで機能するかテストする"""
    
    model = PositionModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"事前学習済みモデル '{PRETRAINED_MODEL_PATH}' を正常にロードしました。")
    except FileNotFoundError:
        print(f"エラー: 事前学習済みモデルが見つかりません: {PRETRAINED_MODEL_PATH}")
        return
        
    optim_ = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    bce_loss_fn = nn.BCELoss()

    try:
        temp_dataset = ExpectCurveDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, gt_path=GT_PATH)
        idx = temp_dataset.pt_baselist.index(TEST_IMAGE_NAME.replace('.png', '_points.txt'))
        single_data = temp_dataset[idx]
        if single_data is None: return
        
        rgba_tensor, g_pts, c_pts, gt_tensor = [d.unsqueeze(0).to(device) for d in single_data]
        background_image = Image.open(os.path.join(COLOR_PATH, TEST_IMAGE_NAME)).convert("RGB")
        print(f"テストデータ '{TEST_IMAGE_NAME}' を読み込み、ファインチューニングの過学習テストを開始します。")
    except (ValueError, IndexError):
        print(f"エラー: {TEST_IMAGE_NAME} がテストセットに見つかりません。")
        return

    model.train()
    losses = []
    for step in tqdm(range(NUM_STEPS), desc="Finetuning Overfit Test"):
        optim_.zero_grad()
        predicted_norm = model(rgba_tensor, g_pts, c_pts)
        
        p1 = predicted_norm[0, 0:2] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
        p2 = predicted_norm[0, 2:4] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
        start_px, end_px = g_pts[0] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device), c_pts[0] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
        points = torch.stack([start_px, p1, p2, end_px])

        path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(STROKE_WIDTH), is_closed=False)
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=torch.tensor([1.0]*4, device=device))
        args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
        
        rendered_mask = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)[:, :, 3].unsqueeze(0).unsqueeze(0)
        target_mask = gt_tensor
        
        blurred_pred = TF.gaussian_blur(rendered_mask, kernel_size=5, sigma=SIGMA)
        blurred_gt = TF.gaussian_blur(target_mask, kernel_size=5, sigma=SIGMA)
        
        loss_dice = dice_loss(blurred_pred, blurred_gt)
        loss_bce = bce_loss_fn(blurred_pred, blurred_gt)
        loss = loss_dice + loss_bce
        
        # ===== ★★★ 修正点2: NaNチェック ★★★ =====
        if torch.isnan(loss):
            print(f"\nステップ {step} で損失がNaNになりました。学習を停止します。")
            break
        
        loss.backward()
        
        # ===== ★★★ 修正点3: 勾配クリッピング ★★★ =====
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim_.step()
        losses.append(loss.item())

    print("ファインチューニング過学習テスト完了。")

    # (以降のプロットと可視化は変更なし)
    final_loss = losses[-1] if losses else float('inf')
    print(f"最終的な損失: {final_loss:.6f}")
    if final_loss < 0.5:
        print("\n✅ 成功: ファインチューニングは機能しています！")
    else:
        print("\n❌ 失敗: ファインチューニングが学習していません。")
    # ... (グラフ保存、オーバーレイ画像保存)

    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Finetuning Overfitting Test Loss (Dice+BCE)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("finetune_overfit_loss.png")
    plt.close()
    print("損失グラフを 'finetune_overfit_loss.png' に保存しました。")
    # 最終的な予測結果をオーバーレイ
    model.eval()
    with torch.no_grad():
        final_pred_norm = model(rgba_tensor, g_pts, c_pts)
        gt_img = Image.open(os.path.join(GT_PATH, TEST_IMAGE_NAME.replace('.png', '_gt.png'))).convert("L")
        
        p1_px = final_pred_norm[0, 0:2].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        p2_px = final_pred_norm[0, 2:4].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        start_px_np = g_pts[0].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]
        end_px_np = c_pts[0].cpu().numpy() * [IMG_WIDTH, IMG_HEIGHT]

        overlay_np = np.array(background_image)
        # GTを赤で描画
        overlay_np[np.array(gt_img) > 0] = [255, 0, 0]

        # ★★★ 予測線をOpenCVで描画 ★★★
        pred_bezier_points = calculate_bezier_points_numpy(start_px_np, p1_px, p2_px, end_px_np)
        cv2.polylines(overlay_np, [pred_bezier_points], isClosed=False, color=(255, 255, 255), thickness=2)
        
        Image.fromarray(overlay_np).save("finetune_overfit_result.png")
        print("オーバーレイ画像を 'finetune_overfit_result.png' に保存しました。")

if __name__ == '__main__':
    run_finetune_overfit_test()