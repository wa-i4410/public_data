# visualize_ac_unet.py

import torch
import os
import numpy as np
import cv2 # OpenCVを使用
from PIL import Image

# ===== ★★★ 修正点1: モデルのインポートを変更 ★★★ =====
# 自分のプロジェクトに合わせて、モデルやデータセットの定義をインポート
from newmodel import AC_UNet # AC-UNetモデルをインポート
from newmodel import UNetDataset     # データセットは既存のものを流用
# =======================================================

# ===== 設定項目 (自分の環境に合わせて変更) =====
# ① 参照したい学習済みAC-UNetモデルのパス
MODEL_WEIGHTS_PATH = 'model_sig_0.0.pth' # ★ご自身のモデルファイル名に変更してください

# ② データセットのパス (テスト用)
PT_SPLIT_TEST_PATH = '/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test'
COLOR_PATH = '/home/onozawa/CALORsample'
GT_PATH = '/home/onozawa/GT2dspline'

# ③ 何サンプル確認するか
NUM_SAMPLES_TO_VISUALIZE = 10

# ④ デバッグ画像の保存先フォルダ
OUTPUT_DIR = 'ac_unet_overlay_predictions'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

def overlay_prediction(model, dataset, index):
    """
    元のカラー画像に、GT(赤)と予測(緑)の経路を重ねて描画する関数
    """
    print(f"--- Processing test sample {index} ---")

    # 1. データセットから入力と正解を取得
    # データセットがNoneを返す可能性を考慮
    data = dataset[index]
    if data is None:
        print(f"-> Sample {index} is None. Skipping.")
        return
    input_tensor, target_tensor = data

    # 2. モデルで予測を実行
    model.eval()
    with torch.no_grad():
        # unsqueeze(0)でバッチ次元を追加してモデルに入力
        input_batch = input_tensor.unsqueeze(0).to(device)
        predicted_mask_tensor = model(input_batch)

    # 3. テンソルを描画用にOpenCV形式(Numpy配列)へ変換
    # 元のカラー画像
    color_image_np = input_tensor[:3, :, :].permute(1, 2, 0).cpu().numpy()
    color_image_np = (color_image_np * 255).astype(np.uint8)
    output_image = cv2.cvtColor(color_image_np, cv2.COLOR_RGB2BGR)

    # GTマスク
    gt_mask_np = (target_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)

    # 予測マスク (確率が0.5以上のピクセルを1(白)、それ以外を0(黒)にする)
    predicted_mask_np = (predicted_mask_tensor.squeeze().cpu().numpy() > 0.05).astype(np.uint8) * 255
    
    # 形態学的クロージング処理で、途切れた線を繋げる（任意）
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(predicted_mask_np, cv2.MORPH_CLOSE, kernel)

    # 4. マスクから輪郭を抽出し、カラー画像に描画
    # GTの輪郭を赤色で描画 
    gt_contours, _ = cv2.findContours(gt_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_image, gt_contours, -1, (0, 0, 255), 2) # BGR: 赤

    # 予測の輪郭を緑色で描画
    pred_contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output_image, pred_contours, -1, (0, 255, 0), 2) # BGR: 緑

    # 5. 結果を保存
    # データセットから元のファイル名を取得して保存ファイル名に利用
    base_name = os.path.basename(dataset.pt_full_list[index]).replace('_points.txt', '.png')
    save_path = os.path.join(OUTPUT_DIR, f'overlay_{base_name}')
    cv2.imwrite(save_path, output_image)
    print(f"-> Saved overlay image to {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== ★★★ 修正点2: AC-UNetモデルをロード ★★★ =====
    # U-Netの代わりにAC_UNetをインスタンス化
    model = AC_UNet(in_channels=4, out_channels=1).to(device)
    # ===================================================

    if os.path.exists(MODEL_WEIGHTS_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
        print(f"モデル '{MODEL_WEIGHTS_PATH}' をロードしました。")
    else:
        print(f"警告: モデルの重みファイル '{MODEL_WEIGHTS_PATH}' が見つかりません。")
        exit()

    # テストデータセットをロード
    # データセットのクラス名はご自身のものに合わせてください
    test_dataset = UNetDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, gt_path=GT_PATH, augment=False)
    
    if len(test_dataset) > 0:
        # 指定した数のサンプル、またはデータセット全体の小さい方だけ実行
        for i in range(min(NUM_SAMPLES_TO_VISUALIZE, len(test_dataset))):
            overlay_prediction(model, test_dataset, i)
    else:
        print("エラー: テストデータセットにデータがありません。")