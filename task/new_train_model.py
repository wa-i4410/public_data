# new_train_model.py (ファインチューニング対応 最終修正版)

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pydiffvg
import shutil
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from newmodel import ExpectCurveDataset, PlusModel
from PIL import Image
from tqdm import tqdm
import time
import numpy as np

# ===== 1. 設定項目 =====
# ★★★ 事前学習で保存したモデルのパスをここに指定 ★★★
PRETRAINED_MODEL_PATH = "pretrained_model.pth" 

# パス設定
PT_PATH = ('/home/onozawa/savepoints')
PT_SPLIT_TRAIN_PATH =('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Train')
PT_SPLIT_TEST_PATH = ('/home/onozawa/デスクトップ/vrl_onozawa/task/SavePointsAll/Test')
COLOR_PATH = ('/home/onozawa/CALORsample')
GT_PATH = ('/home/onozawa/GT2dspline')

# 基本パラメータ
IMG_WIDTH = 640
IMG_HEIGHT = 480
SMOOTH_EPSILON = 1e-6
BATCH_SIZE = 32

# PyTorchデバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pydiffvg.set_device(device)
pydiffvg.set_print_timing(False)
render_func = pydiffvg.RenderFunction.apply

# 訓練の進捗を可視化したいテスト画像のファイル名を指定
VISUALIZE_IMAGES = ["631.png", "5976.png"] 
# 可視化画像を保存するディレクトリ
VISUALIZATION_SAVE_DIR = "training_progress"
os.makedirs(VISUALIZATION_SAVE_DIR, exist_ok=True)

# ===== 2. ユーティリティ関数 =====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def split_data(train_ratio=0.8, seed=42):
    # (この関数は変更なし)
    os.makedirs(PT_SPLIT_TRAIN_PATH, exist_ok=True)
    os.makedirs(PT_SPLIT_TEST_PATH, exist_ok=True)
    tmp_list=[]
    for pt in os.listdir(PT_PATH):
        if pt.endswith('.txt'):
            tmp_list.append(pt)
    random.seed(seed)
    random.shuffle(tmp_list)
    split_idx = int(len(tmp_list) * train_ratio)
    print(f"全ファイル数: {len(tmp_list)}")
    print(f"訓練用ファイル数: {split_idx}")
    print(f"テスト用ファイル数: {len(tmp_list) - split_idx}")
    for t in tmp_list[:split_idx]:
        src_path = os.path.join(PT_PATH,t)
        dst_path = os.path.join(PT_SPLIT_TRAIN_PATH,t)
        shutil.copy(src_path,dst_path)
    for t in tmp_list[split_idx:]:
        src_path = os.path.join(PT_PATH,t)
        dst_path = os.path.join(PT_SPLIT_TEST_PATH,t)
        shutil.copy(src_path,dst_path)

def dice_loss(pred, target, smooth=SMOOTH_EPSILON):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice_score = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice_score

def custom_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents nans when BCE_loss is large
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

seed_everything()

def save_prediction_overlay(model, epoch, image_name, stroke_width):
    """ 指定された単一の画像に対して予測を行い、オーバーレイ画像を保存する """
    model.eval() # 必ず評価モードにする
    def read_points_for_vis(file_path):
        points = {}
        with open(file_path, "r") as f:
            for line in f:
                try:
                    label, x_str, y_str = line.strip().split(',')
                    if label not in points: points[label] = []
                    points[label].append((int(x_str), int(y_str)))
                except ValueError: continue
        return points
    # 必要なファイルのパスを構築
    color_img_path = os.path.join(COLOR_PATH, image_name)
    gt_img_path = os.path.join(GT_PATH, image_name.replace('.png', '_gt.png'))
    points_path = os.path.join(PT_SPLIT_TEST_PATH, image_name.replace('.png', '_points.txt'))

    if not all(os.path.exists(p) for p in [color_img_path, gt_img_path, points_path]):
        return

    # データ準備
    color_img = Image.open(color_img_path).convert('RGB')
    gt_img = Image.open(gt_img_path).convert('L')
    point_data = read_points_for_vis(points_path) 
    
    start_point_coords = point_data.get('growth')[0]
    candidate_point_coords = point_data.get('15cm', point_data.get('nega'))[0]

    # モデルへの入力テンソルを作成
    # (データセットの__getitem__のロジックを簡略化して再現)
    img_tensor = TF.to_tensor(color_img).unsqueeze(0).to(device)
    heatmap = torch.zeros(1, 1, IMG_HEIGHT, IMG_WIDTH, device=device)
    stroke_color = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)
    rgba_tensor = torch.cat([img_tensor, heatmap], dim=1)
    
    g_pts = torch.tensor([[start_point_coords[0]/IMG_WIDTH, start_point_coords[1]/IMG_HEIGHT]], dtype=torch.float32).to(device)
    c_pts = torch.tensor([[candidate_point_coords[0]/IMG_WIDTH, candidate_point_coords[1]/IMG_HEIGHT]], dtype=torch.float32).to(device)

    # 予測
    with torch.no_grad():
        predicted = model(rgba_tensor, g_pts, c_pts)

    # 描画と保存
    p1, p2 = predicted[0, 0:2], predicted[0, 2:4]
    start_px = g_pts[0] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
    end_px = c_pts[0] * torch.tensor([IMG_WIDTH, IMG_HEIGHT], device=device)
    points = torch.stack([start_px, p1, p2, end_px])

    path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(stroke_width), is_closed=False)
    group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=None, stroke_color=stroke_color)
    args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
    rendered_mask = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)[:, :, 3]

    # オーバーレイ画像の作成
    overlay_np = np.array(color_img)
    gt_mask = np.array(gt_img) > 0
    pred_mask = rendered_mask.cpu().numpy() > 0.5
    
    overlay_np[gt_mask] = [255, 0, 0] # 正解を赤
    overlay_np[pred_mask] = [255, 255, 255] # 予測を白

    # 保存
    epoch_save_dir = os.path.join(VISUALIZATION_SAVE_DIR, f"epoch_{epoch+1:03d}")
    os.makedirs(epoch_save_dir, exist_ok=True)
    Image.fromarray(overlay_np).save(os.path.join(epoch_save_dir, image_name))


# ===== 3. 評価専用の関数 =====
def evaluate_model(model, data_loader, lam, stroke_width, sigma):
    model.eval()
    total_loss = 0.0
    processed_batches = 0
    bce_loss_fn = nn.BCELoss()
    focal_loss_fn = FocalLoss().to(device)
    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None: continue
            rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_tensor_cpu = batch_data
            
            rgba_gpu = rgba_tensor.to(device)
            g_pts_gpu = growth_point_tensor.to(device)
            c_pts_gpu = candidate_point_tensor.to(device)
            gt_gpu = gt_tensor_cpu.to(device)
            
            predicted = model(rgba_gpu, g_pts_gpu, c_pts_gpu)

            batch_loss = 0.0
            for i in range(rgba_gpu.shape[0]):
                p1_x, p1_y, p2_x, p2_y = predicted[i, 0], predicted[i, 1], predicted[i, 2], predicted[i, 3]
                start_x, start_y = g_pts_gpu[i, 0] * IMG_WIDTH, g_pts_gpu[i, 1] * IMG_HEIGHT
                end_x, end_y = c_pts_gpu[i, 0] * IMG_WIDTH, c_pts_gpu[i, 1] * IMG_HEIGHT
                start_point_tensor_px = torch.tensor([start_x, start_y], device=device, dtype=torch.float32)
                end_point_tensor_px = torch.tensor([end_x, end_y], device=device, dtype=torch.float32)
                control_points_tensor = torch.stack([torch.stack([p1_x, p1_y]), torch.stack([p2_x, p2_y])])
                points = torch.cat([start_point_tensor_px.unsqueeze(0), control_points_tensor, end_point_tensor_px.unsqueeze(0)], dim=0)
                
                path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(stroke_width, device=device), is_closed=False)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0], device=device), fill_color=None, stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device))
                args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
                img_rendered_raw = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)
                rendered_curve_mask = img_rendered_raw[:, :, 3].unsqueeze(0)

                # 正解マスクはGPU上のものを使用
                target_mask = gt_gpu[i]

                if sigma > 0:
                    rendered_curve_mask = TF.gaussian_blur(rendered_curve_mask.unsqueeze(0), kernel_size=5, sigma=sigma).squeeze(0)
                    target_mask = TF.gaussian_blur(target_mask.unsqueeze(0), kernel_size=5, sigma=sigma).squeeze(0)

                current_dice_loss = dice_loss(rendered_curve_mask, target_mask)
                current_focal_loss = focal_loss_fn(rendered_curve_mask, target_mask)
                current_bce_loss = bce_loss_fn(rendered_curve_mask, target_mask)
                l1_loss_fn = nn.L1Loss()
                current_l1_loss = l1_loss_fn(rendered_curve_mask, target_mask)
                combined_loss = current_dice_loss  + current_bce_loss
                
                v1, v2, v3 = points[1] - points[0], points[2] - points[1], points[3] - points[2]
                cos_sim1 = F.cosine_similarity(v1, v2, dim=0)
                cos_sim2 = F.cosine_similarity(v2, v3, dim=0)
                regularization_loss = (1 + cos_sim1) + (1 + cos_sim2)
                
                total_loss_for_item = combined_loss + 0.0 * regularization_loss
                batch_loss += total_loss_for_item
            
            total_loss += (batch_loss / rgba_gpu.shape[0]).item()
            processed_batches += 1
    return total_loss / processed_batches if processed_batches > 0 else 0.0

# ===== 4. ファインチューニング関数 =====
def finetune_model(l_r, pa, lam, stroke_width, sigma):
    model = PlusModel(img_height=IMG_HEIGHT, img_width=IMG_WIDTH).to(device)
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"事前学習済みモデル '{PRETRAINED_MODEL_PATH}' を正常にロードしました。")
    except FileNotFoundError:
        print(f"エラー: 事前学習済みモデルが見つかりません: {PRETRAINED_MODEL_PATH}")
        return
    
    train_dataset = ExpectCurveDataset(pt_path=PT_SPLIT_TRAIN_PATH, color_path=COLOR_PATH, gt_path=GT_PATH, augment=True)
    test_dataset = ExpectCurveDataset(pt_path=PT_SPLIT_TEST_PATH, color_path=COLOR_PATH, gt_path=GT_PATH, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    optim_ = optim.Adam(model.parameters(), lr=l_r, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_, 'min', factor=0.2, patience=pa, verbose=True)

    bce_loss_fn = nn.BCELoss()
    focal_loss_fn = FocalLoss().to(device)
    train_losses, test_losses = [], []
    best_test_loss, best_epoch = float('inf'), -1
    NUM_EPOCHS = 500

    print(f"-----ファインチューニング開始 (lr={l_r}, patience={pa}, lambda={lam}, width={stroke_width}, sigma={sigma})-----")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [Finetune]", leave=False)

        for batch_data in train_loader_tqdm:
            if batch_data is None: continue
            rgba_tensor, growth_point_tensor, candidate_point_tensor, gt_tensor_cpu = batch_data
            
            rgba_gpu = rgba_tensor.to(device)
            g_pts_gpu = growth_point_tensor.to(device)
            c_pts_gpu = candidate_point_tensor.to(device)
            gt_gpu = gt_tensor_cpu.to(device)

            optim_.zero_grad()
            predicted = model(rgba_gpu, g_pts_gpu, c_pts_gpu)

            batch_loss = 0.0
            for i in range(rgba_gpu.shape[0]):
                p1_x, p1_y, p2_x, p2_y = predicted[i, 0], predicted[i, 1], predicted[i, 2], predicted[i, 3]
                start_x, start_y = g_pts_gpu[i, 0] * IMG_WIDTH, g_pts_gpu[i, 1] * IMG_HEIGHT
                end_x, end_y = c_pts_gpu[i, 0] * IMG_WIDTH, c_pts_gpu[i, 1] * IMG_HEIGHT
                start_point_tensor_px = torch.tensor([start_x, start_y], device=device, dtype=torch.float32)
                end_point_tensor_px = torch.tensor([end_x, end_y], device=device, dtype=torch.float32)
                control_points_tensor = torch.stack([torch.stack([p1_x, p1_y]), torch.stack([p2_x, p2_y])])
                points = torch.cat([start_point_tensor_px.unsqueeze(0), control_points_tensor, end_point_tensor_px.unsqueeze(0)], dim=0)
                
                path = pydiffvg.Path(num_control_points=torch.tensor([2]), points=points, stroke_width=torch.tensor(stroke_width, device=device), is_closed=False)
                group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0], device=device), fill_color=None, stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0], device=device))
                args = pydiffvg.RenderFunction.serialize_scene(IMG_WIDTH, IMG_HEIGHT, [path], [group])
                img_rendered_raw = render_func(IMG_WIDTH, IMG_HEIGHT, 2, 2, 0, None, *args)
                rendered_curve_mask = img_rendered_raw[:, :, 3].unsqueeze(0)
                
                target_mask = gt_gpu[i]

                if sigma > 0:
                    rendered_curve_mask = TF.gaussian_blur(rendered_curve_mask.unsqueeze(0), kernel_size=5, sigma=sigma).squeeze(0)
                    target_mask = TF.gaussian_blur(target_mask.unsqueeze(0), kernel_size=5, sigma=sigma).squeeze(0)

                current_dice_loss = dice_loss(rendered_curve_mask, target_mask)
                current_focal_loss = focal_loss_fn(rendered_curve_mask, target_mask)
                current_bce_loss = bce_loss_fn(rendered_curve_mask, target_mask)
                l1_loss_fn = nn.L1Loss()
                current_l1_loss = l1_loss_fn(rendered_curve_mask, target_mask)
                combined_loss = current_dice_loss + current_bce_loss
                
                v1, v2, v3 = points[1] - points[0], points[2] - points[1], points[3] - points[2]
                cos_sim1 = F.cosine_similarity(v1, v2, dim=0)
                cos_sim2 = F.cosine_similarity(v2, v3, dim=0)
                regularization_loss = (1 + cos_sim1) + (1 + cos_sim2)
                
                total_loss_for_item = combined_loss + 0.0 * regularization_loss
                batch_loss += total_loss_for_item
            
            avg_batch_loss = batch_loss / rgba_gpu.shape[0]
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim_.step()
            total_train_loss += avg_batch_loss.item()
            train_loader_tqdm.set_postfix(loss=avg_batch_loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_test_loss = evaluate_model(model, test_loader, lam, stroke_width, sigma)
        test_losses.append(avg_test_loss)
        
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss, best_epoch = avg_test_loss, epoch
            save_filename = f"finetuned_model_lr{l_r}_lam{lam}_w{stroke_width}_s{sigma}.pth"
            torch.save(model.state_dict(), save_filename)
            print(f"✨ New best model saved at epoch {epoch + 1} with test loss: {best_test_loss:.4f}")

        scheduler.step(avg_test_loss)

        if epoch < 20: # 最初の20エポックのみ実行
            print(f"--- Epoch {epoch+1}: 予測画像を保存中... ---")
            for img_name in VISUALIZE_IMAGES:
                save_prediction_overlay(model, epoch, img_name, stroke_width)

    print("\n-----ファインチューニング終了-----")
    # ... (グラフ保存処理) ...
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, NUM_EPOCHS + 1), test_losses, label='Test Loss')
    plt.title(f'Training & Test Loss (lr={l_r}, lam={lam})')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plot_output_dir = "training_plots_with_test_new"
    os.makedirs(plot_output_dir, exist_ok=True)
    plot_filename = os.path.join(plot_output_dir, f"finetuned_model_lr{l_r}_lam{lam}_w{stroke_width}_s{sigma}.png")
    plt.savefig(plot_filename)
    plt.close()
    print(f"訓練とテストの損失グラフを '{plot_filename}' に保存しました。")
    return best_test_loss

# ===== 5. 実行ブロック =====
if __name__ == "__main__":
    # データ分割は必要に応じて一度だけ実行
    # split_data() 
    
    print("\n🎓 Step 2: Fine-tuning the model...")
    # 推奨パラメータで実行
    best_loss = finetune_model(l_r=1e-5, pa=30, lam=0.01, stroke_width=1.5, sigma=5.0)
    print(f"\n最終的なベストテストLOSS: {best_loss:.4f}")