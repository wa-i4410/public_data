import os
import json
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import pydiffvg
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import csv

# ===== 初期設定 =====
pydiffvg.set_use_gpu(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
render = pydiffvg.RenderFunction.apply

GT_DIR = "/home/onozawa/GTpractice"
PT_DIR = "/home/onozawa/savepoints"
COLOR_IMG_DIR = "/home/onozawa/CALORsample" # カラー画像のパス
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
PATCH_SIZE = 64 # 画像パッチのサイズ (例: 64x64ピクセル)

SAVE_DIR = "predicted_results_with_cnn" # 保存ディレクトリをCNN用に変更
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== データ生成 =====
def natural_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def read_points(file_path):
    points = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    label, x_str, y_str = line.split(',')
                    points.append((label, int(x_str), int(y_str)))
                except ValueError:
                    continue
    return points

# 画像パッチを安全に切り出すヘルパー関数
def get_image_patch(image_path, center_x, center_y, patch_size, canvas_width, canvas_height):
    try:
        img = Image.open(image_path).convert("RGB")
        # 画像サイズをレンダリングサイズに合わせる (重要)
        img = img.resize((canvas_width, canvas_height))

        half_patch = patch_size // 2
        left = int(center_x - half_patch)
        top = int(center_y - half_patch)
        right = int(center_x + half_patch)
        bottom = int(center_y + half_patch)

        # 範囲チェックとクリッピング
        left = max(0, left)
        top = max(0, top)
        right = min(canvas_width, right)
        bottom = min(canvas_height, bottom)

        # 切り出す領域が有効かチェック
        if right <= left or bottom <= top:
            # 切り出し範囲が不正な場合は、すべてゼロの画像パッチを返すなどの代替策
            print(f"Warning: Invalid patch region for {image_path} at ({center_x}, {center_y}). Returning black patch.")
            return Image.new('RGB', (patch_size, patch_size), (0, 0, 0))

        patch = img.crop((left, top, right, bottom))
        # 最終的なパッチサイズにリサイズ（クリッピングでサイズが変わった場合）
        patch = patch.resize((patch_size, patch_size))
        return patch
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Returning black patch.")
        return Image.new('RGB', (patch_size, patch_size), (0, 0, 0))
    except Exception as e:
        print(f"Error processing image {image_path} for patch at ({center_x}, {center_y}): {e}. Returning black patch.")
        return Image.new('RGB', (patch_size, patch_size), (0, 0, 0))


def generate_data():
    gt_files = sorted(os.listdir(GT_DIR), key=natural_key)
    pt_files = set(os.listdir(PT_DIR))
    train_data = {}

    for gt_file in gt_files:
        base_name = gt_file.replace('_gt.png', '')
        pt_file = f"{base_name}_points.txt"
        color_img_path = os.path.join(COLOR_IMG_DIR, f"{base_name}.png") # カラー画像のパス
        if pt_file not in pt_files or not os.path.exists(color_img_path):
            continue

        gt_image = Image.open(os.path.join(GT_DIR, gt_file)).convert("L")
        gt_tensor = T.ToTensor()(gt_image).to(device)
        point_data = read_points(os.path.join(PT_DIR, pt_file))

        start_point = None
        candidates = []
        for label, x, y in point_data:
            pt = torch.tensor([float(x), float(y)], device=device)
            if label.lower() == "growth":
                start_point = pt
            elif label.lower() == "15cm":
                candidates.append(pt)
        if start_point is None or not candidates:
            continue

        ctrl1s, ctrl2s, losses = [], [], []
        # ここで成長点と候補点の画像パッチを抽出（訓練データ生成時に一度だけ行う）
        # パッチを保存するディレクトリ
        patch_save_dir = os.path.join("data_patches", base_name)
        os.makedirs(patch_save_dir, exist_ok=True)

        # 成長点の画像パッチを保存
        start_patch = get_image_patch(color_img_path, start_point[0].item(), start_point[1].item(), PATCH_SIZE, CANVAS_WIDTH, CANVAS_HEIGHT)
        start_patch_filename = os.path.join(patch_save_dir, "growth_patch.png")
        start_patch.save(start_patch_filename)

        candidate_patch_filenames = []
        for i, end_point in enumerate(candidates):
            c1 = (start_point * 0.66 + end_point * 0.34 + torch.tensor([10.0, -10.0], device=device)).detach().clone().requires_grad_()
            c2 = (start_point * 0.34 + end_point * 0.66 + torch.tensor([-10.0, 10.0], device=device)).detach().clone().requires_grad_()

            path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                                 points=torch.stack([start_point, c1, c2, end_point]),
                                 stroke_width=torch.tensor(1.5),
                                 is_closed=False)
            group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                        fill_color=None,
                                        stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0]))
            shapes, shape_groups = [path], [group]
            optim_ = torch.optim.Adam([c1, c2], lr=1.0)
            for t in range(100):
                path.points = torch.stack([start_point, c1, c2, end_point])
                args = pydiffvg.RenderFunction.serialize_scene(gt_image.width, gt_image.height, shapes, shape_groups)
                img = render(gt_image.width, gt_image.height, 2, 2, t, None, *args)[:, :, 3].unsqueeze(0)
                loss = (((img - gt_tensor) * (gt_tensor > 0.5)).pow(2)).sum() / (gt_tensor > 0.5).sum()
                optim_.zero_grad(); loss.backward(); optim_.step()

            ctrl1s.append(c1)
            ctrl2s.append(c2)
            losses.append(loss)

            # 候補点の画像パッチを保存
            end_patch = get_image_patch(color_img_path, end_point[0].item(), end_point[1].item(), PATCH_SIZE, CANVAS_WIDTH, CANVAS_HEIGHT)
            end_patch_filename = os.path.join(patch_save_dir, f"candidate_{i}_patch.png")
            end_patch.save(end_patch_filename)
            candidate_patch_filenames.append(f"candidate_{i}_patch.png")


        min_idx = torch.argmin(torch.tensor(losses)).item()
        sample_list = []
        for i in range(len(candidates)):
            sample = {
                "growth_point": start_point.tolist(),
                "candidate_point": candidates[i].tolist(),
                "control_point1": ctrl1s[i].tolist(),
                "control_point2": ctrl2s[i].tolist(),
                "loss": losses[i].item(),
                "label": int(i == min_idx),
                "growth_patch_path": os.path.join("data_patches", base_name, "growth_patch.png"), # パッチのパスを追加
                "candidate_patch_path": os.path.join("data_patches", base_name, candidate_patch_filenames[i]) # パッチのパスを追加
            }
            sample_list.append(sample)
        train_data[base_name] = sample_list
        print(f"{base_name} is done!")

    with open("train_data_with_patches.json", "w") as f: # ファイル名を変更
        json.dump(train_data, f, indent=2)

# ===== データ分割 =====
def split_data(train_ratio=0.8, seed=42):
    # データ読み込みファイル名を変更
    with open("train_data_with_patches.json") as f:
        full_data = json.load(f)
    keys = list(full_data.keys())
    random.seed(seed)
    random.shuffle(keys)
    split_idx = int(len(keys) * train_ratio)
    train_data = {k: full_data[k] for k in keys[:split_idx]}
    test_data = {k: full_data[k] for k in keys[split_idx:]}

    with open("train_data_with_patches.json", "w") as f: # ファイル名を変更
        json.dump(train_data, f, indent=2)
    with open("test_data_with_patches.json", "w") as f: # ファイル名を変更
        json.dump(test_data, f, indent=2)

# ===== データセットとモデル =====
class CurveDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.samples = [(img_id, s) for img_id, lst in data.items() for s in lst]

        # 画像の前処理
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの平均・標準偏差
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_id, sample = self.samples[idx]

        # 座標と深度情報 (もし深度情報も使用するならここに追加)
        # 座標は正規化することを強く推奨
        # 例: x = torch.tensor(sample["growth_point"] + sample["candidate_point"], dtype=torch.float32)
        # 今回はCNNで画像情報を使うので、座標はそのまま入力とする
        x_coords = torch.tensor(sample["growth_point"] + sample["candidate_point"], dtype=torch.float32)
        
        # 画像パッチの読み込みと前処理
        growth_patch_path = sample["growth_patch_path"]
        candidate_patch_path = sample["candidate_patch_path"]

        growth_patch = Image.open(growth_patch_path).convert("RGB")
        candidate_patch = Image.open(candidate_patch_path).convert("RGB")

        growth_patch_tensor = self.transform(growth_patch)
        candidate_patch_tensor = self.transform(candidate_patch)

        y = torch.tensor(sample["control_point1"] + sample["control_point2"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        
        # モデルに渡す入力はタプルで返す (座標, 成長パッチ, 候補パッチ)
        return (x_coords, growth_patch_tensor, candidate_patch_tensor), y, label, img_id


# 新しいモデル：CNNとMLPの組み合わせ
class HybridModel(nn.Module):
    def __init__(self, cnn_feature_dim=128): # CNNからの特徴量次元
        super().__init__()

        # CNN部分 (VGGの小さなブロックのようなもの)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # PATCH_SIZE/2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # PATCH_SIZE/4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # PATCH_SIZE/8
        )
        
        # CNNの出力次元を計算
        # PATCH_SIZE が 64 の場合、(64/8) = 8 -> 128 * 8 * 8 = 8192
        # ここで `cnn_feature_dim` に合わせるための線形層を入れることもできますが、
        # SimplestNetでは最後にFlattenしてMLPに渡すのが一般的です。
        # ここでは、最終的な出力次元を直接計算して使用します。
        # CNNの出力は (バッチサイズ, 128, PATCH_SIZE/8, PATCH_SIZE/8)
        # 例: PATCH_SIZE=64 -> 128 * 8 * 8 = 8192

        # 座標特徴量 (4次元) とCNN特徴量 (2 * 128 * (PATCH_SIZE/8)^2) を結合
        # 例: PATCH_SIZE=64 -> 4 + 2 * 128 * 8 * 8 = 4 + 16384 = 16388
        
        # CNN出力サイズを計算するためのダミーフォワード
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE)
            cnn_out_dim = self.cnn(dummy_input).numel() // 1 # numel() は要素数を返す

        self.regressor = nn.Sequential(
            nn.Linear(4 + cnn_out_dim * 2, 512), # 座標(4) + 成長パッチCNN特徴量 + 候補パッチCNN特徴量
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4) # 制御点1(x,y) + 制御点2(x,y)
        )

    def forward(self, x_tuple):
        x_coords, growth_patch, candidate_patch = x_tuple

        # CNNで画像特徴量を抽出
        growth_features = self.cnn(growth_patch)
        candidate_features = self.cnn(candidate_patch)

        # Flatten (平坦化)
        growth_features = torch.flatten(growth_features, 1)
        candidate_features = torch.flatten(candidate_features, 1)

        # 座標特徴量と画像特徴量を結合
        combined_features = torch.cat((x_coords, growth_features, candidate_features), dim=1)
        
        return self.regressor(combined_features)

# ===== 学習 =====
def train_model():
    model = HybridModel().to(device) # モデルをHybridModelに変更
    loader = DataLoader(CurveDataset("train_data_with_patches.json"), batch_size=32, shuffle=True)
    optim_ = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(400):
        total_loss = 0.0
        for (x_coords, growth_patch, candidate_patch), y, _, _ in loader:
            x_coords = x_coords.to(device)
            growth_patch = growth_patch.to(device)
            candidate_patch = candidate_patch.to(device)
            y = y.to(device)
            #
            # モデルの入力としてタプルを渡す
            loss = loss_fn(model((x_coords, growth_patch, candidate_patch)), y)
            optim_.zero_grad(); loss.backward(); optim_.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "regressor_cnn.pth") # 保存ファイル名を変更

# ===== 評価と可視化 =====
def evaluate_model():
    model = HybridModel().to(device) # モデルをHybridModelに変更
    model.load_state_dict(torch.load("regressor_cnn.pth", map_location=device)) # ロードファイル名を変更
    model.eval()

    dataset = CurveDataset("test_data_with_patches.json") # データセットパスを変更
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.MSELoss()
    results = []

    with torch.no_grad():
        for i, ((x_coords, growth_patch, candidate_patch), y_true, label, image_id) in enumerate(dataloader):
            x_coords = x_coords.to(device)
            growth_patch = growth_patch.to(device)
            candidate_patch = candidate_patch.to(device)
            
            # モデルの入力としてタプルを渡す
            y_pred = model((x_coords, growth_patch, candidate_patch)).cpu().squeeze()
            y_true = y_true.squeeze()
            image_id = image_id[0]
            loss = loss_fn(y_pred, y_true).item()

            print(f"{image_id} → loss: {loss:.4f}")

            # x_coordsはバッチサイズ1なので、x_coords[0]で最初の要素にアクセス
            start = x_coords[0][:2].cpu()
            end = x_coords[0][2:4].cpu()
            ctrl1 = y_pred[:2]
            ctrl2 = y_pred[2:]

            path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                                points=torch.stack([start, ctrl1, ctrl2, end]),
                                stroke_width=torch.tensor(1.5),
                                is_closed=False)
            group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                        fill_color=None,
                                        stroke_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))
            shapes = [path]
            shape_groups = [group]
            args = pydiffvg.RenderFunction.serialize_scene(CANVAS_WIDTH, CANVAS_HEIGHT, shapes, shape_groups)
            img_pred = render(CANVAS_WIDTH, CANVAS_HEIGHT, 2, 2, i, None, *args)[:, :, :3].cpu().numpy()

            # カラー画像にオーバーレイ
            color_path = os.path.join(COLOR_IMG_DIR, f"{image_id}.png")
            if os.path.exists(color_path):
                color_img = Image.open(color_path).resize((CANVAS_WIDTH, CANVAS_HEIGHT)).convert("RGB")
                color_np = np.array(color_img).astype(np.float32) / 255.0
                pred_np = img_pred / 255.0 if img_pred.max() > 1 else img_pred
                blended = (0.4 * color_np + 0.6 * pred_np) * 255
                Image.fromarray(blended.astype(np.uint8)).save(os.path.join(SAVE_DIR, f"overlay_{i:03d}.png"))

            # GT画像と合成表示
            gt_path = os.path.join(GT_DIR, f"{image_id}_gt.png")
            print(f"Checking GT: {gt_path}")
            if os.path.exists(gt_path):
                gt_img = Image.open(gt_path).resize((CANVAS_WIDTH, CANVAS_HEIGHT)).convert("L")
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(gt_img, cmap='gray')
                axs[0].set_title("Ground Truth")
                axs[1].imshow(img_pred)
                axs[1].set_title("Predicted Bézier")
                for ax in axs:
                    ax.axis("off")
                plt.tight_layout()
                plt.savefig(os.path.join(SAVE_DIR, f"compare_{i:03d}.png"))
                plt.close()
            else:
                print(f"⚠️ GT image not found: {gt_path}")

            results.append({"index": i, "image_id": image_id, "loss": loss})

    with open(os.path.join(SAVE_DIR, "loss_summary_cnn.csv"), "w", newline="") as f: # ファイル名を変更
        writer = csv.DictWriter(f, fieldnames=["index", "image_id", "loss"])
        writer.writeheader()
        writer.writerows(results)


# ===== 実行 =====
if __name__ == "__main__":
    print("🔧 Step 1: Generating training data (with image patches)...")
    generate_data()

    print("📊 Step 2: Splitting train/test...")
    split_data()

    print("🎓 Step 3: Training model (HybridModel with CNN)...")
    train_model()

    print("🧪 Step 4: Evaluating model (HybridModel with CNN)...")
    evaluate_model()

    print("✅ All done. Results saved to 'predicted_results_with_cnn/'")