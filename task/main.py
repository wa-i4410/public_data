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
COLOR_IMG_DIR = "/home/onozawa/CALORsample"
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480

SAVE_DIR = "predicted_results"
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

def generate_data():
    gt_files = sorted(os.listdir(GT_DIR), key=natural_key)
    pt_files = set(os.listdir(PT_DIR))
    train_data = {}

    for gt_file in gt_files:
        base_name = gt_file.replace('_gt.png', '')
        pt_file = f"{base_name}_points.txt"
        if pt_file not in pt_files:
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
        for end_point in candidates:
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

        min_idx = torch.argmin(torch.tensor(losses)).item()
        sample_list = []
        for i in range(len(candidates)):
            sample = {
                "growth_point": start_point.tolist(),
                "candidate_point": candidates[i].tolist(),
                "control_point1": ctrl1s[i].tolist(),
                "control_point2": ctrl2s[i].tolist(),
                "loss": losses[i].item(),
                "label": int(i == min_idx)
            }
            sample_list.append(sample)
        train_data[base_name] = sample_list
        print(f"{base_name} is done!")

    with open("train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)

# ===== データ分割 =====
def split_data(train_ratio=0.8, seed=42):
    with open("train_data.json") as f:
        full_data = json.load(f)
    keys = list(full_data.keys())
    random.seed(seed)
    random.shuffle(keys)
    split_idx = int(len(keys) * train_ratio)
    train_data = {k: full_data[k] for k in keys[:split_idx]}
    test_data = {k: full_data[k] for k in keys[split_idx:]}

    with open("train_data.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open("test_data.json", "w") as f:
        json.dump(test_data, f, indent=2)

# ===== データセットとモデル =====
class CurveDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.samples = [(img_id, s) for img_id, lst in data.items() for s in lst]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_id, sample = self.samples[idx]
        x = torch.tensor(sample["growth_point"] + sample["candidate_point"], dtype=torch.float32)
        y = torch.tensor(sample["control_point1"] + sample["control_point2"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return x, y, label, img_id

class MLPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, x): return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ===== 学習 =====
def train_model():
    model = MLPRegressor().to(device)
    loader = DataLoader(CurveDataset("train_data.json"), batch_size=32, shuffle=True)
    optim_ = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(200):
        total_loss = 0.0
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optim_.zero_grad(); loss.backward(); optim_.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), "regressor.pth")

# ===== 評価と可視化 =====
# ===== 評価と可視化 =====
def evaluate_model():
    model = MLPRegressor().to(device)
    model.load_state_dict(torch.load("regressor.pth", map_location=device))
    model.eval()

    dataset = CurveDataset("test_data.json")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    loss_fn = nn.MSELoss()
    results = []

    with torch.no_grad():
        for i, (x, y_true, label, image_id) in enumerate(dataloader):
            x = x.to(device)
            y_pred = model(x).cpu().squeeze()
            y_true = y_true.squeeze()
            image_id = image_id[0]
            loss = loss_fn(y_pred, y_true).item()

            print(f"{image_id} → loss: {loss:.4f}")

            start = x[0][:2].cpu()
            end = x[0][2:4].cpu()
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

    with open(os.path.join(SAVE_DIR, "loss_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image_id", "loss"])
        writer.writeheader()
        writer.writerows(results)


# ===== 実行 =====
if __name__ == "__main__":
    print("🔧 Step 1: Generating training data...")
    generate_data()

    print("📊 Step 2: Splitting train/test...")
    split_data()

    print("🎓 Step 3: Training model...")
    train_model()

    print("🧪 Step 4: Evaluating model...")
    evaluate_model()

    print("✅ All done. Results saved to 'predicted_results/'")
    
