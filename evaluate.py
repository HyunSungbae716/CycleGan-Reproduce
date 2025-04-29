import os
from PIL import Image
import torch
from torchvision import transforms
from torch.nn import L1Loss

# ─────── CONFIG ───────
results_dir = 'results/my_experiment/test_latest/images'
# Assumes filenames like: horse_001_real_A.png  and  horse_001_fake_B.png
real_suffix = '_real_A.png'
fake_suffix = '_fake_B.png'

# ─────── SETUP ───────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_tensor = transforms.ToTensor()  # scales to [0,1]
l1_loss = L1Loss(reduction='mean').to(device)

# ─────── GATHER PAIRS ───────
pairs = []
for fname in os.listdir(results_dir):
    if fname.endswith(real_suffix):
        base = fname[:-len(real_suffix)]
        real_path = os.path.join(results_dir, fname)
        fake_fname = base + fake_suffix
        fake_path = os.path.join(results_dir, fake_fname)
        if os.path.exists(fake_path):
            pairs.append((real_path, fake_path))
        else:
            print(f"[WARN] No fake image for: {real_path}")

if not pairs:
    raise RuntimeError("No image pairs found. Check your suffixes and directory.")

# ─────── COMPUTE L1 LOSS ───────
losses = []
for real_path, fake_path in sorted(pairs):
    # Load & move to device
    real_img = to_tensor(Image.open(real_path).convert('RGB')).unsqueeze(0).to(device)
    fake_img = to_tensor(Image.open(fake_path).convert('RGB')).unsqueeze(0).to(device)
    # Compute loss
    loss = l1_loss(fake_img, real_img).item()
    losses.append(loss)
    print(f"{os.path.basename(real_path)} ↔ {os.path.basename(fake_path)}: L1 = {loss:.4f}")

# ─────── REPORT ───────
mean_loss = sum(losses) / len(losses)
print(f"\nAverage L1 over {len(losses)} images: {mean_loss:.4f}")
