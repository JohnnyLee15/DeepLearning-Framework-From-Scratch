# pyTorchTest.py
# BN-free, plain SGD (no momentum), stratified split, MPS-aware timing
import time, random
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------
# Config (match your C++)
# ---------------------------
DATA_ROOT    = "DataFiles/chest_xray"
IMG_SIZE     = 224
BATCH_SIZE   = 8
NUM_WORKERS  = 0            # keep 0 on macOS unless you want spawn workers
SEED         = 1337
LR0          = 0.006
DECAY        = 0.03         # lr_k = LR0 / (1 + k*DECAY)
EPOCHS       = 1            # set to 60 to mirror your full run
WEIGHT_DECAY = 1e-3         # L2 on weights only
PRINT_EVERY  = 50

# ---------------------------
# Device + sync
# ---------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

# ---------------------------
# Model (exactly your layout)
# ---------------------------
class XrayNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Stem
        self.conv1   = nn.Conv2d(1,   24, 3, 1, 1)   # 224x224
        self.conv1x1 = nn.Conv2d(24,  16, 1, 1, 0)   # 224x224
        self.conv2   = nn.Conv2d(16,  64, 3, 2, 1)   # 224 -> 112
        self.pool    = nn.MaxPool2d(2,2)             # 112 -> 56, etc.

        # Block 64 @ 56
        self.b64_1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b64_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.b64_3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.do64  = nn.Dropout(p=0.10)

        # Block 256 @ 28
        self.b256_1 = nn.Conv2d(64, 256, 3, 1, 1)
        self.b256_2 = nn.Conv2d(256,256, 3, 1, 1)
        self.b256_3 = nn.Conv2d(256,256, 3, 1, 1)
        self.do256  = nn.Dropout(p=0.10)

        # Block 384 @ 14
        self.b384_1 = nn.Conv2d(256,384, 3, 1, 1)
        self.b384_2 = nn.Conv2d(384,384, 3, 1, 1)
        self.b384_3 = nn.Conv2d(384,384, 3, 1, 1)
        self.do384  = nn.Dropout(p=0.15)

        # Block 512 @ 7
        self.b512_1 = nn.Conv2d(384,512, 3, 1, 1)
        self.b512_2 = nn.Conv2d(512,512, 3, 1, 1)
        self.b512_3 = nn.Conv2d(512,512, 3, 1, 1)
        self.do512  = nn.Dropout(p=0.20)

        # Head
        self.gap      = nn.AdaptiveAvgPool2d(1)
        self.head_do1 = nn.Dropout(p=0.50)
        self.fc1      = nn.Linear(512, 1024)
        self.head_do2 = nn.Dropout(p=0.50)
        self.fc2      = nn.Linear(1024, num_classes)

        # Kaiming init for ReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv1x1(x))
        x = F.relu(self.conv2(x))   # 112
        x = self.pool(x)            # 56

        x = F.relu(self.b64_1(x)); x = F.relu(self.b64_2(x)); x = F.relu(self.b64_3(x))
        x = self.do64(x); x = self.pool(x)   # 28

        x = F.relu(self.b256_1(x)); x = F.relu(self.b256_2(x)); x = F.relu(self.b256_3(x))
        x = self.do256(x); x = self.pool(x)  # 14

        x = F.relu(self.b384_1(x)); x = F.relu(self.b384_2(x)); x = F.relu(self.b384_3(x))
        x = self.do384(x); x = self.pool(x)  # 7

        x = F.relu(self.b512_1(x)); x = F.relu(self.b512_2(x)); x = F.relu(self.b512_3(x))
        x = self.do512(x); x = self.pool(x)  # 4

        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.head_do1(x)
        x = F.relu(self.fc1(x))
        x = self.head_do2(x)
        return self.fc2(x)  # logits

# ---------------------------
# Dataset & stratified split
# ---------------------------
def build_loaders():
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # divides by 255 -> [0,1]
    ])
    full = datasets.ImageFolder(root=DATA_ROOT, transform=tfm)

    per_class = defaultdict(list)
    for idx, (_, y) in enumerate(full.samples):
        per_class[y].append(idx)

    rng = random.Random(SEED)
    train_idx, val_idx, test_idx = [], [], []
    for idxs in per_class.values():
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(0.20 * n)
        test_part = idxs[:n_test]
        remain = idxs[n_test:]
        n_val = int(0.10 * len(remain))
        val_part = remain[:n_val]
        train_part = remain[n_val:]
        test_idx.extend(test_part); val_idx.extend(val_part); train_idx.extend(train_part)

    train_ds = Subset(full, train_idx)
    val_ds   = Subset(full, val_idx)
    test_ds  = Subset(full, test_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=False)

    return train_loader, val_loader, test_loader, len(full.classes)

# ---------------------------
# Optimizer: L2 on weights only, plain SGD
# ---------------------------
def split_weight_bias_params(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() == 1 or name.endswith(".bias"):
            no_decay.append(p)   # biases etc. (no L2)
        else:
            decay.append(p)      # weights (with L2)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# ---------------------------
# Training / eval with epoch timing
# ---------------------------
def train_one_epoch(model, loader, device, criterion, optimizer, epoch_idx):
    model.train()
    sync(device); epoch_t0 = time.time()
    total, correct, loss_sum = 0, 0, 0.0
    step_t0 = time.time()
    for i, (images, labels) in enumerate(loader, 1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += images.size(0)

        if i % PRINT_EVERY == 0:
            elapsed = time.time() - step_t0
            ips = (i * images.size(0)) / elapsed
            print(f"[Train] step {i}/{len(loader)}  loss {loss_sum/total:.4f}  "
                  f"acc {100*correct/total:.2f}%  {ips:.1f} img/s")
    sync(device); epoch_elapsed = time.time() - epoch_t0
    ips_epoch = total / epoch_elapsed
    return loss_sum/total, 100*correct/total, ips_epoch, epoch_elapsed

@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    sync(device); t0 = time.time()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        out = model(images)
        loss = criterion(out, labels)
        loss_sum += loss.item() * images.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += images.size(0)
    sync(device); elapsed = time.time() - t0
    ips = total / elapsed
    return loss_sum/total, 100*correct/total, ips, elapsed

@torch.inference_mode()
def timed_inference(model, loader, device):
    model.eval()
    # warmup
    for images, _ in loader:
        _ = model(images.to(device)); break
    sync(device); t0 = time.time()
    total = 0
    for images, _ in loader:
        _ = model(images.to(device))
        total += images.size(0)
    sync(device); elapsed = time.time() - t0
    print(f"Inference throughput: {total/elapsed:.1f} img/s on {device}")

# ---------------------------
# LR schedule: lr_k = LR0 / (1 + k*DECAY)  (k = epoch index starting at 1)
# ---------------------------
def set_epoch_lr(optimizer, epoch_idx):
    lr = LR0 / (1.0 + epoch_idx * DECAY)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

# ---------------------------
# Main
# ---------------------------
def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, num_classes = build_loaders()
    model = XrayNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        split_weight_bias_params(model),
        lr=LR0,
        weight_decay=0.0  # handled per param group above
    )

    for ep in range(1, EPOCHS+1):
        current_lr = set_epoch_lr(optimizer, ep)
        tr_loss, tr_acc, tr_ips, tr_secs = train_one_epoch(model, train_loader, device, criterion, optimizer, ep)
        va_loss, va_acc, va_ips, va_secs = evaluate(model, val_loader, device, criterion)
        print(f"Epoch {ep}: lr {current_lr:.6f} | "
              f"train {tr_loss:.4f}/{tr_acc:.2f}% {tr_ips:.1f} img/s ({tr_secs:.1f}s) | "
              f"val {va_loss:.4f}/{va_acc:.2f}% {va_ips:.1f} img/s ({va_secs:.1f}s)")

    te_loss, te_acc, te_ips, te_secs = evaluate(model, test_loader, device, criterion)
    print(f"\nTest: loss {te_loss:.4f}, acc {te_acc:.2f}%, throughput {te_ips:.1f} img/s ({te_secs:.1f}s) on {device}")
    timed_inference(model, test_loader, device)

if __name__ == "__main__":
    # macOS spawn guard (keeps things safe if you raise NUM_WORKERS later)
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
