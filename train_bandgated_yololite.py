#!/usr/bin/env python3
"""
Train a YOLO-ish detector with BandGates from a standard YOLO data.yaml
 -- Reads data.yaml with keys: train (txt), val (txt), names, nc
 -- Each line in train/val txt is a path to a RAW mosaic image (e.g., .pgm)
 -- On-the-fly: crop -> 5x5 demosaic -> 25 bands -> per-band norm
 -- On-the-fly: converts YOLO labels from RAW frame to demosaiced frame
 -- Learns BandGate alphas (per-band importances), prints a ranked list each epoch

python train_bandgated_yololite.py \
    --data /path/to/data.yaml \
    --epochs 100 \
    --batch 16 \
    --lr 0.002 \
    --l1 0.001 \
    --workers 4
"""

import os, math, yaml
from pathlib import Path
from typing import Tuple, List, Optional
import csv
import matplotlib.pyplot as plt
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------- Geometry & demosaic params (from your snippet) -----------------------
UNCROPPED_H = 1088
UNCROPPED_W = 2048

CROP_TOP  = 3
CROP_LEFT = 0
DOWNSAMPLE = 5

CROPPED_H = 1080
CROPPED_W = 2045
DEMOSAIC_H = CROPPED_H // DOWNSAMPLE   # 216
DEMOSAIC_W = CROPPED_W // DOWNSAMPLE   # 409

# keep your BW exactly as defined
BW = np.array([
    [886, 896, 877, 867, 951],
    [793, 806, 782, 769, 675],
    [743, 757, 730, 715, 690],
    [926, 933, 918, 910, 946],
    [846, 857, 836, 824, 941],
], dtype=int)

# No phase correction since the crop already aligns to the pattern origin
BAND_ORDER = [int(BW[ro, co]) for ro in range(5) for co in range(5)]

# ----------------------- Label conversion RAW -> DEMOSAICED -----------------------
def convert_bbox_raw_to_demosaic(xc, yc, w, h, img_w=UNCROPPED_W, img_h=UNCROPPED_H):
    # 1) denormalize in RAW
    xc *= img_w; yc *= img_h; w *= img_w; h *= img_h
    # 2) crop
    xc -= CROP_LEFT; yc -= CROP_TOP
    # 3) downsample (5x5)
    xc /= DOWNSAMPLE; yc /= DOWNSAMPLE; w /= DOWNSAMPLE; h /= DOWNSAMPLE
    # 4) renormalize in demosaiced frame
    xc /= DEMOSAIC_W; yc /= DEMOSAIC_H; w /= DEMOSAIC_W; h /= DEMOSAIC_H
    return xc, yc, w, h

def clip_bbox01(xc, yc, w, h) -> Optional[Tuple[float,float,float,float]]:
    x1, y1 = xc - w/2, yc - h/2
    x2, y2 = xc + w/2, yc + h/2
    x1, y1, x2, y2 = map(lambda v: max(0.0, min(1.0, v)), (x1, y1, x2, y2))
    w2, h2 = x2-x1, y2-y1
    if w2 <= 0 or h2 <= 0:
        return None
    return x1 + w2/2, y1 + h2/2, w2, h2

# ----------------------- On-the-fly demosaic -----------------------
def demosaic_5x5_mosaic_to_cube(gray_cropped: np.ndarray) -> np.ndarray:
    h, w = gray_cropped.shape
    assert h % 5 == 0 and w % 5 == 0
    H, W = h // 5, w // 5
    cube_dict = {}
    for ro in range(5):
        for co in range(5):
            band = gray_cropped[ro::5, co::5]
            code = int(BW[ro, co])         # ← no (ro+roff)/(co+coff)
            cube_dict[code] = band
    cube = np.stack([cube_dict[c] for c in BAND_ORDER], axis=0).astype(np.float32)
    return cube

def per_band_standardize(cube: np.ndarray) -> np.ndarray:
    C, H, W = cube.shape
    flat = cube.reshape(C, -1)
    mean = flat.mean(1, keepdims=True)
    std  = flat.std(1, keepdims=True) + 1e-6
    return ((flat - mean) / std).reshape(C, H, W)

# ----------------------- Dataset -----------------------
def _default_label_path_from_image_path(img_path: Path) -> Path:
    # Try YOLO convention: replace /images/ with /labels/ and change suffix to .txt
    parts = list(img_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
    except ValueError:
        # Not standard layout; fallback to same dir with .txt
        return img_path.with_suffix(".txt")
    label_path = Path(*parts).with_suffix(".txt")
    if label_path.exists():
        return label_path
    # Fallback
    return img_path.with_suffix(".txt")

class YOLOMosaicDetDataset(Dataset):
    def __init__(self, list_file: Path):
        """
        list_file: a file that lists absolute or relative image paths (one per line)
        Labels are inferred by replacing /images/ with /labels/ (YOLO convention),
        else falling back to same directory with .txt.
        """
        base = list_file.parent
        self.items: List[Path] = []
        with open(list_file, "r") as f:
            for line in f:
                p = line.strip()
                if not p: continue
                ip = (base / p).resolve() if not os.path.isabs(p) else Path(p)
                self.items.append(ip)
        if not self.items:
            raise RuntimeError(f"No images found from list {list_file}")

    def __len__(self): return len(self.items)

    def _read_labels(self, img_path: Path) -> np.ndarray:
        lp = _default_label_path_from_image_path(img_path)
        if not lp.exists():
            return np.zeros((0,5), dtype=np.float32)
        rows = []
        with open(lp, "r") as f:
            for ln in f:
                if not ln.strip(): continue
                cls, xc, yc, w, h = map(float, ln.split())
                xc, yc, w, h = convert_bbox_raw_to_demosaic(xc, yc, w, h)
                clipped = clip_bbox01(xc, yc, w, h)
                if clipped is None:
                    continue
                xc, yc, w, h = clipped
                rows.append([int(cls), xc, yc, w, h])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0,5), dtype=np.float32)

    def __getitem__(self, i):
        p = self.items[i]
        # Load RAW mosaic as grayscale
        raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"Could not read image {p}")
        # Crop as per spec
        cropped = raw[CROP_TOP:1083, CROP_LEFT:2045]   # (1080, 2045)
        cube = demosaic_5x5_mosaic_to_cube(cropped)    # [25, 216, 409]
        cube = per_band_standardize(cube)              # per-band z-score
        y = self._read_labels(p)                       # [N,5] (cls, xc, yc, w, h)
        return torch.from_numpy(cube), torch.from_numpy(y), p.stem

def collate_det(batch):
    cubes, labels, stems = zip(*batch)
    cubes = torch.stack(cubes, 0)  # [B,25,H,W]
    # Build target tensor [M,6] = (b, cls, xc, yc, w, h)
    all_t = []
    for b, y in enumerate(labels):
        if y.numel():
            bcol = torch.full((y.shape[0], 1), b, dtype=torch.float32)
            all_t.append(torch.cat([bcol, y.float()], dim=1))
    targets = torch.cat(all_t, 0) if all_t else torch.zeros((0,6), dtype=torch.float32)
    return cubes, targets, stems

# ----------------------- Model (BandGate + tiny YOLO-ish) -----------------------
class BandGate(nn.Module):
    def __init__(self, C=25, init_alpha=1.0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(C))  # alpha = exp(beta) starts at 1.0
    def alpha(self): return torch.exp(self.beta)
    def forward(self, x):
        a = self.alpha()
        return x * a.view(1,-1,1,1), a

class ConvAct(nn.Module):
    """Conv + SiLU (no BN), keeps the gate's amplitude alive."""
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        p = (k//2) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=True)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))

class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None):
        super().__init__()
        p = (k//2) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=2):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c1, c_, 1, 1)
        self.m = nn.Sequential(*[ConvBNAct(c_, c_, 3, 1) for _ in range(n)])
        self.cv3 = ConvBNAct(2*c_, c2, 1, 1)
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y2 = self.m(y2)
        return self.cv3(torch.cat([y1, y2], 1))

class YOLOLite(nn.Module):
    """
    Single-scale YOLO-ish detector (stride 8) with BandGate front-end.
    Head predicts per-anchor [obj, cx, cy, w, h, class...].
    """
    def __init__(self, nc=2, img_size=(DEMOSAIC_H, DEMOSAIC_W), anchors=((10,12),(16,19),(23,33))):
        super().__init__()
        self.nc = nc
        self.img_h, self.img_w = img_size
        self.anchors = torch.tensor(anchors, dtype=torch.float32)  # [A,2]
        self.na = len(anchors)
        self.stride = 8
        self.gh = math.ceil(self.img_h / self.stride)
        self.gw = math.ceil(self.img_w / self.stride)

        self.gate = BandGate(25)

        # backbone (8x downsample)
        self.stem  = ConvAct(25, 32, 3, 2)   # /2
        self.c2f1  = C2f(32, 64, 2)
        self.down1 = ConvBNAct(64, 64, 3, 2)   # /4
        self.c2f2  = C2f(64, 128, 2)
        self.down2 = ConvBNAct(128, 128, 3, 2) # /8
        self.c2f3  = C2f(128, 128, 2)
        self.neck  = ConvBNAct(128, 128, 3, 1)
        self.head  = nn.Conv2d(128, self.na*(5+self.nc), 1)

    def forward(self, x):
        x, alpha = self.gate(x)                     # [B,25,H,W]
        x = self.stem(x); x = self.c2f1(x)
        x = self.down1(x); x = self.c2f2(x)
        x = self.down2(x); x = self.c2f3(x)
        x = self.neck(x)
        p = self.head(x)                            # [B, A*(5+nc), gh, gw]
        B, _, gh, gw = p.shape
        p = p.view(B, self.na, 5+self.nc, gh, gw).permute(0,1,3,4,2).contiguous()
        return p, alpha

# ----------------------- YOLO-ish loss helpers -----------------------
def build_targets(targets, grid_shape, na):
    device = targets.device
    gh, gw = grid_shape
    if targets.numel() == 0:
        return targets.new_zeros((0,6)), targets.new_zeros((0,), dtype=torch.long), targets.new_zeros((0,), dtype=torch.long), targets.new_zeros((0,), dtype=torch.long)

    gxy = targets[:,2:4] * torch.tensor([gw, gh], device=device)
    gwh = targets[:,4:6] * torch.tensor([gw, gh], device=device)
    gij = gxy.long()
    gi, gj = gij[:,0].clamp_(0, gw-1), gij[:,1].clamp_(0, gh-1)

    anchors = torch.tensor([[10,12],[16,19],[23,33]], device=device, dtype=torch.float32)
    box_wh = gwh.unsqueeze(1)
    anc_wh = anchors.unsqueeze(0)
    inter = torch.min(box_wh, anc_wh).prod(2)
    union = (box_wh.prod(2) + anc_wh.prod(2) - inter)
    iou_a = inter / (union + 1e-9)
    best_a = iou_a.argmax(1)
    return targets, gi, gj, best_a

def bbox_iou_xywh(a, b, eps=1e-9):
    ax1 = a[...,0]-a[...,2]/2; ay1 = a[...,1]-a[...,3]/2
    ax2 = a[...,0]+a[...,2]/2; ay2 = a[...,1]+a[...,3]/2
    bx1 = b[...,0]-b[...,2]/2; by1 = b[...,1]-b[...,3]/2
    bx2 = b[...,0]+b[...,2]/2; by2 = b[...,1]+b[...,3]/2
    iw = (torch.min(ax2,bx2)-torch.max(ax1,bx1)).clamp(0)
    ih = (torch.min(ay2,by2)-torch.max(ay1,by1)).clamp(0)
    inter = iw*ih
    area_a = (ax2-ax1).clamp(0)*(ay2-ay1).clamp(0)
    area_b = (bx2-bx1).clamp(0)*(by2-by1).clamp(0)
    return inter / (area_a + area_b - inter + eps)

class YoloLoss(nn.Module):
    def __init__(self, nc, grid_shape, na, obj_gain=1.0, cls_gain=0.5, box_gain=5.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.nc = nc
        self.gh, self.gw = grid_shape
        self.na = na
        self.obj_gain, self.cls_gain, self.box_gain = obj_gain, cls_gain, box_gain

    def forward(self, p, targets):
        device = p.device
        B, A, gh, gw, E = p.shape
        obj_t = torch.zeros((B,A,gh,gw), device=device)

        tgts, gi, gj, ga = build_targets(targets, (gh,gw), A)
        box_loss = p.new_zeros(())
        cls_loss = p.new_zeros(())

        if tgts.numel():
            b = tgts[:,0].long()
            c = tgts[:,1].long()
            gxy = tgts[:,2:4] * torch.tensor([gw,gh], device=device)
            gwh = tgts[:,4:6] * torch.tensor([gw,gh], device=device)

            pred = p[b, ga, gj, gi]          # [N,5+nc]
            pobj, pxywh, pcl = pred[...,0], pred[...,1:5], pred[...,5:]

            px = torch.sigmoid(pxywh[:,0])   # 0..1 within cell
            py = torch.sigmoid(pxywh[:,1])
            pw = torch.exp(pxywh[:,2]).clamp(0, 4*gw)
            ph = torch.exp(pxywh[:,3]).clamp(0, 4*gh)

            pred_xywh = torch.stack([px + gi.float(), py + gj.float(), pw, ph], 1)
            tgt_xywh  = torch.stack([gxy[:,0], gxy[:,1], gwh[:,0], gwh[:,1]], 1)
            iou = bbox_iou_xywh(pred_xywh, tgt_xywh)
            box_loss = (1.0 - iou).mean() * self.box_gain

            obj_t[b, ga, gj, gi] = iou.detach().clamp(0,1)

            tcls = torch.zeros_like(pcl)
            tcls[torch.arange(c.shape[0]), c] = 1.0
            cls_loss = self.bce(pcl, tcls).mean() * self.cls_gain

        obj_loss = self.bce(p[...,0], obj_t).mean() * self.obj_gain
        total = box_loss + cls_loss + obj_loss
        return total, {"box": box_loss.item(), "cls": cls_loss.item(), "obj": obj_loss.item()}

# ----------------------- Logging -----------------------
def log_band_importances(alpha, epoch, top_k=10):
    vals = alpha.detach().cpu().tolist()
    ranked = sorted(list(zip(BAND_ORDER, vals)), key=lambda kv: kv[1], reverse=True)
    print(f"\n[Epoch {epoch}] Band importances (alpha):")  # <-- text fix
    for i, (code, v) in enumerate(ranked):
        tag = "★" if i < top_k else " "
        print(f"  {tag} band_code={code:>3}  alpha={v:.4f}")
    print()

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def init_csv(csv_path: Path, band_order: list[int]):
    is_new = not csv_path.exists()
    if is_new:
        headers = ["epoch", "train_loss", "val_loss"] + [f"alpha_{b}" for b in band_order]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)

def append_row(csv_path: Path, epoch: int, train_loss: float, val_loss: float, alphas: list[float], band_order: list[int]):
    row = [epoch, train_loss, val_loss] + [float(a) for a in alphas]
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

def plot_losses(out_path: Path, epochs: list[int], tr_hist: list[float], val_hist: list[float]):
    plt.figure(figsize=(6,4))
    plt.plot(epochs, tr_hist, label="train")
    plt.plot(epochs, val_hist, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Train vs Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_band_alphas(out_path: Path, epochs: list[int], alpha_hist: list[list[float]], band_order: list[int]):
    """
    One plot with all 25 band alphas over epochs + legend (key).
    Colors are stable across bands using a discrete colormap.
    """
    A = np.array(alpha_hist, dtype=np.float32)  # [num_epochs, 25]
    plt.figure(figsize=(9, 5))
    # discrete colormap with as many distinct colors as bands
    cmap = plt.cm.get_cmap("tab20", len(band_order)) if len(band_order) <= 20 else plt.cm.get_cmap("tab20b", len(band_order))
    # Fallback if > 40 bands ever: just sample viridis
    if len(band_order) > 40:
        cmap = lambda i: plt.cm.viridis(i / (len(band_order) - 1))

    for i, code in enumerate(band_order):
        color = cmap(i) if callable(getattr(cmap, "__call__", None)) else cmap(i)
        plt.plot(epochs, A[:, i], label=str(code), linewidth=1.6, color=color)

    plt.xlabel("epoch")
    plt.ylabel("alpha")
    plt.title("Band alphas over epochs")
    plt.grid(True, alpha=0.25)

    # Legend (key): outside the plot, multiple columns so it stays compact
    ncols = 5 if len(band_order) >= 15 else 3
    plt.legend(title="band code", loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, ncol=ncols, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def save_first_sample_bands(out_path: Path, cube_25CH: torch.Tensor, band_order: list[int]):
    """
    cube_25CH: torch.Tensor [25,H,W] AFTER demosaic+standardize, right before the model.
    Saves a 5x5 grid image with per-band min-max normalization for visualization.
    """
    x = cube_25CH.detach().cpu().float().numpy()  # [25,H,W]
    fig, axs = plt.subplots(5, 5, figsize=(10, 8))
    for i, ax in enumerate(axs.ravel()):
        if i >= x.shape[0]:
            ax.axis("off"); continue
        band = x[i]
        # per-band min-max to [0,1] for display
        lo, hi = float(band.min()), float(band.max())
        disp = (band - lo) / (hi - lo + 1e-8)
        ax.imshow(disp, cmap="gray", interpolation="nearest")
        ax.set_title(f"{band_order[i]}", fontsize=9)
        ax.axis("off")
    fig.suptitle("First sample bands (after demosaic+standardize)", y=0.995)
    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(out_path, dpi=150)
    plt.close()

# ----------------------- Training -----------------------
def train(data_yaml: Path, epochs=50, batch=16, lr=2e-3, l1=1e-3, workers=4, logdir: Path=Path("runs/bandgate")):
    # Parse data.yaml
    D = yaml.safe_load(Path(data_yaml).read_text())
    base = Path(data_yaml).parent
    train_list = (base / D["train"]).resolve()
    val_list   = (base / D["val"]).resolve()
    names      = D["names"]
    nc         = int(D["nc"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = YOLOMosaicDetDataset(train_list)
    val_ds   = YOLOMosaicDetDataset(val_list)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers,
                              pin_memory=True, collate_fn=collate_det)
    val_loader   = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers,
                              pin_memory=True, collate_fn=collate_det)

    # --- logging setup ---
    run_dir = logdir / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(run_dir)
    csv_path = run_dir / "metrics.csv"
    init_csv(csv_path, BAND_ORDER)
    losses_png = run_dir / "losses.png"
    alphas_png = run_dir / "band_alphas_over_epochs.png"
    first_bands_png = run_dir / "first_sample_bands.png"

    epochs_hist: list[int] = []
    train_hist:  list[float] = []
    val_hist:    list[float] = []
    alpha_hist:  list[list[float]] = []

    # save the first sample's bands once (from the first training batch)
    first_saved = False

    # Save a visual check of the very first sample's bands (post-demosaic+standardize)
    for cubes_dbg, _, _ in train_loader:
        # cubes_dbg: [B,25,H,W] exactly what's fed into the model
        save_first_sample_bands(first_bands_png, cubes_dbg[0], BAND_ORDER)
        print(f"Saved first-sample band grid → {first_bands_png}")
        break

    model = YOLOLite(nc=nc, img_size=(DEMOSAIC_H, DEMOSAIC_W)).to(device)
    loss_fn = YoloLoss(nc=nc, grid_shape=(model.gh, model.gw), na=model.na)

    gate_params = [model.gate.beta]  # learnable gate parameter
    base_params = [p for n,p in model.named_parameters() if not n.startswith("gate.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": lr,       "weight_decay": 1e-4},
            {"params": gate_params, "lr": lr * 2.5, "weight_decay": 0.0},
        ]
    )

    # Optional warm-up to highlight alphas early
    warmup_epochs = 2
    for n, p in model.named_parameters():
        if not n.startswith("gate."):     # freeze everything except the gate
            p.requires_grad_(False)

    # print after freezing so you see only gate.beta
    trainable = [n for n,p in model.named_parameters() if p.requires_grad]
    print(f"[Warmup] trainable params: {len(trainable)} → {trainable[:8]}{' ...' if len(trainable)>8 else ''}")

    for epoch in range(1, epochs+1):
        model.train()
        if epoch == warmup_epochs + 1:
            for p in model.parameters(): p.requires_grad_(True)
            print("\n[Info] Unfroze backbone/head after warm-up.\n")

        with torch.no_grad():
            a = model.gate.alpha().detach().cpu()
            print(f"[Epoch {epoch}] alpha range: {a.min():.3f}..{a.max():.3f}")

        running = 0.0
        for cubes, targets, stems in train_loader:
            cubes, targets = cubes.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred, alpha = model(cubes)
            total, parts = loss_fn(pred, targets)

            # # L1 sparsity on per-band alphas
            # l1_pen = F.softplus(model.gate.log_alpha).abs().sum()
            # loss = total + l1 * l1_pen

            # Regularize alphas toward 1.0 so they can go up or down.
            alpha_now = model.gate.alpha()             # shape [25], α = exp(β)
            reg = torch.abs(alpha_now - 1.0).sum()     # L1 distance to 1.0
            loss = total + l1 * reg               # total training loss

            loss.backward()
            optimizer.step()
            running += loss.item()

        # quick val loss
        model.eval()
        with torch.no_grad():
            vtot, vN = 0.0, 0
            for cubes, targets, stems in val_loader:
                cubes, targets = cubes.to(device), targets.to(device)
                pred, alpha = model(cubes)
                total, _ = loss_fn(pred, targets)
                vtot += total.item() * cubes.size(0)
                vN += cubes.size(0)
            val_loss = vtot / max(1, vN)
            
            # --- logging epoch metrics & alphas ---
            a_now = model.gate.alpha().detach().cpu().numpy().tolist()

            epochs_hist.append(epoch)
            train_hist.append(running/len(train_loader))
            val_hist.append(val_loss)
            alpha_hist.append(a_now)

            append_row(csv_path, epoch, train_hist[-1], val_hist[-1], a_now, BAND_ORDER)
            plot_losses(losses_png, epochs_hist, train_hist, val_hist)
            plot_band_alphas(alphas_png, epochs_hist, alpha_hist, BAND_ORDER)

            # (console peek remains handy)
            print(f"[Epoch {epoch}] alpha range: {min(a_now):.3f}..{max(a_now):.3f}")
            log_band_importances(model.gate.alpha(), epoch, top_k=10)

        print(f"Epoch {epoch:03d} | train_loss {running/len(train_loader):.4f} | val_loss {val_loss:.4f}")

    alphas = model.gate.alpha().detach().cpu().numpy().tolist()
    torch.save({
        "state_dict": model.state_dict(),
        "alphas": alphas,
        "band_order": BAND_ORDER,
        "names": names
    }, "bandgated_detector.pt")
    print("\nSaved → bandgated_detector.pt")
    print("Final band importances (band_code: alpha):")
    for code, a in sorted(zip(BAND_ORDER, alphas), key=lambda kv: kv[1], reverse=True):
        print(f"  {code}: {a:.4f}")

# ----------------------- CLI -----------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Train BandGated detector from YOLO data.yaml")
    ap.add_argument("--data", type=Path, required=True, help="path to data.yaml (with train, val, names, nc)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l1", type=float, default=1e-3, help="L1 penalty on band-gate alphas")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--logdir", type=Path, default=Path("runs/bandgate"), help="where to write csv and plots")
    args = ap.parse_args()
    train(args.data, epochs=args.epochs, batch=args.batch, lr=args.lr, l1=args.l1, workers=args.workers, logdir=args.logdir)
