# analyze_first_conv_25ch.py
# Usage:
#   python analyze_first_conv_25ch.py --weights runs/train/exp/weights/best.pt
# Optional:
#   python analyze_first_conv_25ch.py --weights best.pt --plot

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ultralytics >= 8.x
from y8_mosaic_trainer import YOLOMosaic, DetectionModel, DetectionModelMosaic
from ultralytics.utils.torch_utils import de_parallel

# ---- your 25 band labels (edit as needed) ----
BAND_LABELS = [
    886, 896, 877, 867, 951, 793, 806, 782, 769, 675,
    743, 757, 730, 715, 690, 926, 933, 918, 910, 946,
    846, 857, 836, 824, 941
]  # wavelengths nm, or replace with names/ids

def compute_band_importance(first_conv_weight: torch.Tensor) -> np.ndarray:
    """
    first_conv_weight: [C_out, 25, k, k]
    Returns importance per band: [25], computed as L1 norm over filters & kernel.
    """
    with torch.no_grad():
        # abs then sum over out_channels and spatial kernel dims
        imp = first_conv_weight.abs().sum(dim=(0, 2, 3))  # [25]
        imp = imp.cpu().float().numpy()
    return imp

def per_filter_pref(first_conv_weight: torch.Tensor) -> np.ndarray:
    """
    Returns a [C_out, 25] matrix: per-filter preference over bands
    (L1 over kernel; normalized per filter to sum=1).
    """
    with torch.no_grad():
        w = first_conv_weight.abs().sum(dim=3).sum(dim=2)  # [C_out, 25]
        w = w / (w.sum(dim=1, keepdim=True) + 1e-9)
        return w.cpu().float().numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to Ultralytics .pt checkpoint")
    ap.add_argument("--plot", action="store_true", help="Show a matplotlib bar chart")
    ap.add_argument("--save_dir", type=str, default="band_analysis", help="Where to save CSV/JSON/figs")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = YOLOMosaic(args.weights)
    m = de_parallel(model.model)  # nn.Module (DetectionModel)
    first = m.model[0]            # first module is Conv wrapper
    w = first.conv.weight         # [C_out, 25, k, k]

    if w.shape[1] != 25:
        raise RuntimeError(f"Expected 25 input channels, got {w.shape[1]}")

    # 2) Compute importance & normalize
    imp = compute_band_importance(w)            # [25]
    imp_norm = imp / (imp.sum() + 1e-12)

    # 3) Rank bands
    labels = [str(b) for b in BAND_LABELS] if len(BAND_LABELS) == len(imp) else [f"band_{i}" for i in range(len(imp))]
    order = np.argsort(-imp)  # descending
    ranked = [(int(i), labels[i], float(imp[i]), float(imp_norm[i])) for i in order]

    # Save CSV
    df = pd.DataFrame(ranked, columns=["band_index", "band_label", "importance_raw", "importance_frac"])
    csv_path = save_dir / "first_conv_band_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved per-band importance to {csv_path}\n")
    
    # Print ranking summary
    print("="*60)
    print("BAND IMPORTANCE RANKING (Top to Bottom)")
    print("="*60)
    print(f"{'Rank':<6} {'Band':<12} {'Label':<12} {'Raw Imp':<12} {'Norm Imp':<12} {'Percent':<8}")
    print("-"*60)
    for rank, (band_idx, band_label, raw_imp, norm_imp) in enumerate(ranked[:15], 1):
        percent = norm_imp * 100
        print(f"{rank:<6} {band_idx:<12} {band_label:<12} {raw_imp:<12.4f} {norm_imp:<12.4f} {percent:<8.2f}%")
    
    if len(ranked) > 15:
        print(f"... and {len(ranked) - 15} more bands")
    
    print("\n" + "="*60)
    print(f"TOP 5 BANDS: {', '.join([labels[i] for i in order[:5]])}")
    print(f"BOTTOM 5 BANDS: {', '.join([labels[i] for i in order[-5:]])}")
    print("="*60)
    
    print(f"\nFull ranking table saved to {csv_path}")
    print("\nAll bands ranking:")
    print(df.to_string(index=False))

    # Save a JSON too
    json_path = save_dir / "first_conv_band_importance.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "weights": str(args.weights),
                "band_labels": labels,
                "importance_raw": imp.tolist(),
                "importance_frac": imp_norm.tolist(),
                "rank_desc_indices": order.tolist(),
            },
            f,
            indent=2,
        )
    print(f"Saved JSON to {json_path}")

    # 4) Optional: per-filter preference heatmap data
    pref = per_filter_pref(w)  # [C_out, 25]
    pref_path = save_dir / "first_conv_per_filter_preference.npy"
    np.save(pref_path, pref)
    print(f"Saved per-filter preference matrix to {pref_path} (shape {pref.shape})")

    # 5) Plotting - Create ranked bands visualization
    if args.plot:
        # Create subplots for multiple visualizations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Ranked bands (top to bottom importance)
        ranked_labels = [labels[i] for i in order]
        ranked_importance = [imp_norm[i] for i in order]
        
        # Color coding for top bands
        colors = ['red' if i < 5 else 'orange' if i < 10 else 'skyblue' for i in range(len(ranked_importance))]
        
        y_pos = np.arange(len(ranked_importance))
        bars1 = ax1.barh(y_pos, ranked_importance, color=colors, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{ranked_labels[i]} (band {order[i]})" for i in range(len(ranked_labels))])
        ax1.set_xlabel("Normalized Importance (L1 over filters & kernel)")
        ax1.set_title("Ranked Band Importance (Top to Bottom)", fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars1, ranked_importance)):
            ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.3f}', ha='left', va='center', fontsize=8)
        
        # Add legend for color coding
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='Top 5 bands'),
                          Patch(facecolor='orange', alpha=0.7, label='Bands 6-10'),
                          Patch(facecolor='skyblue', alpha=0.7, label='Remaining bands')]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Original band order with ranking indicators
        xs = np.arange(len(imp))
        bars2 = ax2.bar(xs, imp_norm, color=['red' if i in order[:5] else 'orange' if i in order[:10] else 'lightgray' for i in range(len(imp))], alpha=0.7)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel("Normalized Importance")
        ax2.set_title("Band Importance in Original Order (Colored by Rank)", fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add rank numbers on top of bars
        for i, (bar, rank_pos) in enumerate(zip(bars2, np.argsort(order))):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'#{rank_pos + 1}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        fig_path = save_dir / "ranked_band_importance.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved ranked bands visualization to {fig_path}")
        
        # Create a separate focused plot for top 10 bands
        plt.figure(figsize=(12, 6))
        top_10_indices = order[:10]
        top_10_labels = [f"{labels[i]} (#{j+1})" for j, i in enumerate(top_10_indices)]
        top_10_importance = [imp_norm[i] for i in top_10_indices]
        
        colors_top10 = plt.cm.viridis(np.linspace(0, 1, 10))
        bars = plt.bar(range(10), top_10_importance, color=colors_top10, alpha=0.8)
        plt.xticks(range(10), top_10_labels, rotation=45, ha='right')
        plt.ylabel("Normalized Importance")
        plt.title("Top 10 Most Important Bands", fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, top_10_importance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        top10_path = save_dir / "top_10_bands.png"
        plt.savefig(top10_path, dpi=300, bbox_inches='tight')
        print(f"Saved top 10 bands plot to {top10_path}")
        
        plt.show()

if __name__ == "__main__":
    main()
