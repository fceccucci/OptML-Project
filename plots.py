#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Heat-maps for four FL algorithms (comparable colour scale) —
only the *last* plot gets the shared colour-bar.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------ AXIS LABELS ---------------------------------
alpha_labels = [1000, 1, 0.1, 0.001]
C_labels     = [1.0, 0.5, 0.1]

# --------------------------- DRAWING HELPER ---------------------------------
def make_combined_heatmap(alg_name: str,
                          E1: np.ndarray,
                          E5: np.ndarray,
                          vmin: float,
                          vmax: float,
                          add_cbar: bool,
                          save_dir: Path = Path(".")) -> Path:
    """
    Draw a single heat-map (E=1 left, E=5 right).
    If add_cbar=True, put one colour-bar below the plot.
    """
    combined = np.hstack([E1, E5])
    mid = (vmin + vmax) / 2

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(combined, vmin=vmin, vmax=vmax, cmap="viridis")

    # ticks
    ax.set_xticks(np.arange(combined.shape[1]))
    ax.set_xticklabels(alpha_labels + alpha_labels)
    ax.set_xlabel(r"$\alpha$")
    ax.set_yticks(np.arange(len(C_labels)))
    ax.set_yticklabels(C_labels)
    ax.set_ylabel("Client Fraction $C$")

    # annotate numbers
    for i in range(combined.shape[0]):
        for j in range(combined.shape[1]):
            ax.text(j, i, f"{combined[i,j]:.3f}",
                    ha="center", va="center",
                    color=("white" if combined[i,j] < mid else "black"),
                    fontsize=8)

    # divider & sub-titles
    split = len(alpha_labels) - 0.5
    ax.axvline(split, color="white", linewidth=2)
    ax.text(split/2,           -0.7, r"$E = 1$", ha="center", va="center", fontsize=12)
    ax.text(split + split/2,   -0.7, r"$E = 5$", ha="center", va="center", fontsize=12)

    # OPTIONAL colour-bar (only for last plot)
    if add_cbar:
        # put it **below** the axes (feel free to move/resize)
        cbar_ax = fig.add_axes([0.15, -0.12, 0.7, 0.05])   # left, bottom, width, height
        cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.set_xlabel("Accuracy")

    ax.set_title(f"{alg_name}: Test Accuracy", pad=25, fontsize=14)

    save_path = save_dir / f"{alg_name}_heatmap.pdf"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


# --------------------------- (FAKE) DATA ------------------------------------
algorithms = ["FedAvg", "FedAvgM", "FedProx", "FedYogi"]
rng = np.random.default_rng(seed=42)
E1_dict, E5_dict = {}, {}

AVG_E1 = np.array([
    #1000, 1, 0.1, 0.001
    [0.975100000447035, 0.973000001525879, 0.73430000236705 , 0.113500000017881],  # C = 1
    [0.977528099490877, 0.975261252237602, 0.798887468507823, 0.222988958990536],  # C = 0.5
    [0.970193743083942, 0.976253306960053, 0.919975190393386, 0.081267780959242],  # C = 0.1
])

AVG_E5 = np.array([
    [0.985899999511242, 0.984100007021427, 0.833400001439452, 0.103523740875900],  # C = 1
    [0.986870904767492, 0.987526004987186, 0.719964852653544, 0.271353166986564],  # C = 0.5
    [0.979848869561548, 0.988304106088785, 0.736026566960103, 0.052376086653874],  # C = 0.1
])
E1_dict["FedAvg"] = np.array(AVG_E1)
E5_dict["FedAvg"] = np.array(AVG_E5)

AVGM_E1 = np.array([
    [0.97370000, 0.97680000, 0.79090000, 0.08920000],   # C = 1
    [0.97049378, 0.97393413, 0.46404109, 0.27784578],   # C = 0.5
    [0.97092730, 0.96993168, 0.95116772, 0.01879000],    # C = 0.1 
])

AVGM_E5 = np.array([
    [0.97800001, 0.98080000, 0.94209999, 0.08930000],   # C = 1
    [0.98127118, 0.98102067, 0.89561802, 0.04868050],   # C = 0.5
    [0.98008961, 0.97881997, 0.87067913, 0.01768210],    # C = 0.1
])

E1_dict["FedAvgM"] = np.array(AVGM_E1)
E5_dict["FedAvgM"] = np.array(AVGM_E5)

PROX_E1 = np.array([
    [0.97270000, 0.97560001, 0.77920000, 0.10090000],   # C = 1
    [0.97310856, 0.97957596, 0.75850475, 0.05678760],    # C = 0.5
    [0.97713720, 0.96358838, 0.49227272, 0.08776241],    # C = 0.1
])

PROX_E5 = np.array([
    [0.98430001, 0.98530001, 0.82950001, 0.09580000],   # C = 1
    [0.98538830, 0.98194946, 0.96574958, 0.07230009],    # C = 0.5
    [0.98269893, 0.98861387, 0.98651133, 0.49412341],   # C = 0.1
])
E1_dict["FedProx"] = np.array(PROX_E1)
E5_dict["FedProx"] = np.array(PROX_E5)

YOGI_E1 = [
  #1000, 1, 0.1, 0.001
 [1,    2,  3,   4], # C: 1
 [5,    6,  7,   8], # C: 0.5
 [9,    10, 11, 12], # C: 0.1
]
YOGI_E5 = [
  #1000, 1, 0.1, 0.001
 [1,    2,  3,   4], # C: 1
 [5,    6,  7,   8], # C: 0.5
 [9,    10, 11, 12], # C: 0.1
]
E1_dict["FedYogi"] = np.array(YOGI_E1)
E5_dict["FedYogi"] = np.array(YOGI_E5)


# for alg in algorithms:
#     base = rng.uniform(0.60, 0.85, size=(len(C_labels), len(alpha_labels)))
#     E1_dict[alg] = np.round(base, 3)
#     E5_dict[alg] = np.round(base + rng.uniform(-0.02, 0.03, size=base.shape), 3)

# global colour range
# all_vals = np.concatenate([*E1_dict.values(), *E5_dict.values()])
# vmin, vmax = all_vals.min(), all_vals.max()



vmin, vmax = 0.05, 0.99
print(f"Shared colour scale → vmin={vmin:.3f}, vmax={vmax:.3f}")

# ------------------------------ PLOTTING ------------------------------------
out_dir = Path(".")

for idx, alg in enumerate(algorithms):
    add_cbar = (idx == len(algorithms) - 1)   # only last one gets colour-bar
    path = make_combined_heatmap(
        alg,
        E1_dict[alg],
        E5_dict[alg],
        vmin, vmax,
        add_cbar=add_cbar,
        save_dir=out_dir
    )
    print(f"Saved {alg} → {path.resolve()}")
