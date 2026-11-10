"""Paper figure for the default BrainVLN v2 forward pass (B5, conv, pooled).

The left side is the data path in Frontend and BrainVLNModel._external.
The photograph is only an anatomical key for the four populations.
Cross-population routes are the pairs in synapses.ALLOWED_STRUCTURED.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent
LATERAL = OUT / "BrainVLN_v2_cortex_lateral.jpg"

INK = "#1c2430"
MUTED = "#5c6b7a"
ARROW = "#243044"
REGION = {
    "goal": "#C47E12",
    "action": "#1C8A55",
    "spatial": "#2C64C8",
    "episodic": "#C43766",
}
WASH = {
    "goal": np.array([232, 168, 64], dtype=np.float32),
    "action": np.array([64, 168, 112], dtype=np.float32),
    "spatial": np.array([72, 124, 214], dtype=np.float32),
    "episodic": np.array([214, 92, 132], dtype=np.float32),
}


def prepare_cortex():
    """Crop the lateral photograph and lay a light wash over four territories.

    Dark sulci keep their luminance, so the folds stay visible.
    """
    image = np.asarray(Image.open(LATERAL).convert("RGB")).astype(np.float32)
    tissue = image.min(axis=-1) < 236
    ys, xs = np.where(tissue)
    pad = 18
    y0, y1 = max(0, ys.min() - pad), min(image.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(image.shape[1], xs.max() + pad)
    crop = image[y0:y1, x0:x1]
    local = tissue[y0:y1, x0:x1]
    height, width = crop.shape[:2]
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    brain_y, brain_x = np.where(local)
    nx = (grid_x - brain_x.min()) / max(1, brain_x.max() - brain_x.min())
    ny = (grid_y - brain_y.min()) / max(1, brain_y.max() - brain_y.min())
    # Anterior is left. The Sylvian fissure climbs slightly toward the occiput.
    sylvian = 0.585 + 0.07 * np.clip(nx - 0.25, 0, 1)
    above = ny < sylvian
    names = np.full((height, width), "episodic", dtype=object)
    names[above & (nx < 0.40)] = "goal"
    names[above & (nx >= 0.40) & (nx < 0.52)] = "action"
    names[above & (nx >= 0.52)] = "spatial"
    # Soft boundaries. The photograph stays dominant; color only marks territory.
    color_image = np.zeros_like(crop)
    for name, color in WASH.items():
        color_image[names == name] = color
    color_image = ndimage.gaussian_filter(color_image, sigma=10)
    alpha = np.where(local, 0.16, 0.0).astype(np.float32)
    alpha = ndimage.gaussian_filter(alpha, sigma=6)
    painted = crop * (1.0 - alpha[..., None]) + color_image * alpha[..., None]
    anchors = {}
    for name, side in (("goal", "left"), ("action", "top"), ("spatial", "right"), ("episodic", "bottom")):
        mask = local & (names == name)
        yy, xx = np.where(mask)
        if side == "left":
            pick = int(np.argmin(xx))
        elif side == "right":
            pick = int(np.argmax(xx))
        elif side == "top":
            pick = int(np.argmin(yy))
        else:
            pick = int(np.argmax(yy))
        anchors[name] = np.array([xx[pick], yy[pick]], dtype=float)
    return np.clip(painted, 0, 255).astype(np.uint8), anchors


def _card(ax, x, y, w, h, face, edge, title, body="", title_color=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.15,
            zorder=3,
        )
    )
    ax.add_patch(Rectangle((x, y), 0.005, h, facecolor=edge, edgecolor="none", zorder=4))
    ax.text(x + 0.012, y + h - 0.016, title, fontsize=8.0, fontweight="semibold", color=title_color, va="top", ha="left", zorder=4)
    if body:
        ax.text(x + 0.012, y + h - 0.038, body, fontsize=6.7, color=MUTED, va="top", ha="left", linespacing=1.25, zorder=4)


def _seg(fig, a, b, color=ARROW, head=False, ls="-"):
    fig.add_artist(
        FancyArrowPatch(
            a,
            b,
            transform=fig.transFigure,
            arrowstyle="-|>" if head else "-",
            mutation_scale=8,
            linewidth=1.25,
            linestyle=ls,
            color=color,
            shrinkA=0,
            shrinkB=0,
            clip_on=False,
            zorder=5,
            joinstyle="miter",
        )
    )


def _elbow(fig, points, color=ARROW):
    for start, end in zip(points[:-2], points[1:-1]):
        _seg(fig, start, end, color, head=False)
    _seg(fig, points[-2], points[-1], color, head=True)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    cortex, anchors = prepare_cortex()
    image_h, image_w = cortex.shape[:2]
    fig_w, fig_h = 17.6, 8.6
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=180, facecolor="white")
    board = fig.add_axes([0, 0, 1, 1], zorder=4)
    board.set_xlim(0, 1)
    board.set_ylim(0, 1)
    board.axis("off")

    fig.text(0.02, 0.965, "BrainVLN v2", fontsize=15.5, fontweight="semibold", color=INK, va="center")
    fig.text(0.145, 0.965, "B5  ·  conv frontend  ·  pooled layout  ·  4096 LIF/ALIF", fontsize=8, color=MUTED, va="center")

    # --- column 1–3: one horizontal arrow per row, plus one vertical drop ---
    _card(board, 0.018, 0.845, 0.112, 0.078, "#F4F7FB", "#9AABBE", "RGB", "[B, 3, 224, 224]")
    _card(board, 0.018, 0.748, 0.112, 0.078, "#F4F7FB", "#9AABBE", "Depth", "[B, 1, 224, 224]\n× depth_valid")
    _card(board, 0.150, 0.845, 0.128, 0.078, "#F7F8FA", "#C5CED6", "RGB ConvEncoder", "7×7/4, 3×3/4, 3×3/2\n→ 64-d")
    _card(board, 0.150, 0.748, 0.128, 0.078, "#F7F8FA", "#C5CED6", "Depth ConvEncoder", "same tower\nmasked depth")
    _card(board, 0.300, 0.748, 0.132, 0.175, "#EAF2FB", "#7FA4D4", "RGBDFusion", "concat\nLinear 128 → 64\ncross-attention off")
    _card(board, 0.018, 0.630, 0.112, 0.078, "#F3FBF6", "#8FBF9F", "Previous action", "id 0–4\nstart id = 4")
    _card(board, 0.150, 0.630, 0.128, 0.078, "#F3FBF6", "#8FBF9F", "Action embedding", "Embedding(5, 64)")
    _card(board, 0.018, 0.490, 0.112, 0.078, "#FFF6E8", "#E2B15A", "Instruction", "tokens [B, L]\ntoken_mask")
    _card(board, 0.150, 0.490, 0.128, 0.078, "#FFF6E8", "#E2B15A", "Language", "masked mean → 64\nsubinstruction = 0")
    _card(board, 0.300, 0.490, 0.132, 0.078, "#FFF6E8", REGION["goal"], "goal_proj", "Linear 128 → 768", title_color=REGION["goal"])
    _card(board, 0.300, 0.600, 0.132, 0.080, "#FDEFF3", REGION["episodic"], "EventMemory", "surprise > 0.5\nwrite = fused × gate", title_color=REGION["episodic"])

    # Population targets. These are the only boxes the projections enter.
    _card(board, 0.458, 0.800, 0.130, 0.078, "#F4F8FF", REGION["spatial"], "Spatial  1536", "spatial_proj(fused ∥ history)", title_color=REGION["spatial"])
    _card(board, 0.458, 0.600, 0.130, 0.080, "#FFF0F4", REGION["episodic"], "Episodic  1280", "episodic_proj  64 → 1280", title_color=REGION["episodic"])
    _card(board, 0.600, 0.600, 0.125, 0.080, "#FFF0F4", REGION["episodic"], "Progress", "Linear(Episodic)\n1280 → 1", title_color=REGION["episodic"])
    _card(board, 0.458, 0.400, 0.130, 0.078, "#FFF8EE", REGION["goal"], "Goal  768", "instruction current", title_color=REGION["goal"])
    _card(board, 0.458, 0.280, 0.130, 0.078, "#F3FBF6", REGION["action"], "Action  512", "external current = 0", title_color=REGION["action"])

    # Horizontal links.
    for y in (0.884, 0.787, 0.669, 0.529):
        _seg(fig, (0.130, y), (0.150, y), head=True)
    _seg(fig, (0.278, 0.884), (0.300, 0.860), head=True)
    _seg(fig, (0.278, 0.787), (0.300, 0.800), head=True)
    _seg(fig, (0.278, 0.529), (0.300, 0.529), head=True)
    _elbow(fig, [(0.278, 0.669), (0.290, 0.710), (0.440, 0.710), (0.440, 0.839), (0.458, 0.839)], REGION["spatial"])
    _seg(fig, (0.432, 0.835), (0.458, 0.839), REGION["spatial"], head=True)
    _elbow(fig, [(0.432, 0.529), (0.445, 0.529), (0.445, 0.439), (0.458, 0.439)], REGION["goal"])
    _seg(fig, (0.432, 0.640), (0.458, 0.640), REGION["episodic"], head=True)
    _seg(fig, (0.588, 0.640), (0.600, 0.640), REGION["episodic"], head=True)
    # One vertical drop: fusion features into the surprise gate.
    _elbow(fig, [(0.366, 0.748), (0.366, 0.680)], REGION["episodic"])

    board.text(0.441, 0.755, "history", fontsize=6.4, color=REGION["spatial"], ha="center", va="center", rotation=90, zorder=6)
    board.text(0.444, 0.860, "fused", fontsize=6.4, color=REGION["spatial"], ha="left", va="center", zorder=6,
               bbox=dict(facecolor="white", edgecolor="none", pad=0.1))

    # Readout sits under the two populations that actually enter it.
    _card(board, 0.458, 0.145, 0.130, 0.090, "#F7F4EF", "#B7A898", "Readout", "Linear(Action ∥ Goal)\n→ 4 logits")
    _card(board, 0.608, 0.145, 0.150, 0.090, "#F7F8FA", "#C5CED6", "Actions", "STOP · MOVE_FORWARD\nTURN_LEFT · TURN_RIGHT")
    _elbow(fig, [(0.560, 0.400), (0.598, 0.400), (0.598, 0.190), (0.588, 0.190)], REGION["goal"])
    _seg(fig, (0.523, 0.280), (0.523, 0.235), REGION["action"], head=True)
    _seg(fig, (0.588, 0.190), (0.608, 0.190), head=True)

    # Photograph. Size the axes to the crop so the picture is not stretched.
    box_w = 0.255
    box_h = box_w * (image_h / image_w) * (fig_w / fig_h)
    brain_box = (0.728, 0.30, box_w, box_h)
    brain = fig.add_axes(brain_box, zorder=1)
    brain.imshow(cortex, interpolation="lanczos")
    brain.set_axis_off()

    def page(point):
        x0, y0, w, h = brain_box
        return x0 + point[0] / image_w * w, y0 + (1.0 - point[1] / image_h) * h

    # Labels sit just outside the silhouette. Leaders do not cross the data path.
    # Place names in the white margin of the photograph, one per side.
    x0, y0, bw, bh = brain_box
    placements = {
        "goal": (x0 + 0.04, y0 + bh + 0.012, "frontal  ·  Goal"),
        "action": (x0 + bw * 0.46, y0 + bh + 0.012, "precentral  ·  Action"),
        "spatial": (x0 + bw - 0.01, y0 - 0.008, "parieto-occipital · Spatial"),
        "episodic": (x0 + 0.02, y0 - 0.008, "temporal · Episodic"),
    }
    for name, (lx, ly, title) in placements.items():
        px, py = page(anchors[name])
        board.text(
            lx,
            ly,
            title,
            fontsize=6.6,
            fontweight="semibold",
            color=REGION[name],
            ha="right" if name == "spatial" else "center",
            va="bottom" if name in ("goal", "action") else "top",
            zorder=6,
        )
        _seg(fig, (lx, ly), (px, py), REGION[name], head=False, ls=(0, (2, 1.5)))

    # Synapses, separated from the sensory path so the two graphs are not mixed.
    board.text(0.018, 0.075, "SynapticGraph", fontsize=8.0, fontweight="semibold", color=INK, va="center")
    routes = [
        (0.018, "spatial", "Spatial → Goal\nSpatial → Episodic\nSpatial → Action\nSpatial ↺ local"),
        (0.175, "goal", "Goal → Episodic\nGoal → Action\nno self-edges"),
        (0.332, "episodic", "Episodic → Goal\nEpisodic → Action\nEpisodic ↺ recurrent"),
        (0.489, "action", "Action ↺ recurrent\nno cross-population\noutgoing edges"),
    ]
    for x, name, body in routes:
        _card(board, x, 0.012, 0.148, 0.052, "white", REGION[name], name.capitalize(), "", title_color=REGION[name])
        board.text(x + 0.058, 0.038, body, fontsize=6.15, color=MUTED, va="center", ha="left", linespacing=1.2, zorder=4)

    board.text(
        0.655,
        0.040,
        "2 LIF/ALIF steps per navigation step\ndelay 1–4   ·   out-degree 48   ·   80% excitatory\nclosed loop: action → next RGB, depth, previous action",
        fontsize=6.5,
        color=MUTED,
        va="center",
        ha="left",
        linespacing=1.35,
    )

    png = OUT / "BrainVLN_v2_paper_figure.png"
    pdf = OUT / "BrainVLN_v2_paper_figure.pdf"
    fig.savefig(png, dpi=190, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    print(brain_box)
    print(png)


if __name__ == "__main__":
    main()
