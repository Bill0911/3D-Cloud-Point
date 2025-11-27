import numpy as np
import laspy
import csv
import json
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes


# Removed: from shapely.geometry import MultiPoint, Polygon
# Removed: from shapely.ops import unary_union


# --------------------------------------------------------------------
# ROOM ID LOADING
# --------------------------------------------------------------------
def load_room_ids(las):
    if hasattr(las, "room_id"):
        return np.asarray(las.room_id, dtype=np.int32)
    if "room_id" in las.point_format.extra_dimension_names:
        return np.asarray(las["room_id"], dtype=np.int32)
    return None


# --------------------------------------------------------------------
# GRID BUILDING (Raw 2D Pixel Map)
# --------------------------------------------------------------------
def build_grid(points_xy, room_ids, cell):
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x_min, y_min = np.min(points_xy, axis=0)
    x_max, y_max = np.max(points_xy, axis=0)

    w = int(np.ceil((x_max - x_min) / cell)) + 2
    h = int(np.ceil((y_max - y_min) / cell)) + 2

    grid = np.zeros((h, w), dtype=np.int32)

    cx = ((x - x_min) / cell).astype(int)
    cy = ((y - y_min) / cell).astype(int)
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    # best label per pixel (majority vote resolution)
    cell_idx = cy * w + cx
    mask = room_ids > 0

    if np.any(mask):
        pairs = np.column_stack((cell_idx[mask], room_ids[mask]))
        uniq, counts = np.unique(pairs, axis=0, return_counts=True)

        best = {}
        for (idx, rid), c in zip(uniq, counts):
            if idx not in best or c > best[idx][1]:
                best[idx] = (rid, c)

        for idx, (rid, _) in best.items():
            gy = idx // w
            gx = idx % w
            grid[gy, gx] = rid

    # gap closing (fill holes within the grid boundary)
    mask = grid > 0
    mask = binary_fill_holes(mask)
    grid = mask.astype(int) * grid

    return grid, (x_min, y_min, x_max, y_max)


# --------------------------------------------------------------------
# AREA STATS
# --------------------------------------------------------------------
def compute_stats(grid, cell):
    out = {}
    ids, counts = np.unique(grid, return_counts=True)

    for rid, cnt in zip(ids, counts):
        if rid == 0:
            continue

        out[int(rid)] = {
            "pixel_count": int(cnt),
            "area_m2": float(cnt * (cell * cell))
        }

    return out


# --------------------------------------------------------------------
# JSON EXPORT (Outputs Bounding Box and Centroid of the grid cells)
# --------------------------------------------------------------------
def save_json(stats, grid, bounds, cell, out_path):
    x_min, y_min, _, _ = bounds

    data = {"rooms": []}

    for rid, row in stats.items():
        ys, xs = np.where(grid == rid)
        if len(xs) < 3:
            continue

        # Calculate Bounding Box and Centroid based on grid pixels
        xmin_g, xmax_g = xs.min(), xs.max()
        ymin_g, ymax_g = ys.min(), ys.max()
        mx_g, my_g = xs.mean(), ys.mean()

        # Convert grid indices to world coordinates
        xmin = x_min + xmin_g * cell
        ymin = y_min + ymin_g * cell
        xmax = x_min + (xmax_g + 1) * cell
        ymax = y_min + (ymax_g + 1) * cell

        # Centroid
        cx = x_min + (mx_g + 0.5) * cell
        cy = y_min + (my_g + 0.5) * cell

        data["rooms"].append({
            "room_id": rid,
            "area_sqm": row["area_m2"],
            "centroid": [float(cx), float(cy)],
            "bbox": [[float(xmin), float(ymin)], [float(xmax), float(ymax)]],
            "boundary_type": "grid_based"  # Indicate that no smoothing was used
        })

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------
# PNG MAP (Plots the blocky grid using plt.imshow)
# --------------------------------------------------------------------
def plot_map(grid, bounds, stats, out_path):
    x_min, y_min, x_max, y_max = bounds
    extent = (x_min, x_max, y_min, y_max)

    cmap = plt.get_cmap("tab20")

    plt.figure(figsize=(12, 12))

    ids = [rid for rid in np.unique(grid) if rid > 0]
    # Create an index-based color map for plt.imshow
    id_to_color_idx = {rid: i + 1 for i, rid in enumerate(ids)}
    color_grid = np.zeros_like(grid, dtype=np.int32)
    for rid, idx in id_to_color_idx.items():
        color_grid[grid == rid] = idx

    # CRITICAL: interpolation='nearest' ensures the blocky, grid-cell look
    plt.imshow(color_grid, origin="lower", extent=extent, interpolation='nearest', cmap=cmap)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Room Areas (m²) - Grid-based")
    plt.colorbar(ticks=np.arange(1, len(ids) + 1),
                 label="Room ID",
                 boundaries=np.arange(len(ids) + 2) - 0.5)

    # Annotate
    for rid, row in stats.items():
        ys, xs = np.where(grid == rid)
        if len(xs) == 0:
            continue

        # Calculate centroid in world coordinates for labeling
        cell_w = (x_max - x_min) / grid.shape[1]
        cell_h = (y_max - y_min) / grid.shape[0]
        mx_g, my_g = xs.mean(), ys.mean()

        px = x_min + (mx_g + 0.5) * cell_w
        py = y_min + (my_g + 0.5) * cell_h

        plt.text(px, py, f"{rid}\n{row['area_m2']:.1f} m²",
                 ha="center", va="center",
                 fontsize=8,
                 bbox=dict(facecolor="white", alpha=0.7))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------------------------------------------------
# MAIN ENTRYPOINT
# --------------------------------------------------------------------
def run_measurement(segmented_file, png_out, csv_out, json_out, cfg):
    las = laspy.read(segmented_file)
    points = np.vstack((las.x, las.y, las.z)).T

    room_ids = load_room_ids(las)
    if room_ids is None:
        raise RuntimeError("room_id missing → segmentation failed")

    # Filter only valid rooms
    mask = room_ids > 0

    # Ceiling-only filtering (using classification = 7)
    if cfg["ceiling_only"] and hasattr(las, "classification"):
        cls = np.asarray(las.classification)
        mask &= (cls == 7)

    pts2d = points[mask, :2]
    ids = room_ids[mask]

    grid, bounds = build_grid(pts2d, ids, cfg["grid_size"])
    stats = compute_stats(grid, cfg["grid_size"])

    # Filter out tiny noise rooms
    min_cells = cfg["min_cells_per_room"]
    for rid in list(stats.keys()):
        if stats[rid]["pixel_count"] < min_cells:
            # Remove from grid and stats
            grid[grid == rid] = 0
            del stats[rid]

    # CSV
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "pixel_count", "area_m2"])
        for rid, row in stats.items():
            w.writerow([rid, row["pixel_count"], row["area_m2"]])

    # JSON (Saves bounds/centroid)
    save_json(stats, grid, bounds, cfg["grid_size"], json_out)

    # PNG (Plots blocky map)
    plot_map(grid, bounds, stats, png_out)

    return stats