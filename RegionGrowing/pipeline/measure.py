import numpy as np
import laspy
import csv
import json
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from scipy.ndimage import binary_dilation


def load_room_ids(las):
    if hasattr(las, "room_id"):
        return np.asarray(las.room_id, dtype=np.int32)
    if "room_id" in las.point_format.extra_dimension_names:
        return np.asarray(las["room_id"], dtype=np.int32)
    return None


def build_grid(points_xy, room_ids, cell_size):
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x_min, y_min = np.min(points_xy, axis=0)
    x_max, y_max = np.max(points_xy, axis=0)

    w = int(np.ceil((x_max - x_min) / cell_size)) + 1
    h = int(np.ceil((y_max - y_min) / cell_size)) + 1

    grid = np.zeros((h, w), dtype=np.int32)

    cx = np.floor((x - x_min) / cell_size).astype(int)
    cy = np.floor((y - y_min) / cell_size).astype(int)
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)

    cell_idx = cy * w + cx
    mask = room_ids > 0

    if np.any(mask):
        combos = np.vstack((cell_idx[mask], room_ids[mask])).T
        uniq, counts = np.unique(combos, axis=0, return_counts=True)

        cell_best = {}
        for (cidx, rid), c in zip(uniq, counts):
            if cidx not in cell_best or c > cell_best[cidx][1]:
                cell_best[cidx] = (rid, c)

        for cidx, (rid, _) in cell_best.items():
            gy = cidx // w
            gx = cidx % w
            grid[gy, gx] = int(rid)

    # light dilation: fills minor holes
    grid = binary_dilation(grid > 0, iterations=1).astype(int) * grid

    bounds = (x_min, y_min, x_max, y_max)
    return grid, bounds


def compute_area_stats(grid, cell_size):
    unique_ids, counts = np.unique(grid, return_counts=True)

    out = {}
    for rid, cnt in zip(unique_ids, counts):
        if rid == 0:
            continue
        out[int(rid)] = {
            "pixel_count": int(cnt),
            "area_m2": float(cnt * (cell_size ** 2))
        }
    return out


def alpha_shape(points, alpha=1.0):
    if len(points) < 4:
        return MultiPoint(points).convex_hull
    try:
        from shapely.ops import triangulate
        triangles = triangulate(points)
        alpha_polygons = [
            tri for tri in triangles
            if tri.area > 0 and tri.length / tri.area < alpha * 10
        ]
        return unary_union(alpha_polygons).convex_hull
    except Exception:
        return MultiPoint(points).convex_hull


def save_json_polygons(stats, grid, bounds, cell, out_path):
    x_min, y_min, _, _ = bounds

    data = {"rooms": []}

    for rid in sorted(stats.keys()):
        ys, xs = np.where(grid == rid)
        if len(xs) < 6:
            continue

        pts = np.column_stack([
            x_min + xs * cell + (cell / 2),
            y_min + ys * cell + (cell / 2)
        ])

        hull = alpha_shape(MultiPoint(pts), alpha=1.1)

        if not isinstance(hull, Polygon):
            hull = hull.convex_hull

        coords = [[float(round(a, 3)), float(round(b, 3))]
                  for a, b in hull.exterior.coords]

        centroid = hull.centroid
        xmin, ymin, xmax, ymax = hull.bounds

        data["rooms"].append({
            "room_id": int(rid),
            "area_sqm": float(stats[rid]["area_m2"]),
            "centroid": [float(centroid.x), float(centroid.y)],
            "bbox": [[float(xmin), float(ymin)], [float(xmax), float(ymax)]],
            "polygon": coords
        })

    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)


def save_csv(stats, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["room_id", "pixel_count", "area_m2"])
        for rid in sorted(stats.keys()):
            s = stats[rid]
            w.writerow([rid, s["pixel_count"], f"{s['area_m2']:.4f}"])


def plot_map(grid, bounds, stats, out_path):
    x_min, y_min, x_max, y_max = bounds
    extent = (x_min, x_max, y_min, y_max)

    plt.figure(figsize=(10, 10))

    unique_ids = np.unique(grid)
    unique_ids = unique_ids[unique_ids > 0]
    color_idx = {rid: i + 1 for i, rid in enumerate(unique_ids)}

    heat = np.zeros_like(grid)
    for rid, idx in color_idx.items():
        heat[grid == rid] = idx

    plt.imshow(heat, origin="lower", extent=extent, interpolation="nearest")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Room Map")

    plt.savefig(out_path, dpi=300)
    plt.close()


def run_measurement(segmented_file, png_out, csv_out, json_out, cfg):
    las = laspy.read(segmented_file)
    pts = np.vstack((las.x, las.y, las.z)).T

    room_ids = load_room_ids(las)
    if room_ids is None:
        raise RuntimeError("segment.py did not assign room_id.")

    mask = room_ids > 0

    if cfg["ceiling_only"] and hasattr(las, "classification"):
        cls = np.asarray(las.classification)
        mask &= (cls == 7)

    pts2d = pts[mask, :2]
    ids = room_ids[mask]

    grid, bounds = build_grid(pts2d, ids, cfg["grid_size"])
    stats = compute_area_stats(grid, cfg["grid_size"])

    # remove tiny junk rooms
    min_cells = cfg["min_cells_per_room"]
    for rid in list(stats.keys()):
        if stats[rid]["pixel_count"] < min_cells:
            grid[grid == rid] = 0
            del stats[rid]

    save_csv(stats, csv_out)
    save_json_polygons(stats, grid, bounds, cfg["grid_size"], json_out)
    plot_map(grid, bounds, stats, png_out)

    return stats
