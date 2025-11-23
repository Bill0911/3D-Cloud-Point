#!/usr/bin/env python3
"""
Complete Room Analysis Pipeline

Executes all three steps:
1. Classification of walls, floors, and ceilings (region growing)
2. Segmentation of rooms based on ceiling layout
3. Calculation of room areas and generation of 2D map

Usage:
    python complete_room_analysis.py input.laz output_folder [options]

Output files in output_folder:
    - classified.las: Classified point cloud
    - segmented.las: Point cloud with room_id per room
    - room_map.png: 2D map with room labels and areas
    - room_areas.csv: CSV with room areas
"""

import argparse
import os
import sys
from pathlib import Path

# Import functies uit de drie scripts
import laspy
import numpy as np
import open3d as o3d
from collections import deque
from sklearn.neighbors import KDTree
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt
import csv
import json
from shapely.geometry import MultiPoint


# ============================================================================
# Region Growing Classification
# ============================================================================

def region_growing(points, normals, kd_tree, seed_idx, visited, 
                   normal_threshold=0.9, distance_threshold=0.1, min_region_size=1200, max_z_diff=None):
    """Perform region growing from a seed point."""
    region = []
    queue = deque([seed_idx])
    visited[seed_idx] = True

    # seed_normal = normals[seed_idx]
    
    while queue:
        current_idx = queue.popleft()
        region.append(current_idx)

        current_point = points[current_idx];
        current_normal = normals[current_idx];
        
        neighbor_indices = kd_tree.query_radius(
            points[current_idx].reshape(1, -1), 
            r=distance_threshold
        )[0]
        
        for neighbor_idx in neighbor_indices:
            if visited[neighbor_idx]:
                continue

            if max_z_diff is not None:
                if abs(points[neighbor_idx, 2] - current_point[2]) > max_z_diff:
                    continue

            dot_product = np.dot(normals[neighbor_idx], current_normal)

            if dot_product > normal_threshold:
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)
    
    return region if len(region) >= min_region_size else []


def classify_region(points_in_region, normals_in_region, z_stats):
    """Classify a region as wall, floor, or ceiling."""
    mean_normal = np.mean(normals_in_region, axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    
    nz = mean_normal[2]
    mean_z = np.mean(points_in_region[:, 2])
    
    if abs(nz) > 0.85:
        if mean_z < z_stats['threshold_floor']:
            return 2
        elif mean_z > z_stats['threshold_ceiling']:
            return 7
        else:
            return 7 if nz > 0 else 2
    elif abs(nz) < 0.2:
        return 6
    else:
        return 1 


def step1_classify(input_file,
                   output_file,
                   voxel_size=0.02,
                   normal_threshold=0.7,
                   distance_threshold=0.15,
                   min_region_size=500,
                   floor_pct=1.0,
                   ceil_pct=99.0,
                   max_z_diff=None):

    print("\n" + "=" * 70)
    print("STEP 1: CLASSIFICATION (Region Growing)")
    print("=" * 70)

    print(f"Reading point cloud from {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Original points: {len(points)}")

    print("Downsampling point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_down = np.asarray(pcd.points)
    print(f"Points after downsampling: {len(points_down)}")

    if len(points_down) == 0:
        raise RuntimeError("Downsampling produced 0 points. Check voxel_size.")

    print("Estimating normals on downsampled cloud...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
    )
    pcd.orient_normals_consistent_tangent_plane(k=20)
    normals = np.asarray(pcd.normals)

    print("Building KD-tree...")
    kd_tree = KDTree(points_down)

    z_vals = points_down[:, 2]
    z_min = float(np.min(z_vals))
    z_max = float(np.max(z_vals))
    z_floor = float(np.percentile(z_vals, floor_pct))
    z_ceil = float(np.percentile(z_vals, ceil_pct))
    z_range = z_max - z_min

    # Margins inside floor/ceiling bands (10% of span between percentiles)
    span = max(z_ceil - z_floor, 1e-6)
    floor_margin = 0.10 * span
    ceil_margin = 0.10 * span

    z_stats = {
        "min": z_min,
        "max": z_max,
        "range": z_range,
        "floor_z": z_floor,
        "ceiling_z": z_ceil,
        "threshold_floor": z_floor + floor_margin,
        "threshold_ceiling": z_ceil - ceil_margin,
    }

    print(f"Z-range: {z_min:.3f} m to {z_max:.3f} m")
    print(f"Estimated floor (p{floor_pct:.1f}):   {z_floor:.3f} m")
    print(f"Estimated ceiling (p{ceil_pct:.1f}): {z_ceil:.3f} m")
    print(f"Floor threshold:   < {z_stats['threshold_floor']:.3f} m")
    print(f"Ceiling threshold: > {z_stats['threshold_ceiling']:.3f} m")

    print("Performing region growing classification...")
    classification = np.ones(len(points_down), dtype=np.uint8)  # default: unclassified
    visited = np.zeros(len(points_down), dtype=bool)

    # Seeds ordered to prefer very vertical OR very horizontal surfaces first
    normal_z_abs = np.abs(normals[:, 2])
    seed_order = np.argsort(-np.maximum(normal_z_abs, 1 - normal_z_abs))

    regions_found = 0
    points_classified = 0

    for seed_idx in seed_order:
        if visited[seed_idx]:
            continue

        region_indices = region_growing(
            points_down,
            normals,
            kd_tree,
            seed_idx,
            visited,
            normal_threshold=normal_threshold,
            distance_threshold=distance_threshold,
            min_region_size=min_region_size,
            max_z_diff=max_z_diff,
        )

        if not region_indices:
            continue

        region_points = points_down[region_indices]
        region_normals = normals[region_indices]
        region_class = classify_region(region_points, region_normals, z_stats)

        classification[region_indices] = region_class

        regions_found += 1
        points_classified += len(region_indices)

        if regions_found % 10 == 0:
            print(
                f"  Regions found: {regions_found}, "
                f"Points classified: {points_classified}/{len(points_down)}"
            )

    print(f"\nTotal regions found: {regions_found}")
    print(f"Points classified: {points_classified}/{len(points_down)}")

    unique, counts = np.unique(classification, return_counts=True)
    class_names = {1: "Unclassified", 2: "Floor", 6: "Wall", 7: "Ceiling"}

    print("\nClassification statistics (downsampled):")
    for cls, count in zip(unique, counts):
        name = class_names.get(cls, f"Class {cls}")
        pct = 100.0 * count / len(classification)
        print(f"  {name:12s}: {count:7d} points ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # Fallback: ensure we actually have floor (2) and ceiling (7) labels
    # ------------------------------------------------------------------
    total_pts = len(points_down)
    num_floor = np.sum(classification == 2)
    num_ceil = np.sum(classification == 7)

    # Bands based purely on Z thresholds
    floor_band = points_down[:, 2] < z_stats["threshold_floor"]
    ceil_band = points_down[:, 2] > z_stats["threshold_ceiling"]

    if num_ceil == 0:
        new_ceil = np.count_nonzero(ceil_band)
        print(f"\n[Fallback] No ceiling regions from region growing. "
              f"Assigning {new_ceil} points in top Z band as ceiling (7).")
        classification[ceil_band] = 7

    if num_floor == 0:
        new_floor = np.count_nonzero(floor_band)
        print(f"[Fallback] No floor regions from region growing. "
              f"Assigning {new_floor} points in bottom Z band as floor (2).")
        classification[floor_band] = 2

    # Reprint stats after fallback so we see the final distribution
    unique, counts = np.unique(classification, return_counts=True)
    print("\nClassification statistics after fallback (downsampled):")
    for cls, count in zip(unique, counts):
        name = class_names.get(cls, f"Class {cls}")
        pct = 100.0 * count / len(classification)
        print(f"  {name:12s}: {count:7d} points ({pct:5.1f}%)")

    # ------------------------------------------------------------------
    # Extra heuristic: label vertical middle-band points as walls (6)
    # ------------------------------------------------------------------
    # Recompute bands using current z_stats
    z = points_down[:, 2]
    floor_band = z < z_stats["threshold_floor"]
    ceil_band  = z > z_stats["threshold_ceiling"]
    middle_band = ~(floor_band | ceil_band)

    # Vertical surfaces: normals with small |nz|
    nz_all = normals[:, 2]
    vertical = np.abs(nz_all) < 0.4   # 0.4 ≈ tilt > ~66° from horizontal

    wall_candidates = middle_band & vertical

    prev_wall = np.sum(classification == 6)
    classification[wall_candidates] = 6
    new_wall = np.sum(classification == 6)

    print(f"\n[Heuristic] Wall points: {prev_wall} -> {new_wall}")

    # Final stats
    unique, counts = np.unique(classification, return_counts=True)
    class_names = {1: "Unclassified", 2: "Floor", 6: "Wall", 7: "Ceiling"}

    print("\nClassification statistics after fallback + wall heuristic (downsampled):")
    for cls, count in zip(unique, counts):
        name = class_names.get(cls, f"Class {cls}")
        pct = 100.0 * count / len(classification)
        print(f"  {name:12s}: {count:7d} points ({pct:5.1f}%)")

    print(f"\nSaving classified point cloud to {output_file}...")

    # New header with same point format & version, but no pre-set point count
    out_header = laspy.LasHeader(
        point_format=las.header.point_format.id,
        version=las.header.version
    )
    # Preserve scale/offset so coordinates stay consistent
    out_header.scales = las.header.scales
    out_header.offsets = las.header.offsets

    out = laspy.LasData(out_header)

    # Use downsampled coordinates
    out.x = points_down[:, 0]
    out.y = points_down[:, 1]
    out.z = points_down[:, 2]

    # Ensure classification dimension exists and assign
    if "classification" in out.point_format.dimension_names:
        out.classification = classification.astype(np.uint8)
    else:
        raise RuntimeError(
            "Output point format has no 'classification' dimension; "
            "adjust point_format or add an extra dimension here."
        )

    out.write(output_file)
    print(f"✓ Saved {len(points_down)} classified points\n")

# ============================================================================
# Room Segmentation
# ============================================================================

def step2_segment_rooms(input_file, output_file, grid_size=0.02,
                        dilation_iterations=7, min_room_points=2000):

    print("\n" + "=" * 70)
    print("STEP 2: ROOM SEGMENTATION")
    print("=" * 70)

    print(f"Reading classified point cloud from {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    classification = np.array(las.classification)
    print(f"Loaded {len(points)} points.")

    # DEBUG: show class distribution from classified.las
    unique_cls, cls_counts = np.unique(classification, return_counts=True)
    print("Classification histogram (from classified.las):")
    for c, n in zip(unique_cls, cls_counts):
        print(f"  class {int(c)}: {n} points")

    # Masks
    wall_mask = (classification == 6)
    ceiling_mask = (classification == 7)

    wall_points = points[wall_mask]
    ceiling_points = points[ceiling_mask]

    if len(ceiling_points) == 0:
        print("Error: No ceiling points (class 7) found in classified point cloud.")
        return
    if len(wall_points) == 0:
        print("Warning: No wall points (class 6) found. "
              "Segmentation will rely only on ceiling connectivity (no wall barriers).")

    print(f"Found {len(wall_points)} wall points and {len(ceiling_points)} ceiling points.")

    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    grid_width = int(np.ceil((x_max - x_min) / grid_size))
    grid_height = int(np.ceil((y_max - y_min) / grid_size))

    if grid_width <= 0 or grid_height <= 0:
        raise RuntimeError("Invalid grid size or point bounds; got zero-sized grid.")

    print(f"Grid size: {grid_width} x {grid_height} cells (cell = {grid_size:.3f} m)")

    wall_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    if len(wall_points) > 0:
        wall_x = ((wall_points[:, 0] - x_min) / grid_size).astype(int)
        wall_y = ((wall_points[:, 1] - y_min) / grid_size).astype(int)
        wall_x = np.clip(wall_x, 0, grid_width - 1)
        wall_y = np.clip(wall_y, 0, grid_height - 1)
        wall_grid[wall_y, wall_x] = 1

        # Dilate walls to strengthen barriers
        if dilation_iterations > 0:
            print(f"Dilating wall grid ({dilation_iterations} iterations)...")
            wall_grid = binary_dilation(wall_grid, iterations=dilation_iterations).astype(np.uint8)

    ceil_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    ceil_x = ((ceiling_points[:, 0] - x_min) / grid_size).astype(int)
    ceil_y = ((ceiling_points[:, 1] - y_min) / grid_size).astype(int)
    ceil_x = np.clip(ceil_x, 0, grid_width - 1)
    ceil_y = np.clip(ceil_y, 0, grid_height - 1)
    ceil_grid[ceil_y, ceil_x] = 1

    # Candidate cells: have ceiling AND no wall
    segment_grid = np.where((ceil_grid == 1) & (wall_grid == 0), 1, 0).astype(np.uint8)

    # Connected-component labeling on ceiling occupancy
    structure = np.ones((3, 3), dtype=int)
    room_labels_grid, num_labels = label(segment_grid, structure=structure)
    print(f"Initially found {num_labels} raw ceiling-connected components.")

    # Map each ceiling point to its room label from the grid
    point_room_labels = room_labels_grid[ceil_y, ceil_x]

    unique_labels, counts = np.unique(point_room_labels, return_counts=True)
    valid_rooms = 0
    final_room_id_map = {}

    print("Filtering small ceiling components by min_room_points...")
    for raw_id, count in zip(unique_labels, counts):
        if raw_id == 0:
            continue  # label 0 = background
        if count >= min_room_points:
            valid_rooms += 1
            final_room_id_map[raw_id] = valid_rooms
            print(f"  Room {valid_rooms} (raw label {raw_id}) has {count} ceiling points. [VALID]")
        else:
            print(f"  Component (raw label {raw_id}) has {count} ceiling points. [REMOVED]")

    if valid_rooms == 0:
        print("Error: No valid rooms found after filtering by min_room_points.")
        return

    # Apply mapping: raw label -> compact room_id (1..N)
    final_point_labels = np.zeros_like(point_room_labels, dtype=np.uint16)
    for raw_id, final_id in final_room_id_map.items():
        final_point_labels[point_room_labels == raw_id] = final_id

    print(f"\nNumber of valid rooms: {valid_rooms}")

    print(f"Saving segmented point cloud to {output_file}...")

    # Add extra dimensions only if they don't already exist
    extra_names = set(las.point_format.extra_dimension_names)
    if "room_id" not in extra_names:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="room_id",
            type="u2",
            description="Room ID"
        ))
    if "room_class" not in extra_names:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="room_class",
            type="u2",
            description="Room ceiling class"
        ))

    # Initialize to zero
    if not hasattr(las, "room_id"):
        raise RuntimeError("Failed to add 'room_id' extra dimension to LAS.")
    if not hasattr(las, "room_class"):
        raise RuntimeError("Failed to add 'room_class' extra dimension to LAS.")

    las.room_id = np.zeros(len(las.points), dtype=np.uint16)
    las.room_class = np.zeros(len(las.points), dtype=np.uint16)

    # Assign room_id to ceiling points only
    las.room_id[ceiling_mask] = final_point_labels

    # Unique class per room: 700 + room_id (0 -> background)
    ceiling_room_class = np.where(final_point_labels > 0,
                                  700 + final_point_labels,
                                  0).astype(np.uint16)
    las.room_class[ceiling_mask] = ceiling_room_class

    las.write(output_file)
    print("✓ Each room's ceiling now has a unique room_id and room_class (700 + room_id).")

def save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path):
    """Generates a polygon for each room and saves to JSON."""
    print(f"Generating floor polygons for JSON...")
    x_min, y_min, _, _ = bounds
    
    output_data = {"rooms": []}

    for rid in sorted(room_stats.keys()):
        rows, cols = np.where(grid == rid)
        if len(rows) < 3: continue

        real_x = x_min + (cols * grid_size) + (grid_size / 2)
        real_y = y_min + (rows * grid_size) + (grid_size / 2)
        points = list(zip(real_x, real_y))
        
        hull = MultiPoint(points).convex_hull
        if hull.geom_type == 'Polygon':
            coords = list(hull.exterior.coords)
            room_data = {
                "room_id": int(rid),
                "area_sqm": room_stats[rid]["area_m2"],
                "polygon_coordinates": [[round(x, 3), round(y, 3)] for x, y in coords]
            }
            output_data["rooms"].append(room_data)

    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"✓ Saved room polygons to: {json_path}")


# ============================================================================
# Room Area Calculation & Map Generation
# ============================================================================

def get_room_ids_from_las(las):
    """Get room IDs from LAS file."""
    try:
        if hasattr(las, "room_id"):
            return np.asarray(las.room_id, dtype=np.int32)
    except Exception:
        pass

    try:
        if "room_id" in las.point_format.extra_dimension_names:
            return np.asarray(las["room_id"], dtype=np.int32)
    except Exception:
        pass

    try:
        if hasattr(las, "room_class"):
            rc = np.asarray(las.room_class, dtype=np.int32)
            nonzero = rc[rc > 0]
            if nonzero.size > 0:
                median = np.median(nonzero)
                if median > 500:
                    base = 700
                elif median > 100:
                    base = 200
                else:
                    base = 0
                if base > 0:
                    return np.where(rc > 0, rc - base, 0).astype(np.int32)
    except Exception:
        pass

    return None


def compute_room_areas(points_xy, room_ids, grid_size):
    """Compute area for each room."""
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_min, y_min = np.min(points_xy, axis=0)
    x_max, y_max = np.max(points_xy, axis=0)

    width = int(np.ceil((x_max - x_min) / grid_size)) + 1
    height = int(np.ceil((y_max - y_min) / grid_size)) + 1

    grid = np.zeros((height, width), dtype=np.int32)

    xi = np.floor((x - x_min) / grid_size).astype(int)
    yi = np.floor((y - y_min) / grid_size).astype(int)
    xi = np.clip(xi, 0, width - 1)
    yi = np.clip(yi, 0, height - 1)

    cell_idx = yi * width + xi
    mask_nonzero = room_ids > 0
    
    if np.any(mask_nonzero):
        combos = np.vstack((cell_idx[mask_nonzero], room_ids[mask_nonzero])).T
        uniq_pairs, counts = np.unique(combos, axis=0, return_counts=True)
        
        from collections import defaultdict
        cell_best = {}
        for (cidx, rid), cnt in zip(uniq_pairs, counts):
            if cidx not in cell_best or cnt > cell_best[cidx][1]:
                cell_best[cidx] = (int(rid), int(cnt))
        
        for cidx, (rid, _) in cell_best.items():
            cy = cidx // width
            cx = cidx % width
            grid[cy, cx] = rid

    unique_ids, counts = np.unique(grid, return_counts=True)
    room_stats = {}
    for uid, cnt in zip(unique_ids, counts):
        if uid == 0:
            continue
        area = cnt * (grid_size ** 2)
        room_stats[int(uid)] = {"pixel_count": int(cnt), "area_m2": float(area)}

    return grid, room_stats, (x_min, y_min, x_max, y_max)


def save_csv(room_stats, csv_path):
    """Save room areas to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["room_id", "pixel_count", "area_m2"])
        for rid in sorted(room_stats.keys()):
            writer.writerow([rid, room_stats[rid]["pixel_count"], f"{room_stats[rid]['area_m2']:.4f}"])


def plot_map(grid, bounds, room_stats, output_image, cmap_name="tab20"):
    """Plot room map with areas."""
    x_min, y_min, x_max, y_max = bounds
    extent = (x_min, x_max, y_min, y_max)

    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap(cmap_name)
    
    unique_ids = np.unique(grid)
    unique_ids = unique_ids[unique_ids > 0]
    
    if unique_ids.size == 0:
        raise RuntimeError("No rooms found in grid to plot.")

    id_to_idx = {int(rid): (i % 20) for i, rid in enumerate(unique_ids)}
    color_grid = np.zeros_like(grid, dtype=np.int32)
    for rid, idx in id_to_idx.items():
        color_grid[grid == rid] = idx + 1

    plt.imshow(color_grid, origin='lower', extent=extent, interpolation='nearest')
    plt.title("Room segmentation map with surface areas")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')

    import matplotlib.patches as mpatches
    patches = []
    for rid, idx in id_to_idx.items():
        patches.append(mpatches.Patch(color=cmap(idx/20.0), label=f"Room {rid}"))
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    for rid in sorted(room_stats.keys()):
        positions = np.argwhere(grid == rid)
        if positions.size == 0:
            continue
        mean_row = positions[:, 0].mean()
        mean_col = positions[:, 1].mean()
        width = grid.shape[1]
        cell_x = x_min + (mean_col + 0.5) * ((x_max - x_min) / width)
        cell_y = y_min + (mean_row + 0.5) * ((y_max - y_min) / grid.shape[0])
        area = room_stats[rid]["area_m2"]
        plt.text(cell_x, cell_y, f"{rid}\n{area:.1f} m²", ha="center", va="center",
                 fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()


def step3_measure_rooms(input_file,
                        output_image,
                        csv_path,
                        json_path,
                        grid_size=0.05,
                        ceiling_only=True,
                        min_cells_per_room=5):

    print("\n" + "=" * 70)
    print("STEP 3: ROOM AREA CALCULATION & MAP GENERATION")
    print("=" * 70)

    print(f"Reading {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Total points in LAS: {len(points)}")

    room_ids_per_point = get_room_ids_from_las(las)
    if room_ids_per_point is None:
        print("Error: Could not find room_id in LAS file (room_id/room_class missing).")
        return

    mask = room_ids_per_point > 0

    if ceiling_only and hasattr(las, "classification"):
        cls = np.asarray(las.classification, dtype=np.uint8)
        mask &= (cls == 7)  # class 7 = ceiling (from step1_classify)
        print("Using ceiling-only points for area calculation (room_id > 0 AND class == 7).")
    else:
        if ceiling_only:
            print("Warning: ceiling_only=True but LAS has no 'classification' attribute; "
                  "falling back to room_id > 0 only.")
        else:
            print("Using all points with room_id > 0 for area calculation.")

    if not np.any(mask):
        print("Error: No points satisfy the selection (room_id > 0 and optional ceiling filter).")
        return

    points_xy = points[mask, :2]
    room_ids = room_ids_per_point[mask]

    unique_room_ids = np.unique(room_ids)
    print(f"Found {len(unique_room_ids)} unique room IDs among {points_xy.shape[0]} selected points.")
    print(f"Room IDs: {unique_room_ids.tolist()}")

    print(f"Computing room areas on grid (grid_size = {grid_size:.3f} m)...")
    grid, room_stats, bounds = compute_room_areas(points_xy, room_ids, grid_size)

    if not room_stats:
        print("Error: compute_room_areas returned no rooms (room_stats is empty).")
        return

    print("Raw room_stats (before QC):")
    for rid in sorted(room_stats.keys()):
        s = room_stats[rid]
        print(f"  Room {rid}: {s['area_m2']:.3f} m²  ({s['pixel_count']} cells)")

    if min_cells_per_room is not None and min_cells_per_room > 0:
        to_drop = [rid for rid, s in room_stats.items()
                   if s["pixel_count"] < min_cells_per_room]

        if to_drop:
            print(f"\nQC: Removing rooms with < {min_cells_per_room} cells as noise: {to_drop}")
            for rid in to_drop:
                # Zero out their cells in the grid
                grid[grid == rid] = 0
                del room_stats[rid]

            if not room_stats:
                print("Error: All rooms removed by QC (min_cells_per_room too high?).")
                return

    print("\nFinal room_stats (after QC):")
    for rid in sorted(room_stats.keys()):
        s = room_stats[rid]
        print(f"  Room {rid}: {s['area_m2']:.3f} m²  ({s['pixel_count']} cells)")

    save_csv(room_stats, csv_path)
    print(f"\n✓ Wrote CSV to: {csv_path}")

    save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path)

    print(f"Plotting map to {output_image}...")
    plot_map(grid, bounds, room_stats, output_image)
    print(f"✓ Saved map image to {output_image}")

    print("\nRoom areas (m²):")
    for rid in sorted(room_stats.keys()):
        s = room_stats[rid]
        print(f"  Room {rid}: {s['area_m2']:.3f} m²  ({s['pixel_count']} cells)")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete room analysis pipeline: classify, segment, and measure rooms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python complete_room_analysis.py input.laz output_folder
  python complete_room_analysis.py input.laz output_folder --voxel 0.03 --min-room-points 3000

Output files:
  output_folder/classified.las      - Classified point cloud
  output_folder/segmented.las       - Segmented rooms
  output_folder/room_map.png        - 2D visualization with areas
  output_folder/room_areas.csv      - Table with room areas
  output_folder/room_polygons.json  - Polygon coordinates (JSON)
        """
    )

    parser.add_argument("input", help="Input .las/.laz file")
    parser.add_argument("output_folder", help="Output folder for all results")

    # ------------------------------------------------------------------
    # Classification parameters (Step 1)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--voxel", type=float, default=0.02,
        help="Voxel size for downsampling in meters (default: 0.02)"
    )
    parser.add_argument(
        "--normal-threshold", type=float, default=0.7,
        help="Normal similarity threshold (dot product 0-1, default: 0.7)"
    )
    parser.add_argument(
        "--distance-threshold", type=float, default=0.15,
        help="Neighbor distance threshold in meters (default: 0.15)"
    )
    parser.add_argument(
        "--min-region-size", type=int, default=500,
        help="Minimum points per region (default: 500)"
    )
    parser.add_argument(
        "--floor-pct", type=float, default=1.0,
        help="Lower percentile of Z for floor estimate (default: 1.0)"
    )
    parser.add_argument(
        "--ceil-pct", type=float, default=99.0,
        help="Upper percentile of Z for ceiling estimate (default: 99.0)"
    )
    parser.add_argument(
        "--max-z-diff", type=float, default=0.0,
        help="Max |ΔZ| within a region [m]; <= 0 disables Z constraint (default: 0.0)"
    )
    parser.add_argument(
        "--skip-classify", action="store_true",
        help="Skip Step 1 if classified.las already exists"
    )

    # ------------------------------------------------------------------
    # Segmentation parameters (Step 2)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--grid-size", type=float, default=0.01,
        help="Grid cell size for segmentation (default: 0.01)"
    )
    parser.add_argument(
        "--dilation-iterations", type=int, default=3,
        help="Wall dilation iterations (default: 3)"
    )
    parser.add_argument(
        "--min-room-points", type=int, default=5000,
        help="Minimum ceiling points per room (default: 5000)"
    )

    # ------------------------------------------------------------------
    # Measurement parameters (Step 3)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--measure-grid-size", type=float, default=0.05,
        help="Grid size for area calculation (default: 0.05)"
    )
    parser.add_argument(
        "--min-cells-per-room", type=int, default=5,
        help="Minimum grid cells per room to keep in area calc (default: 5)"
    )
    parser.add_argument(
        "--ceiling-only", dest="ceiling_only", action="store_true",
        help="Use only ceiling points (class 7) with room_id > 0 for area calc (default)."
    )
    parser.add_argument(
        "--no-ceiling-only", dest="ceiling_only", action="store_false",
        help="Use all points with room_id > 0 for area calc."
    )
    parser.set_defaults(ceiling_only=True)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input bestand niet gevonden: {args.input}")
        sys.exit(1)

    # Create output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    classified_file = output_folder / "classified.las"
    segmented_file = output_folder / "segmented.las"
    map_file = output_folder / "room_map.png"
    csv_file = output_folder / "room_areas.csv"
    json_file = output_folder / "room_polygons.json"

    print("\n" + "=" * 70)
    print("COMPLETE ROOM ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"Input file:      {args.input}")
    print(f"Output folder:   {output_folder}")
    print("\nOutput files:")
    print(f"  - {classified_file.name}")
    print(f"  - {segmented_file.name}")
    print(f"  - {map_file.name}")
    print(f"  - {csv_file.name}")
    print(f"  - {json_file.name}")

    try:
        # ------------------------------------------------------------------
        # STEP 1: Classification (Region Growing)
        # ------------------------------------------------------------------
        run_classify = True
        if args.skip_classify and classified_file.exists():
            print("Skipping Step 1 (using existing classified.las)...")
            run_classify = False

        if run_classify:
            max_z_diff = args.max_z_diff if args.max_z_diff > 0 else None
            step1_classify(
                args.input,
                str(classified_file),
                voxel_size=args.voxel,
                normal_threshold=args.normal_threshold,
                distance_threshold=args.distance_threshold,
                min_region_size=args.min_region_size,
                floor_pct=args.floor_pct,
                ceil_pct=args.ceil_pct,
                max_z_diff=max_z_diff,
            )

        # ------------------------------------------------------------------
        # STEP 2: Room Segmentation (Ceiling-driven)
        # ------------------------------------------------------------------
        step2_segment_rooms(
            str(classified_file),
            str(segmented_file),
            grid_size=args.grid_size,
            dilation_iterations=args.dilation_iterations,
            min_room_points=args.min_room_points,
        )

        # If segmentation failed or produced no file, stop here
        if not os.path.exists(segmented_file):
            print(f"\nERROR: Segmentation did not produce '{segmented_file}'. "
                  f"Check STEP 2 logs above (likely no ceiling points or no valid rooms).")
            sys.exit(1)

        # ------------------------------------------------------------------
        # STEP 3: Surface Area Calculation & Map Generation
        # ------------------------------------------------------------------
        step3_measure_rooms(
            str(segmented_file),
            str(map_file),
            str(csv_file),
            str(json_file),
            grid_size=args.measure_grid_size,
            ceiling_only=args.ceiling_only,
            min_cells_per_room=args.min_cells_per_room,
        )

        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\nAll results saved in: {output_folder}")
        print("\nFiles:")
        print(f"  1. {classified_file.name}  - Classified point cloud")
        print(f"  2. {segmented_file.name}   - Segmented rooms")
        print(f"  3. {map_file.name}         - 2D visualization")
        print(f"  4. {csv_file.name}         - Area table")
        print(f"  5. {json_file.name}        - Room polygons (JSON)")

    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
