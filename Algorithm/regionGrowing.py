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
from scipy.ndimage import binary_closing, label
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max


# ============================================================================
# Region Growing Classification
# ============================================================================

def region_growing(points, normals, kd_tree, seed_idx, visited, 
                   normal_threshold=0.9, distance_threshold=0.1, min_region_size=1200):
    """Perform region growing from a seed point."""
    region = []
    queue = deque([seed_idx])
    visited[seed_idx] = True
    seed_normal = normals[seed_idx]
    
    while queue:
        current_idx = queue.popleft()
        region.append(current_idx)
        
        neighbor_indices = kd_tree.query_radius(
            points[current_idx].reshape(1, -1), 
            r=distance_threshold
        )[0]
        
        for neighbor_idx in neighbor_indices:
            if visited[neighbor_idx]:
                continue
            
            dot_product = np.dot(normals[neighbor_idx], seed_normal)
            
            if dot_product > normal_threshold:
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)
    
    return region if len(region) >= min_region_size else []


def classify_region(points_in_region, normals_in_region, z_stats):
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


def step1_classify(input_file, output_file, voxel_size=0.02,
                   normal_threshold=0.9, distance_threshold=0.1,
                   min_region_size=1500):
    print("\n" + "="*70)
    print("CHECKPOINT 1: CLASSIFICATION (Region Growing)")
    print("="*70)
    
    print("Reading point cloud...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Original points: {len(points)}")

    print("Downsampling point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_down = np.asarray(pcd.points)
    print(f"Points after downsampling: {len(points_down)}")

    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_consistent_tangent_plane(k=20)
    normals = np.asarray(pcd.normals)

    print("Building KD-Tree...")
    kd_tree = KDTree(points_down)

    print("Performing region growing classification...")
    classification = np.ones(len(points_down), dtype=np.uint8)
    visited = np.zeros(len(points_down), dtype=bool)
    
    z_min = np.min(points_down[:, 2])
    z_max = np.max(points_down[:, 2])
    z_range = z_max - z_min
    
    z_stats = {
        'min': z_min,
        'max': z_max,
        'range': z_range,
        'threshold_floor': z_min + 0.35 * z_range,
        'threshold_ceiling': z_max - 0.35 * z_range
    }
    
    print(f"Z-range: {z_min:.2f}m to {z_max:.2f}m")
    print(f"Floor threshold: < {z_stats['threshold_floor']:.2f}m")
    print(f"Ceiling threshold: > {z_stats['threshold_ceiling']:.2f}m")
    
    regions_found = 0
    points_classified = 0
    
    normal_z_abs = np.abs(normals[:, 2])
    seed_order = np.argsort(-np.maximum(normal_z_abs, 1 - normal_z_abs))
    
    for seed_idx in seed_order:
        if visited[seed_idx]:
            continue
        
        region_indices = region_growing(
            points_down, normals, kd_tree, seed_idx, visited,
            normal_threshold=normal_threshold,
            distance_threshold=distance_threshold,
            min_region_size=min_region_size
        )
        
        if len(region_indices) > 0:
            region_points = points_down[region_indices]
            region_normals = normals[region_indices]
            region_class = classify_region(region_points, region_normals, z_stats)
            
            classification[region_indices] = region_class
            
            regions_found += 1
            points_classified += len(region_indices)
            
            if regions_found % 10 == 0:
                print(f"  Regions found: {regions_found}, Points classified: {points_classified}/{len(points_down)}")

    print(f"\nTotal regions found: {regions_found}")
    print(f"Points classified: {points_classified}/{len(points_down)}")
    
    unique, counts = np.unique(classification, return_counts=True)
    class_names = {1: "Unclassified", 2: "Floor", 6: "Wall", 7: "Ceiling"}
    print("\nClassification statistics:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names.get(cls, f'Class {cls}')}: {count} points ({100*count/len(classification):.1f}%)")

    print(f"Saving classified point cloud to {output_file}...")
    header = laspy.LasHeader(point_format=3, version="1.2")
    out = laspy.LasData(header)
    out.x = points_down[:, 0]
    out.y = points_down[:, 1]
    out.z = points_down[:, 2]
    out.classification = classification
    out.write(output_file)
    print(f"✓ Saved {len(points_down)} classified points")


# ============================================================================
# CHECKPOINT 2: ROOM SEGMENTATION (Watershed - The Fix)
# ============================================================================
# Make sure to import these at the top of your file:
# from scipy.ndimage import distance_transform_edt
# from skimage.segmentation import watershed
# from skimage.feature import peak_local_max

def step2_segment_rooms(input_file, output_file, grid_size=0.05, 
                        dilation_iterations=3, min_room_points=500): # <--- Lowered this to find the missing room
    print("\n" + "="*70)
    print("CHECKPOINT 2: ROOM SEGMENTATION (Watershed Method)")
    print("="*70)
    
    print(f"Reading classified point cloud from {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    classification = np.array(las.classification)
    
    # 1. SLICE WALLS (Robust Slice)
    z_min = np.min(points[:, 2])
    # Slice between 0.5m and 2.0m to catch the main wall structure
    wall_mask = (classification == 6) & (points[:, 2] > z_min + 0.5) & (points[:, 2] < z_min + 2.0)
    ceiling_mask = classification == 7
    
    wall_points = points[wall_mask]
    ceiling_points = points[ceiling_mask]
    
    print(f"Using {len(wall_points)} wall points for segmentation.")

    # 2. CREATE 2D GRID
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)
    
    # Use the grid_size passed in arguments (Recommend 0.05)
    grid_width = int(np.ceil((x_max - x_min) / grid_size))
    grid_height = int(np.ceil((y_max - y_min) / grid_size))
    
    occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
    
    wall_x = ((wall_points[:, 0] - x_min) / grid_size).astype(int)
    wall_y = ((wall_points[:, 1] - y_min) / grid_size).astype(int)
    
    # Clip to bounds
    wall_x = np.clip(wall_x, 0, grid_width - 1)
    wall_y = np.clip(wall_y, 0, grid_height - 1)
    
    occupancy_grid[wall_y, wall_x] = 1 # Walls are 1

    # 3. WATERSHED LOGIC (Fixes the Merged Room Problem)
    print("Running Watershed Segmentation...")
    
    # A. Distance Map: Calculate distance from every empty pixel to the nearest wall
    dist_transform = distance_transform_edt(1 - occupancy_grid)
    
    # B. Find Peaks (Centers of Rooms)
    # REMOVED 'indices=False' because it is deprecated. 
    # This now returns a list of coordinates [[r, c], ...].
    local_maxi = peak_local_max(dist_transform, min_distance=8, labels=(1-occupancy_grid))
    
    # NEW STEP: Convert coordinates back to a Boolean Mask
    local_maxi_mask = np.zeros_like(dist_transform, dtype=bool)
    if len(local_maxi) > 0:
        local_maxi_mask[tuple(local_maxi.T)] = True

    # C. Create Markers
    from scipy.ndimage import label
    # Now we pass the boolean mask to label(), not the raw coordinates
    markers = label(local_maxi_mask)[0]
    
    # D. Run Watershed
    room_labels_grid = watershed(-dist_transform, markers, mask=(1-occupancy_grid))
    
    num_labels = room_labels_grid.max()
    print(f"Watershed found {num_labels} potential regions.")

    # 4. MAP TO 3D & FILTER (Fixes the Missing Room Problem)
    ceil_x = ((ceiling_points[:, 0] - x_min) / grid_size).astype(int)
    ceil_y = ((ceiling_points[:, 1] - y_min) / grid_size).astype(int)
    ceil_x = np.clip(ceil_x, 0, grid_width - 1)
    ceil_y = np.clip(ceil_y, 0, grid_height - 1)
    
    point_room_labels = room_labels_grid[ceil_y, ceil_x]

    unique_labels, counts = np.unique(point_room_labels, return_counts=True)
    valid_rooms = 0
    final_room_id_map = {}
    
    print("\nFiltering rooms by size...")
    for room_id, count in zip(unique_labels, counts):
        if room_id == 0: continue
        
        # We lowered the threshold to catch the small missing room
        if count >= min_room_points:
            valid_rooms += 1
            final_room_id_map[room_id] = valid_rooms
        else:
            # Print what we are deleting to debug
            print(f"  Deleted tiny region {room_id} ({count} points) - likely noise.")
            
    print(f"Found {valid_rooms} VALID rooms.")

    # Apply mapping
    final_point_labels = np.zeros_like(point_room_labels)
    for original_id, final_id in final_room_id_map.items():
        final_point_labels[point_room_labels == original_id] = final_id

    # 5. SAVE
    print(f"Saving segmented point cloud to {output_file}...")
    try:
        las.add_extra_dim(laspy.ExtraBytesParams(name="room_id", type="u2"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="room_class", type="u2"))
    except: pass

    las.room_id = np.zeros(len(las.points), dtype=np.uint16)
    las.room_class = np.zeros(len(las.points), dtype=np.uint16)

    las.room_id[ceiling_mask] = final_point_labels
    
    # Color logic
    ceiling_room_class = np.where(final_point_labels > 0, 700 + final_point_labels, 0)
    las.room_class[ceiling_mask] = ceiling_room_class

    las.write(output_file)
    print("✓ Segmentation done.")
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
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["room_id", "pixel_count", "area_m2"])
        for rid in sorted(room_stats.keys()):
            writer.writerow([rid, room_stats[rid]["pixel_count"], f"{room_stats[rid]['area_m2']:.4f}"])


def plot_map(grid, bounds, room_stats, output_image, cmap_name="tab20"):
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

def save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path):
    """Generates a polygon for each room and saves to JSON."""
    print(f"Generating floor polygons for JSON...")
    x_min, y_min, _, _ = bounds
    
    output_data = {"rooms": []}

    for rid in sorted(room_stats.keys()):
        # Find all grid cells belonging to this room
        rows, cols = np.where(grid == rid)
        if len(rows) < 3: continue # Need at least 3 points for a polygon

        # Convert grid coordinates back to real-world coordinates
        real_x = x_min + (cols * grid_size) + (grid_size / 2)
        real_y = y_min + (rows * grid_size) + (grid_size / 2)
        points = list(zip(real_x, real_y))
        
        # Calculate Convex Hull (Rubber band around the room)
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

def step3_measure_rooms(input_file, output_image, csv_path, json_path, grid_size=0.05):
    print("\n" + "="*70)
    print("CHECKPOINT 3: ROOM AREA CALCULATION & MAP GENERATION")
    print("="*70)
    
    print(f"Reading {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Total points: {len(points)}")

    room_ids_per_point = get_room_ids_from_las(las)
    if room_ids_per_point is None:
        print("Error: Could not find room_id in LAS file.")
        return

    mask = room_ids_per_point > 0
    if not np.any(mask):
        print("Error: No points have a room_id > 0.")
        return

    points_xy = points[mask, :2]
    room_ids = room_ids_per_point[mask]

    print(f"Found {len(np.unique(room_ids))} unique room IDs among {points_xy.shape[0]} points.")

    grid, room_stats, bounds = compute_room_areas(points_xy, room_ids, grid_size)

    save_csv(room_stats, csv_path)
    print(f"✓ Wrote CSV to: {csv_path}")

    save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path)

    print(f"Plotting map to {output_image}...")
    plot_map(grid, bounds, room_stats, output_image)
    print(f"✓ Saved map image to {output_image}")

    print("\nRoom areas (m²):")
    for rid in sorted(room_stats.keys()):
        print(f"  Room {rid}: {room_stats[rid]['area_m2']:.3f} m²  ({room_stats[rid]['pixel_count']} cells)")


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
    
    # Classification parameters
    parser.add_argument("--voxel", type=float, default=0.02,
                       help="Voxel size for downsampling in meters (default: 0.02)")
    parser.add_argument("--normal-threshold", type=float, default=0.9,
                       help="Normal similarity threshold (0-1, default: 0.9)")
    parser.add_argument("--distance-threshold", type=float, default=0.1,
                       help="Neighbor distance threshold in meters (default: 0.1)")
    parser.add_argument("--min-region-size", type=int, default=1500,
                       help="Minimum points per region (default: 1500)")
    
    # Segmentation parameters
    parser.add_argument("--grid-size", type=float, default=0.01,
                       help="Grid cell size for segmentation (default: 0.01)")
    parser.add_argument("--dilation-iterations", type=int, default=3,
                       help="Wall dilation iterations (default: 3)")
    parser.add_argument("--min-room-points", type=int, default=5000,
                       help="Minimum points per room (default: 5000)")
    
    # Measurement parameters
    parser.add_argument("--measure-grid-size", type=float, default=0.05,
                       help="Grid size for area calculation (default: 0.05)")
    
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
    
    print("\n" + "="*70)
    print("COMPLETE ROOM ANALYSIS PIPELINE")
    print("="*70)
    print(f"Input file:    {args.input}")
    print(f"Output folder:    {output_folder}")
    print(f"\nOutput files:")
    print(f"  - {classified_file.name}")
    print(f"  - {segmented_file.name}")
    print(f"  - {map_file.name}")
    print(f"  - {csv_file.name}")
    print(f"  - {json_file.name}")
    
    try:
        # Classification
        # print("Skipping Step 1...")  <--- You can comment this out
        # step1_classify(              <--- COMMENT THIS BLOCK OUT
        #     args.input,
        #     str(classified_file),
        #     voxel_size=args.voxel,
        #     normal_threshold=args.normal_threshold,
        #     distance_threshold=args.distance_threshold,
        #     min_region_size=args.min_region_size
        # )

        # Segmentation
        # step2_segment_rooms(         <--- COMMENT THIS BLOCK OUT
        #     str(classified_file),
        #     str(segmented_file),
        #     grid_size=args.grid_size,
        #     dilation_iterations=args.dilation_iterations,
        #     min_room_points=args.min_room_points
        # )
        
        # Surface Area Calculation & Map Generation
        step3_measure_rooms(           # <--- KEEP THIS ONE!
            str(segmented_file),
            str(map_file),
            str(csv_file),
            str(json_file),
            grid_size=args.measure_grid_size
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nAll results saved in: {output_folder}")
        print(f"\nFiles:")
        print(f"  1. {classified_file.name} - Classified point cloud")
        print(f"  2. {segmented_file.name} - Segmented rooms")
        print(f"  3. {map_file.name} - 2D visualization")
        print(f"  4. {csv_file.name} - Area table")
        print(f"  5. {json_file.name} - Room Polygons (JSON)")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()