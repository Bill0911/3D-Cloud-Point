import numpy as np
from scipy.ndimage import label, binary_dilation, binary_fill_holes
from ..io.las_io import load_las_with_classification, save_las_with_room_ids


def run_segmentation(input_file, output_file, cfg):
    print("STEP 2: SEGMENTATION (robust floor+ceiling mode)")

    las, points, classification = load_las_with_classification(input_file)

    # Filter points to only include classified ones (safety check)
    classified_mask = classification > 1  # Assuming 1 is unclassified
    points_classified = points[classified_mask]
    classification_classified = classification[classified_mask]

    if len(points_classified) == 0:
        print("ERROR: No classified points found in the input file.")
        return None

    # Masks based on classified points
    wall_mask = classification_classified == 6
    ceil_mask = classification_classified == 7
    floor_mask = classification_classified == 2

    walls = points_classified[wall_mask]
    ceils = points_classified[ceil_mask]
    floors = points_classified[floor_mask]

    if len(ceils) == 0 and len(floors) == 0:
        print("ERROR: no ceiling or floor points → cannot segment")
        return None

    # --------------------------------------------
    # 2D bounds
    # --------------------------------------------
    # Bounds must be based on ALL points, not just classified ones, for correct origin
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    gs = cfg["grid_size"]
    gw = int(np.ceil((x_max - x_min) / gs)) + 2
    gh = int(np.ceil((y_max - y_min) / gs)) + 2

    # --------------------------------------------
    # WALL GRID (REDUCED DILATION)
    # --------------------------------------------
    wall_grid = np.zeros((gh, gw), dtype=np.uint8)

    if len(walls) > 0:
        # Calculate grid coordinates for classified walls
        wx = ((walls[:, 0] - x_min) / gs).astype(int)
        wy = ((walls[:, 1] - y_min) / gs).astype(int)
        wx = np.clip(wx, 0, gw - 1)
        wy = np.clip(wy, 0, gh - 1)
        wall_grid[wy, wx] = 1

        # Dilation: Reduced factor for more accurate boundaries
        # We use a factor of 1, not 3, as the initial grid creation is already sparse.
        iterations = max(cfg["dilation_iterations"], 3)  # Use dilation_iterations directly, min 3
        print(f"Wall dilation iterations = {iterations}")
        wall_grid = binary_dilation(wall_grid, iterations=iterations).astype(np.uint8)

    # --------------------------------------------
    # SURFACE GRID = CEILING + FLOOR (based on classified points)
    # --------------------------------------------
    surf_grid = np.zeros((gh, gw), dtype=np.uint8)

    # Combined surface points for grid mapping
    if len(ceils) > 0:
        cx = ((ceils[:, 0] - x_min) / gs).astype(int)
        cy = ((ceils[:, 1] - y_min) / gs).astype(int)
        surf_grid[np.clip(cy, 0, gh - 1), np.clip(cx, 0, gw - 1)] = 1

    if len(floors) > 0:
        fx = ((floors[:, 0] - x_min) / gs).astype(int)
        fy = ((floors[:, 1] - y_min) / gs).astype(int)
        # Only set if not already set by ceiling (should be rare)
        surf_grid[np.clip(fy, 0, gh - 1), np.clip(fx, 0, gw - 1)] = 1

        # Fill missing gaps between scan trajectories
    surf_grid = binary_fill_holes(surf_grid).astype(np.uint8)

    # --------------------------------------------
    # SEGMENTATION GRID = surface but NOT walls
    # --------------------------------------------
    seg_grid = (surf_grid == 1) & (wall_grid == 0)
    seg_grid = seg_grid.astype(np.uint8)

    # Connected components
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, raw_count = label(seg_grid, structure=structure)
    print(f"Raw connected components: {raw_count}")

    # --------------------------------------------
    # Assign each ceiling/floor point a label (UPDATED LOGIC)
    # --------------------------------------------
    # Use only the ceiling points if possible, otherwise use floor points
    if len(ceils) > 0:
        use_pts = ceils
        use_mask_classified = ceil_mask
        print("Using ceiling points for room ID assignment.")
    else:
        use_pts = floors
        use_mask_classified = floor_mask
        print("WARNING: no ceiling points available → using floor for room ID assignment.")

    # Map points to the 2D component labels
    ux = ((use_pts[:, 0] - x_min) / gs).astype(int)
    uy = ((use_pts[:, 1] - y_min) / gs).astype(int)
    ux = np.clip(ux, 0, gw - 1)
    uy = np.clip(uy, 0, gh - 1)

    point_labels = labels[uy, ux]

    # Find the unique component IDs and their size (point count)
    uniq, counts = np.unique(point_labels, return_counts=True)

    # --------------------------------------------
    # Filter out tiny components
    # --------------------------------------------
    out_map = {}
    next_id = 1

    for raw_id, cnt in zip(uniq, counts):
        if raw_id == 0:
            continue
        if cnt >= cfg["min_room_points"]:
            out_map[raw_id] = next_id
            next_id += 1

    if not out_map:
        print("ERROR: No valid rooms detected after filtering.")
        print("Try lowering min_room_points in config.")
        return None

    print(f"Valid rooms: {len(out_map)}")

    # --------------------------------------------
    # Build final room ID map for the entire point cloud
    # --------------------------------------------

    # 1. Map the component labels (raw_id) to the new room IDs (new_id)
    final_ids_classified_pts = np.zeros_like(point_labels, dtype=np.uint16)
    for raw_id, new_id in out_map.items():
        final_ids_classified_pts[point_labels == raw_id] = new_id

    # 2. Assign the room IDs back to the original full point cloud array
    full_room_ids = np.zeros(len(points), dtype=np.uint16)

    # Get the indices of the subset we used (ceiling/floor classified points)
    # This requires re-calculating the mask relative to the *full* point cloud (points)

    # We need the indices of the original points array that correspond to `use_pts`
    # This is tricky because `use_pts` is a subset of `points_classified` which is a subset of `points`.

    # A cleaner approach: use the original classification array to index into the room IDs.

    # Mask of all ceiling/floor points in the ORIGINAL point cloud
    surf_points_mask = (classification == 7) | (classification == 2)

    # Temporarily store the indices of the surface points in the original array
    surf_indices = np.where(surf_points_mask)[0]

    # Create the point cloud subset used for segmentation
    temp_pts = points[surf_points_mask]

    # Map these points to the grid labels again (necessary step to handle the wall grid blocking)
    tx = ((temp_pts[:, 0] - x_min) / gs).astype(int)
    ty = ((temp_pts[:, 1] - y_min) / gs).astype(int)
    tx = np.clip(tx, 0, gw - 1)
    ty = np.clip(ty, 0, gh - 1)

    # Get the raw component label for EVERY surface point
    raw_labels_for_all_surf_pts = labels[ty, tx]

    # Now, map those raw labels to the final, filtered room IDs
    final_ids_for_all_surf_pts = np.zeros_like(raw_labels_for_all_surf_pts, dtype=np.uint16)
    for raw_id, new_id in out_map.items():
        # Only assign new ID if the raw label corresponds to a large enough component
        final_ids_for_all_surf_pts[raw_labels_for_all_surf_pts == raw_id] = new_id

    # Assign final room IDs back to the original full point cloud
    full_room_ids[surf_indices] = final_ids_for_all_surf_pts

    # Save
    save_las_with_room_ids(las, full_room_ids, output_file)
    print(f"Saved → {output_file}")

    return {
        "rooms": len(out_map),
        "grid_size": gs,
        "bounds": (x_min, y_min, x_max, y_max),
    }