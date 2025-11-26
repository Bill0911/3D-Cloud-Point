import numpy as np
from scipy.ndimage import label, binary_dilation, binary_fill_holes
from ..io.las_io import load_las_with_classification, save_las_with_room_ids


def run_segmentation(input_file, output_file, cfg):
    print("STEP 2: Segmentation")

    las, points, classification = load_las_with_classification(input_file)
    print(f"Loaded {len(points)} points")

    wall_mask = classification == 6
    ceil_mask = classification == 7

    walls = points[wall_mask]
    ceils = points[ceil_mask]

    if len(ceils) == 0:
        print("No ceiling detected → stopping.")
        return None

    # Bounds → 2D raster coordinate system
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    gs = cfg["grid_size"]
    if gs < 0.02:
        print(f"grid_size={gs} is too fine. Use ≥ 0.03")
        return None

    gw = int(np.ceil((x_max - x_min) / gs))
    gh = int(np.ceil((y_max - y_min) / gs))

    if gw <= 5 or gh <= 5:
        raise ValueError("Grid collapse — check grid_size.")

    # -------- WALL GRID --------
    wall_grid = np.zeros((gh, gw), dtype=np.uint8)
    if len(walls) > 0:
        wx = ((walls[:, 0] - x_min) / gs).astype(int)
        wy = ((walls[:, 1] - y_min) / gs).astype(int)
        wx = np.clip(wx, 0, gw - 1)
        wy = np.clip(wy, 0, gh - 1)
        wall_grid[wy, wx] = 1

        # Strengthen barriers
        wall_grid = binary_dilation(
            wall_grid,
            iterations=cfg["dilation_iterations"]
        ).astype(np.uint8)

    # -------- CEILING GRID --------
    ceil_grid = np.zeros((gh, gw), dtype=np.uint8)
    cx = ((ceils[:, 0] - x_min) / gs).astype(int)
    cy = ((ceils[:, 1] - y_min) / gs).astype(int)
    cx = np.clip(cx, 0, gw - 1)
    cy = np.clip(cy, 0, gh - 1)
    ceil_grid[cy, cx] = 1

    # Smooth out gaps (extremely important)
    ceil_grid = binary_dilation(
        ceil_grid,
        iterations=cfg.get("ceil_dilation", 2)
    ).astype(np.uint8)

    # fill small ceiling holes
    ceil_grid = binary_fill_holes(ceil_grid).astype(np.uint8)

    # -------- SEGMENTATION GRID --------
    seg_grid = ((ceil_grid == 1) & (wall_grid == 0)).astype(np.uint8)

    # label connected ceiling regions
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, raw_count = label(seg_grid, structure=structure)
    print(f"Raw components: {raw_count}")

    # assign each ceiling point a raw label
    sample_labels = labels[cy, cx]

    uniq, counts = np.unique(sample_labels, return_counts=True)

    # filter small ceiling blobs
    out_map = {}
    next_id = 1
    for rid, cnt in zip(uniq, counts):
        if rid == 0:
            continue
        if cnt >= cfg["min_room_points"]:
            out_map[rid] = next_id
            next_id += 1

    if len(out_map) == 0:
        print("No valid rooms after filtering")
        return None

    print(f"Valid rooms: {len(out_map)}")

    # build final room_id array for ceiling points
    final_ceiling_ids = np.zeros_like(sample_labels, dtype=np.uint16)
    for raw_id, new_id in out_map.items():
        final_ceiling_ids[sample_labels == raw_id] = new_id

    # synchronize with full cloud
    full_room_ids = np.zeros(len(points), dtype=np.uint16)
    full_room_ids[ceil_mask] = final_ceiling_ids

    save_las_with_room_ids(
        las,
        full_room_ids,
        output_path=output_file
    )

    print(f"Saved segmented → {output_file}")

    return {
        "room_ids": full_room_ids,
        "rooms": len(out_map),
        "bounds": (x_min, y_min, x_max, y_max),
        "grid_size": gs,
    }
