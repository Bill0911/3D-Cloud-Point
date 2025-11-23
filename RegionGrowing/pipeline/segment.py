import numpy as np
from scipy.ndimage import label, binary_dilation
from ..io.las_io import load_las_with_classification, save_las_with_room_ids

def run_segmentation(input_file, output_file, cfg):
    print("STEP 2: Segmentation")

    # Load LAS and extract points + classification
    las, points, classification = load_las_with_classification(input_file)
    print(f"Loaded {len(points)} points")

    # Masks for walls and ceilings
    wall_mask = classification == 6
    ceil_mask = classification == 7

    walls = points[wall_mask]
    ceils = points[ceil_mask]

    if len(ceils) == 0:
        print("No ceiling detected")
        return None

    # Bounds for converting world coordinates -> grid indices
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    gs = cfg["grid_size"]
    gw = int(np.ceil((x_max - x_min) / gs))
    gh = int(np.ceil((y_max - y_min) / gs))

    if gw <= 0 or gh <= 0:
        raise ValueError("Grid size produced zero-size matrix")

    # Wall occupancy grid
    wall_grid = np.zeros((gh, gw), dtype=np.uint8)
    if len(walls) > 0:
        wx = ((walls[:, 0] - x_min) / gs).astype(int)
        wy = ((walls[:, 1] - y_min) / gs).astype(int)
        wx = np.clip(wx, 0, gw - 1)
        wy = np.clip(wy, 0, gh - 1)
        wall_grid[wy, wx] = 1
        if cfg["dilation_iterations"] > 0:
            wall_grid = binary_dilation(
                wall_grid,
                iterations=cfg["dilation_iterations"]
            ).astype(np.uint8)

    # Ceiling occupancy grid
    ceil_grid = np.zeros((gh, gw), dtype=np.uint8)
    cx = ((ceils[:, 0] - x_min) / gs).astype(int)
    cy = ((ceils[:, 1] - y_min) / gs).astype(int)
    cx = np.clip(cx, 0, gw - 1)
    cy = np.clip(cy, 0, gh - 1)
    ceil_grid[cy, cx] = 1

    # Valid room areas: ceiling present and no wall
    seg = ((ceil_grid == 1) & (wall_grid == 0)).astype(np.uint8)

    # 8-connected component labeling
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, raw_count = label(seg, structure=structure)
    print(f"Raw components: {raw_count}")

    # Assign each ceiling point a raw room label from the grid
    sample_labels = labels[cy, cx]

    uniq, cnts = np.unique(sample_labels, return_counts=True)

    # Filter tiny ceiling components
    out_map = {}
    next_id = 1
    for raw_id, count in zip(uniq, cnts):
        if raw_id == 0:
            continue
        if count >= cfg["min_room_points"]:
            out_map[raw_id] = next_id
            next_id += 1

    if len(out_map) == 0:
        print("No valid rooms after filtering")
        return None

    # Construct final room_id per ceiling point
    final = np.zeros_like(sample_labels, dtype=np.uint16)
    for raw_id, new_id in out_map.items():
        final[sample_labels == raw_id] = new_id

    print(f"Valid rooms: {len(out_map)}")

    # Save to LAS with room_id + room_class
    save_las_with_room_ids(
        las,
        ceil_mask,
        final,
        output_file
    )

    print(f"Saved segmented -> {output_file}")

    return {
        "room_ids": final,
        "rooms": len(out_map),
        "bounds": (x_min, y_min, x_max, y_max),
        "grid_size": gs,
    }
