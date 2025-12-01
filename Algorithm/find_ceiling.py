import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, label
from skimage.segmentation import watershed
from skimage.morphology import disk, remove_small_objects, convex_hull_image
from scipy.ndimage import distance_transform_edt
import argparse
import os
import json

CONFIG = {
    "IO": {
        "default_input_path": r"C:\Users\dober\Downloads\emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332\appartement_SOR_NoiseFiltered_5mm.las",
        "output_image": "segmentation_hull.png",
        "output_json": "room_data.json"
    },
    "SLICING": {
        "height_min_offset": 2.1,  
        "height_max_offset": 2.54  
    },
    "PROCESSING": {
        "voxel_resolution": 0.05, 
        "erosion_radius": 0.4,     
        "grid_padding": 20,       
        "min_wall_size_px": 200    
    },
    "VISUALIZATION": {
        "figure_size": (15, 15),
        "font_size": 9,
        "cmap_walls": "gray",
        "cmap_rooms": "tab20",
        "opacity": 0.9
    }
}

def hull_segmentation(input_file, config):
    print(f"Processing {input_file}...")
    
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    z_min = np.min(points[:, 2])
    z_bottom = z_min + config["SLICING"]["height_min_offset"]
    z_top = z_min + config["SLICING"]["height_max_offset"]
    
    print(f"Slicing from {z_bottom:.2f}m to {z_top:.2f}m...")
    mask = (points[:, 2] > z_bottom) & (points[:, 2] < z_top)
    slice_points = points[mask]
    
    if len(slice_points) == 0:
        raise ValueError("No points found in the specified Z-slice range.")

    res = config["PROCESSING"]["voxel_resolution"]
    padding = config["PROCESSING"]["grid_padding"]
    
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    
    w = int(np.ceil((x_max - x_min) / res)) + (padding * 2)
    h = int(np.ceil((y_max - y_min) / res)) + (padding * 2)
    grid = np.zeros((h, w), dtype=bool)
    
    xi = ((slice_points[:, 0] - x_min) / res).astype(int) + padding
    yi = ((slice_points[:, 1] - y_min) / res).astype(int) + padding
    
    xi = np.clip(xi, 0, w - 1)
    yi = np.clip(yi, 0, h - 1)
    
    grid[yi, xi] = True

    print("Filtering walls...")
    min_size = config["PROCESSING"]["min_wall_size_px"]
    clean_walls = remove_small_objects(grid, min_size=min_size)

    print("Applying Convex Hull to seal the apartment...")
    apartment_footprint = convex_hull_image(clean_walls)
    
    indoor_air = apartment_footprint & ~clean_walls
    
    erosion_m = config["PROCESSING"]["erosion_radius"]
    print(f"Segmenting rooms (Erosion: {erosion_m}m)...")
    
    erode_iter = int(erosion_m / res)
    seeds_mask = binary_erosion(indoor_air, disk(erode_iter))
    seeds_lbl, n_seeds = label(seeds_mask)
    
    print(f"Found {n_seeds} rooms.")
    
    dist = distance_transform_edt(indoor_air)
    room_labels = watershed(-dist, seeds_lbl, mask=indoor_air)
    
    return clean_walls, room_labels

def save_results(walls, labels, config):
    viz_cfg = config["VISUALIZATION"]
    io_cfg = config["IO"]
    res = config["PROCESSING"]["voxel_resolution"]
    
    plt.figure(figsize=viz_cfg["figure_size"])
    
    plt.imshow(~walls, cmap=viz_cfg["cmap_walls"], interpolation='nearest')
    
    masked_lbl = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_lbl, cmap=viz_cfg["cmap_rooms"], alpha=viz_cfg["opacity"], interpolation='nearest')
    
    unique_ids = np.unique(labels)
    stats = []
    
    for uid in unique_ids:
        if uid == 0: continue
        
        count = np.sum(labels == uid)
        area = count * (res * res)
        stats.append({"id": int(uid), "area": round(area, 2)})
        
        coords = np.argwhere(labels == uid)
        y, x = coords.mean(axis=0)
        plt.text(x, y, f"{uid}\n{area:.1f}m²", 
                 color='white', ha='center', va='center',
                 fontweight='bold', fontsize=viz_cfg["font_size"],
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(io_cfg["output_image"], dpi=150)
    print(f"Saved image to {io_cfg['output_image']}")
    
    with open(io_cfg["output_json"], "w") as f:
        json.dump({"rooms": stats}, f, indent=2)
    print(f"Saved data to {io_cfg['output_json']}")

def main():
    parser = argparse.ArgumentParser(description="Room Segmentation Tool")
    parser.add_argument("input_file", nargs='?', default=CONFIG["IO"]["default_input_path"])
    
    parser.add_argument("--erosion", type=float, default=CONFIG["PROCESSING"]["erosion_radius"], 
                        help="Override erosion radius (meters)")
    parser.add_argument("--resolution", type=float, default=CONFIG["PROCESSING"]["voxel_resolution"], 
                        help="Override voxel resolution (meters)")
    
    args = parser.parse_args()
    
    CONFIG["PROCESSING"]["erosion_radius"] = args.erosion
    CONFIG["PROCESSING"]["voxel_resolution"] = args.resolution
    
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found at {args.input_file}")
        return

    # Run pipeline
    try:
        walls, labels = hull_segmentation(args.input_file, CONFIG)
        save_results(walls, labels, CONFIG)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()