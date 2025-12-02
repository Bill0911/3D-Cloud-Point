import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, label, binary_closing, binary_fill_holes
from skimage.segmentation import watershed
from skimage.morphology import disk, remove_small_objects
from scipy.ndimage import distance_transform_edt
import argparse
import os
import json

#-------CONTROL PANEL--------#
CONFIG = {
    "IO": {
        "default_input_path": r"C:\Users\dober\Downloads\emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332\appartement_SOR.laz",
        "output_image": "segmentation_shrink_wrap.png",
        "output_json": "room_data.json"
    },
    "SLICING": {
        "ignore_top_cm": 0.06, # KEY (set 0.07 disappears Room 8 and Room 1) 
        "slice_thickness": 0.185 # KEY (set 0.186 disappears Room 8 (real room))
    },
    "PROCESSING": {
        "voxel_resolution": 0.05, # KEY (0.06 makes a small 'phantom' room in Room 5)
        "erosion_radius": 0.5, # KEY (0.56 makes several non-sense rooms)
        "grid_padding": 20,
        "min_wall_size_px": 100,
        "room_polish_radius": 12,
        
        "seal_radius_px": 20 
    },
    "VISUALIZATION": {
        "figure_size": (15, 15),
        "font_size": 10,
        "cmap_walls": "gray",
        "cmap_rooms": "tab20",
        "opacity": 1
    }
}
#--------------------------------#

def find_ceiling_plane(z_coords, bin_size=0.04):
    top_percentile = np.percentile(z_coords, 80)
    high_points = z_coords[z_coords > top_percentile]
    bins = np.arange(np.min(high_points), np.max(high_points), bin_size)
    counts, bin_edges = np.histogram(high_points, bins=bins)
    peak_idx = np.argmax(counts)
    return bin_edges[peak_idx]

def segmentation_logic(input_file, config):
    print(f"Processing {input_file}...")
    
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    z_ceil = find_ceiling_plane(points[:, 2])
    print(f"Ceiling Plane: {z_ceil:.2f}m")
    
    z_top = z_ceil - config["SLICING"]["ignore_top_cm"]
    z_bottom = z_top - config["SLICING"]["slice_thickness"]
    mask = (points[:, 2] > z_bottom) & (points[:, 2] < z_top)
    slice_points = points[mask]

    res = config["PROCESSING"]["voxel_resolution"]
    pad = config["PROCESSING"]["grid_padding"]
    
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    
    w = int(np.ceil((x_max - x_min) / res)) + (pad * 2)
    h = int(np.ceil((y_max - y_min) / res)) + (pad * 2)
    grid = np.zeros((h, w), dtype=bool)
    
    xi = ((slice_points[:, 0] - x_min) / res).astype(int) + pad
    yi = ((slice_points[:, 1] - y_min) / res).astype(int) + pad
    grid[yi, xi] = True

    print("Cleaning noise...")
    clean_walls = remove_small_objects(grid, min_size=config["PROCESSING"]["min_wall_size_px"])

    print("Sealing apartment (Shrink Wrap)...")
    seal_radius = config["PROCESSING"]["seal_radius_px"]
    
    closed_structure = binary_closing(clean_walls, disk(seal_radius))
    
    apartment_footprint = binary_fill_holes(closed_structure)
    
    indoor_air = apartment_footprint & ~clean_walls

    print("Segmenting...")
    erode_iter = int(config["PROCESSING"]["erosion_radius"] / res)
    seeds_mask = binary_erosion(indoor_air, disk(erode_iter))
    seeds_lbl, n_seeds = label(seeds_mask)
    
    dist = distance_transform_edt(indoor_air)
    room_labels = watershed(-dist, seeds_lbl, mask=indoor_air)
    print(f"Found {n_seeds} rooms.")

    print("Polishing...")
    polish_radius = config["PROCESSING"]["room_polish_radius"]
    final_labels = room_labels.copy()
    unique_ids = np.unique(final_labels)
    
    for uid in unique_ids:
        if uid == 0: continue
        room_mask = (final_labels == uid)
        polished = binary_closing(room_mask, disk(polish_radius))
        final_labels[(polished) & (final_labels == 0)] = uid

    return clean_walls, final_labels

def save_results(walls, labels, config):
    viz = config["VISUALIZATION"]
    res = config["PROCESSING"]["voxel_resolution"]
    
    plt.figure(figsize=viz["figure_size"])
    plt.imshow(~walls, cmap=viz["cmap_walls"], interpolation='nearest')
    
    masked_lbl = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_lbl, cmap=viz["cmap_rooms"], alpha=viz["opacity"], interpolation='nearest')
    
    unique_ids = np.unique(labels)
    stats = []
    for uid in unique_ids:
        if uid == 0: continue
        count = np.sum(labels == uid)
        area = count * (res * res)
        stats.append({"id": int(uid), "area": round(area, 2)})
        
        coords = np.argwhere(labels == uid)
        y, x = coords.mean(axis=0)
        plt.text(x, y, f"{uid}\n{area:.1f}m²", color='white', ha='center', va='center',
                 fontweight='bold', fontsize=viz["font_size"],
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(config["IO"]["output_image"], dpi=150)
    print(f"Saved image to {config['IO']['output_image']}")
    
    with open(config["IO"]["output_json"], "w") as f:
        json.dump({"rooms": stats}, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default=CONFIG["IO"]["default_input_path"])
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    walls, labels = segmentation_logic(args.input_file, CONFIG)
    save_results(walls, labels, CONFIG)

if __name__ == "__main__":
    main()