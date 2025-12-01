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

def hull_segmentation(input_file, resolution=0.05, erosion_m=0.4):
    print(f"Processing {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    # 1. SLICE HIGH (1.95m)
    # This height effectively ignores low furniture/kitchen islands
    z_min = np.min(points[:, 2])
    mask = (points[:, 2] > z_min + 2.1) & (points[:, 2] < z_min + 2.54)
    slice_points = points[mask]
    
    # 2. CREATE GRID
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    x_max, y_max = np.max(points[:, 0]), np.max(points[:, 1])
    
    padding = 20
    w = int(np.ceil((x_max - x_min) / resolution)) + (padding * 2)
    h = int(np.ceil((y_max - y_min) / resolution)) + (padding * 2)
    grid = np.zeros((h, w), dtype=bool)
    
    xi = ((slice_points[:, 0] - x_min) / resolution).astype(int) + padding
    yi = ((slice_points[:, 1] - y_min) / resolution).astype(int) + padding
    grid[yi, xi] = True

    # 3. WALL FILTERING (Kitchen Island Removal)
    print("Filtering walls...")
    # Remove small floating blobs (noise/islands) < 200 pixels
    # We DO NOT filter by "Largest Component" anymore to avoid deleting valid walls.
    clean_walls = remove_small_objects(grid, min_size=200)

    # 4. THE RUBBER BAND (Convex Hull)
    print("Applying Convex Hull (Rubber Band) to seal the apartment...")
    # This wraps the entire set of walls in a closed polygon.
    # It creates a perfect "Floorplate" with zero leaks.
    apartment_footprint = convex_hull_image(clean_walls)
    
    # Define "Indoor Air": The Footprint MINUS the Walls
    indoor_air = apartment_footprint & ~clean_walls
    
    # 5. SEGMENTATION
    print(f"Segmenting rooms (Erosion: {erosion_m}m)...")
    erode_iter = int(erosion_m / resolution)
    
    # Erode to find seeds (centers of rooms)
    seeds_mask = binary_erosion(indoor_air, disk(erode_iter))
    seeds_lbl, n_seeds = label(seeds_mask)
    print(f"✓ Found {n_seeds} rooms.")
    
    # Watershed to grow seeds back to the walls
    dist = distance_transform_edt(indoor_air)
    room_labels = watershed(-dist, seeds_lbl, mask=indoor_air)
    
    return clean_walls, room_labels, resolution

def save_results(walls, labels, resolution):
    # 1. Visualization
    plt.figure(figsize=(15, 15))
    # Plot Walls (Black)
    plt.imshow(~walls, cmap='gray', interpolation='nearest')
    
    # Plot Rooms (Colors)
    masked_lbl = np.ma.masked_where(labels == 0, labels)
    plt.imshow(masked_lbl, cmap='tab20', alpha=0.9, interpolation='nearest')
    
    # Calculate Stats & Plot Text
    unique_ids = np.unique(labels)
    stats = []
    
    for uid in unique_ids:
        if uid == 0: continue
        count = np.sum(labels == uid)
        area = count * (resolution * resolution)
        stats.append({"id": int(uid), "area": round(area, 2)})
        
        # Center of room
        coords = np.argwhere(labels == uid)
        y, x = coords.mean(axis=0)
        plt.text(x, y, f"{uid}\n{area:.1f}m²", color='white', ha='center', va='center',
                 fontweight='bold', fontsize=9,
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("segmentation_hull.png", dpi=150)
    print("✓ Saved image to segmentation_hull.png")
    
    # 2. JSON
    with open("room_data.json", "w") as f:
        json.dump({"rooms": stats}, f, indent=2)
    print("✓ Saved data to room_data.json")

def main():
    # --- HARDCODED PATH ---
    default_path = "C:/Users/dober/Downloads/emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332/appartement_SOR_NoiseFiltered_5mm.las"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs='?', default=default_path)
    # Default erosion 0.4m. 
    # If the Big Room is split into two, INCREASE this to 0.6 or 0.8.
    parser.add_argument("--erosion", type=float, default=0.4)
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print("Error: Input file not found.")
        return

    walls, labels, res = hull_segmentation(args.input_file, erosion_m=args.erosion)
    save_results(walls, labels, res)

if __name__ == "__main__":
    main()