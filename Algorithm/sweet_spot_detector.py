import laspy
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, label, binary_closing, binary_fill_holes
from skimage.morphology import disk, remove_small_objects
import os
import time

TEST_VALUES = np.arange(0.15, 0.40, 0.005)

EROSION = 0.30
TOP_CM = 0.05
RES = 0.05 
MIN_WALL = 50  
SEAL_RADIUS = 20   

def find_ceiling_histogram(z_coords, bin_size=0.01):
    top_percentile = np.percentile(z_coords, 80)
    high_points = z_coords[z_coords > top_percentile]
    
    if len(high_points) == 0: return np.max(z_coords)

    bins = np.arange(np.min(high_points), np.max(high_points), bin_size)
    counts, bin_edges = np.histogram(high_points, bins=bins)
    peak_idx = np.argmax(counts)
    return bin_edges[peak_idx]

def run_segmentation(points, z_ceil, thickness):
    z_top = z_ceil - TOP_CM
    z_bottom = z_top - thickness
    mask = (points[:, 2] > z_bottom) & (points[:, 2] < z_top)
    slice_points = points[mask]
    
    if len(slice_points) == 0: return 0, 0.0

    pad = 10
    x_min, y_min = np.min(points[:, 0]), np.min(points[:, 1])
    w = int(np.ceil((np.max(points[:, 0]) - x_min) / RES)) + (pad * 2)
    h = int(np.ceil((np.max(points[:, 1]) - y_min) / RES)) + (pad * 2)
    grid = np.zeros((h, w), dtype=bool)
    
    xi = ((slice_points[:, 0] - x_min) / RES).astype(int) + pad
    yi = ((slice_points[:, 1] - y_min) / RES).astype(int) + pad
    valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    grid[yi[valid], xi[valid]] = True

    clean_walls = remove_small_objects(grid, min_size=MIN_WALL)
    closed = binary_closing(clean_walls, disk(SEAL_RADIUS))
    footprint = binary_fill_holes(closed)
    air = footprint & ~clean_walls
    
    iter_n = int(EROSION / RES)
    seeds_mask = binary_erosion(air, disk(iter_n))
    seeds_lbl, n_rooms = label(seeds_mask)
    
    total_area = np.sum(seeds_lbl > 0) * (RES * RES)
    
    return n_rooms, total_area

def run_sweep(input_file):
    print(f"Loading {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    
    print("Calculating Ceiling Plane (Histogram)...")
    z_ceil = find_ceiling_histogram(points[:, 2])
    print(f"Ceiling: {z_ceil:.4f}m")
    
    results_area = []
    results_rooms = []
    
    print(f"\nRunning Sweep ({len(TEST_VALUES)} steps)...")
    start = time.time()
    
    for i, val in enumerate(TEST_VALUES):
        n, area = run_segmentation(points, z_ceil, val)
        results_rooms.append(n)
        results_area.append(area)
        
        if i % 10 == 0:
            print(f"  {val:.3f}m -> Rooms: {n}, Area: {area:.1f}")

    print(f"Done in {time.time() - start:.1f}s")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Slice Thickness (m)', fontsize=12)
    ax1.set_ylabel('Total Area (m²)', color=color, fontsize=12)
    ax1.plot(TEST_VALUES, results_area, color=color, marker='o', markersize=4, label='Total Area')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Room Count', color=color, fontsize=12)
    ax2.plot(TEST_VALUES, results_rooms, color=color, marker='x', linestyle='--', label='Room Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f'Sweet Spot Graph (Universal)\nCeiling: {z_ceil:.2f}m | Erosion: {EROSION}m', fontsize=14)
    plt.tight_layout()
    
    output = "sweet_spot_graph_v2.png"
    plt.savefig(output, dpi=150)
    print(f"\nSaved graph to: {output}")

if __name__ == "__main__":
    path = r"C:\Users\dober\Downloads\emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332\appartement_SOR.laz"
    
    if os.path.exists(path):
        run_sweep(path)
    else:
        print("File not found.")