import laspy
import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_closing, generate_binary_structure

def calculate_ceiling_area(las_file_path, output_json="ceiling_results.json"):
    print(f"--- Processing: {las_file_path} ---")
    
    print("1. Loading LAS file...")
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    target_points = 80_000_000
    if len(points) > target_points:
        step = len(points) // target_points
        points = points[::step]

    z_mean = np.mean(points[:, 2])
    z_cut = max(z_mean, np.min(points[:, 2]) + 1.5) 
    print(f"2. Isolating upper room (Z > {z_cut:.2f}m)...")
    top_half_mask = points[:, 2] > z_cut
    top_points = points[top_half_mask]

    print("   Applying SOAR Filter (Statistical Outlier Removal)...")
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(top_points)
    
    cl, ind = pcd_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    top_points = top_points[ind]
    print(f"Points remaining after cleaning: {len(top_points)}")

    print("Running RANSAC to find Ceiling Plane...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(top_points)
    
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model

    print("Isolating ceiling slice...")
    norm = np.sqrt(a**2 + b**2 + c**2)
    distances = np.abs(np.dot(points, [a, b, c]) + d) / norm
    ceiling_mask = distances < 0.10
    ceiling_points = points[ceiling_mask]

    print("Projecting & Closing Gaps...")
    grid_size = 0.05
    
    min_x, min_y = np.min(ceiling_points[:, 0]), np.min(ceiling_points[:, 1])
    max_x, max_y = np.max(ceiling_points[:, 0]), np.max(ceiling_points[:, 1])
    
    width = int(np.ceil((max_x - min_x) / grid_size))
    height = int(np.ceil((max_y - min_y) / grid_size))
    
    grid = np.zeros((height + 1, width + 1), dtype=int)
    indices_x = ((ceiling_points[:, 0] - min_x) / grid_size).astype(int)
    indices_y = ((ceiling_points[:, 1] - min_y) / grid_size).astype(int)
    grid[indices_y, indices_x] = 1


    struct = generate_binary_structure(2, 2) 
    processed_grid = binary_closing(grid, structure=struct, iterations=5)
    
    pixel_count = np.sum(processed_grid)
    area_m2 = pixel_count * (grid_size * grid_size)
    
    print(f"\n[OK] FINAL CALCULATION:")
    print(f"   Service Area: {area_m2:.2f} square meters")
    
    result_data = {
        "file": las_file_path,
        "service_area_m2": round(area_m2, 2),
        "plane_equation": [a, b, c, d],
        "floor_z_estimate": np.mean(ceiling_points[:, 2]) - 2.4
    }
    
    with open(output_json, "w") as f:
        json.dump(result_data, f, indent=4)
    print(f"   Saved JSON: {output_json}")

    plt.figure(figsize=(10, 10))
    plt.title(f"Ceiling Projection (Area: {area_m2:.2f} m2)")
    plt.imshow(processed_grid, origin='lower', cmap='Greys_r')
    plt.colorbar(label='Occupancy')
    plt.show()

if __name__ == "__main__":
    input_file = "C:/Users/dober/Downloads/emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332/appartement.las"
    
    import os
    if os.path.exists(input_file):
        calculate_ceiling_area(input_file)
    else:
        print(f"File not found: {input_file}")