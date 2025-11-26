import laspy
import numpy as np
import open3d as o3d
import os

def classify_surfaces(input_file):
    print(f"--- Processing: {input_file} ---")

    print("1. Loading LAS file...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    print(f"   Loaded {len(points)} points.")

    target_points = 12_000_000
    if len(points) > target_points:
        step = len(points) // target_points
        points = points[::step]
        print(f"   Decimated to {len(points)} points for analysis.")

    print("3. Estimating Normals (this may take a moment)...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    normals = np.asarray(pcd.normals)

    print("4. Classifying Surfaces...")
    
    nz = normals[:, 2]

    # If nz is near 1.0 or -1.0 -> It's pointing Up/Down -> FLOOR/CEILING
    # If nz is near 0.0        -> It's pointing Sideways -> WALL
    
    # Thresholds (Mark's "Parameters" to tune)
    floor_threshold = 0.85  # Roughly 30 degrees tolerance from vertical
    wall_threshold = 0.15   # Roughly 80 degrees from vertical (nearly straight up walls)

    flat_mask = np.abs(nz) > floor_threshold
    wall_mask = np.abs(nz) < wall_threshold
    
    flat_points = points[flat_mask]
    wall_points = points[wall_mask]

    print(f"   Found {len(flat_points)} Floor/Ceiling points.")
    print(f"   Found {len(wall_points)} Wall points.")

    print("5. Generating Visualization...")
    
    pcd.colors = o3d.utility.Vector3dVector(np.zeros(points.shape))
    colors = np.asarray(pcd.colors)
    
    colors[flat_mask] = [1, 0, 0]   # red for Floors
    colors[wall_mask] = [0, 1, 0]   # green for Walls
    print("   Opening visualization window...")
    print("   RED = Floor/Ceiling")
    print("   GREEN = Walls")
    print("   BLACK = Unclassified (Noise/Slanted)")
    
    o3d.visualization.draw_geometries([pcd], window_name="Surface Classification")

if __name__ == "__main__":
    input_file = "C:/Users/dober/Downloads/emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332/appartement_SOR_NoiseFiltered_5mm.las"
    
    if os.path.exists(input_file):
        classify_surfaces(input_file)
    else:
        print("File not found.")