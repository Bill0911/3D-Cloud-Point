import laspy
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

CONFIG = {
    "decimation_target": 300_000,

    "normal_radius": 2.0,        
    "normal_max_nn": 60,         
    
    "normal_feature_weight": 2.5, 

    "cluster_eps": 0.3,
    "cluster_min_points": 20,
    
    "vertical_axis_index": 2,
    "viz_floor_color_channel": 0,    
    "viz_wall_color_channel": 2,      
    
    "viz_colormap_name": "tab20",
    "viz_noise_color": [0, 0, 0],     
    
    "viz_rgb_channel_count": 3,
    "math_safe_divisor": 1.0,  
    "min_clusters_for_viz": 0 
}

def validate_initial_setup(input_file, cfg):
    print(f"--- Validation Run: {input_file} ---", flush=True)

    print("1. Loading & Decimating...", flush=True)
    try:
        las = laspy.read(input_file)
        points = np.vstack((las.x, las.y, las.z)).transpose()
    except Exception as e:
        print(f"Error loading file: {e}", flush=True)
        return
    
    target_points = cfg["decimation_target"]
    if len(points) > target_points:
        step = len(points) // target_points
        points = points[::step]
    
    print(f"   Working with {len(points)} points.", flush=True)

    print("2. Estimating Normals", flush=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=cfg["normal_radius"], 
        max_nn=cfg["normal_max_nn"]
    ))
    
    normals = np.asarray(pcd.normals)
    
    print("   Showing Normal Visualization (Close window to continue)...", flush=True)
    colors = np.zeros_like(points)
    vert_idx = cfg["vertical_axis_index"]
    floor_ch = cfg["viz_floor_color_channel"]
    wall_ch = cfg["viz_wall_color_channel"]
    verticality = np.abs(normals[:, vert_idx])
    colors[:, floor_ch] = verticality
    colors[:, wall_ch] = 1 - verticality
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="AC1: Normal Visualization")

    print("3. Running Clustering (using Scikit-Learn)...", flush=True)
    
    weighted_normals = normals * cfg["normal_feature_weight"]
    combined_features = np.hstack((points, weighted_normals))
    
    clusterer = DBSCAN(eps=cfg["cluster_eps"], min_samples=cfg["cluster_min_points"])
    labels = clusterer.fit_predict(combined_features)
    
    max_label = labels.max()
    num_clusters = max_label + 1 
    print(f"   Found {num_clusters} clusters.", flush=True)

    print("   Showing Cluster Visualization...", flush=True)
    
    if num_clusters > cfg["min_clusters_for_viz"]:
        cmap = plt.get_cmap(cfg["viz_colormap_name"])
        safe_denominator = max_label if max_label > 0 else cfg["math_safe_divisor"]
        normalized_labels = labels / safe_denominator
        
        num_channels = cfg["viz_rgb_channel_count"]
        cluster_colors = cmap(normalized_labels)[:, :num_channels]
        
        noise_mask = labels < 0
        cluster_colors[noise_mask] = cfg["viz_noise_color"]
        
        pcd.colors = o3d.utility.Vector3dVector(cluster_colors)
        o3d.visualization.draw_geometries([pcd], window_name="AC2: Cluster Validation")
    else:
        print("   No clusters found to visualize.", flush=True)

if __name__ == "__main__":
    input_path = "C:/Users/dober/Downloads/emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332/appartement_SOR_NoiseFiltered_5mm.las" 
    if os.path.exists(input_path):
        validate_initial_setup(input_path, CONFIG)
    else:
        print(f"File not found: {input_path}", flush=True)