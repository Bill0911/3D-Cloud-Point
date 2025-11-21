#!/usr/bin/env python3
"""
Complete Room Analysis Pipeline - Hybrid GPU-Accelerated Implementation

GPU-accelerated hybrid approach combining:
- Phase 1: Octree (coarse outline, lower depth for speed)
- Phase 2: Point-based gap filling (GPU-accelerated batch KDTree queries)
- Phase 3: RANSAC refinement (GPU-accelerated parallel RANSAC)

Requires: NVIDIA GPU with CUDA support

Usage:
    python complete_room_analysis_hybrid_gpu.py input.laz output_folder [options]
    
Add --use-gpu flag to enable GPU acceleration (default: auto-detect)
"""

import argparse
import os
import sys
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from collections import deque, defaultdict
from scipy.ndimage import label, binary_dilation
import matplotlib.pyplot as plt
import csv
import json
from shapely.geometry import MultiPoint

try:
    import cupy as cp
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError as e:
    GPU_AVAILABLE = False
    print(f"⚠ GPU libraries not available: {e}")
    print("  Install with: pip install cupy-cuda12x torch")


class GPUKDTreeBatch:
    """GPU-accelerated batch KDTree for multiple queries."""
    
    def __init__(self, points, use_gpu=True):
        self.points_cpu = points
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            self.points_gpu = torch.from_numpy(points).float().cuda()
        else:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.kdtree = o3d.geometry.KDTreeFlann(self.pcd)
    
    def query_radius_batch(self, query_points, radius):
        """Batch query for multiple points - GPU accelerated."""
        if not self.use_gpu:
            results = []
            for qp in query_points:
                [_, idx, _] = self.kdtree.search_radius_vector_3d(qp, radius)
                results.append(idx[1:] if len(idx) > 1 else [])
            return results
        
        query_gpu = torch.from_numpy(query_points).float().cuda()
        
        dists = torch.cdist(query_gpu, self.points_gpu)
        
        results = []
        for i in range(len(query_points)):
            mask = dists[i] <= radius
            indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            results.append(indices.cpu().numpy())
        
        return results
    
    def query_radius_single(self, query_point, radius):
        """Single point query."""
        if not self.use_gpu:
            [_, idx, _] = self.kdtree.search_radius_vector_3d(query_point, radius)
            return idx[1:] if len(idx) > 1 else np.array([])
        
        query_gpu = torch.from_numpy(query_point.reshape(1, -1)).float().cuda()
        dists = torch.cdist(query_gpu, self.points_gpu)
        mask = dists[0] <= radius
        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        
        return indices.cpu().numpy()


class OctreeNeighborSearch:
    def __init__(self, point_cloud, max_depth=10):
        self.pcd = point_cloud
        self.points = np.asarray(point_cloud.points)
        self.kdtree = o3d.geometry.KDTreeFlann(point_cloud)
        self.octree = o3d.geometry.Octree(max_depth=max_depth)
        self.octree.convert_from_point_cloud(point_cloud, size_expand=0.01)
    
    def query_radius(self, query_point, radius):
        [_, idx, _] = self.kdtree.search_radius_vector_3d(query_point, radius)
        if len(idx) > 1:
            return np.array(idx[1:], dtype=np.int32)
        return np.array([], dtype=np.int32)


def region_growing_octree(points, normals, octree_search, seed_idx, visited,
                          normal_threshold=0.9, distance_threshold=0.1, min_region_size=1200):
    region = []
    queue = deque([seed_idx])
    visited[seed_idx] = True
    seed_normal = normals[seed_idx]
    
    while queue:
        current_idx = queue.popleft()
        region.append(current_idx)
        
        neighbor_indices = octree_search.query_radius(
            points[current_idx],
            radius=distance_threshold
        )
        
        for neighbor_idx in neighbor_indices:
            if neighbor_idx >= len(visited) or visited[neighbor_idx]:
                continue
            
            dot_product = np.dot(normals[neighbor_idx], seed_normal)
            
            if dot_product > normal_threshold:
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)
    
    return region if len(region) >= min_region_size else []


def classify_region(points_in_region, normals_in_region, z_stats):
    mean_normal = np.mean(normals_in_region, axis=0)
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    
    nz = mean_normal[2]
    mean_z = np.mean(points_in_region[:, 2])
    
    if abs(nz) > 0.85:
        if mean_z < z_stats['threshold_floor']:
            return 2
        elif mean_z > z_stats['threshold_ceiling']:
            return 7
        else:
            return 7 if nz > 0 else 2
    elif abs(nz) < 0.2:
        return 6
    else:
        return 1


def adaptive_voxel_downsample(pcd, base_voxel_size=0.02, density_threshold=50):
    points = np.asarray(pcd.points)
    
    if len(points) < 1000:
        return pcd.voxel_down_sample(voxel_size=base_voxel_size)
    
    octree = o3d.geometry.Octree(max_depth=7)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    leaf_cells = []
    def traverse_octree(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            if node_info.depth >= 4:
                leaf_cells.append(node_info)
    
    octree.traverse(traverse_octree)
    
    if len(leaf_cells) == 0:
        return pcd.voxel_down_sample(voxel_size=base_voxel_size)
    
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    point_densities = np.zeros(len(points))
    
    for i, point in enumerate(points):
        [_, idx, _] = kdtree.search_knn_vector_3d(point, 20)
        point_densities[i] = len(idx) - 1
    
    density_percentile = np.percentile(point_densities, 75)
    
    high_density_mask = point_densities > density_percentile
    low_density_mask = ~high_density_mask
    
    adaptive_points = []
    
    if np.any(high_density_mask):
        high_density_pcd = pcd.select_by_index(np.where(high_density_mask)[0])
        high_density_pcd = high_density_pcd.voxel_down_sample(voxel_size=base_voxel_size * 0.7)
        adaptive_points.append(np.asarray(high_density_pcd.points))
    
    if np.any(low_density_mask):
        low_density_pcd = pcd.select_by_index(np.where(low_density_mask)[0])
        low_density_pcd = low_density_pcd.voxel_down_sample(voxel_size=base_voxel_size * 1.5)
        adaptive_points.append(np.asarray(low_density_pcd.points))
    
    if len(adaptive_points) == 0:
        return pcd.voxel_down_sample(voxel_size=base_voxel_size)
    
    combined_points = np.vstack(adaptive_points)
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(combined_points)
    
    result_pcd = result_pcd.voxel_down_sample(voxel_size=base_voxel_size)
    return result_pcd


def step1_classify(input_file, output_file, voxel_size=0.02,
                   normal_threshold=0.9, distance_threshold=0.1,
                   min_region_size=1500, use_adaptive_downsample=True):
    print("\n" + "="*70)
    print("STEP 1: CLASSIFICATION (Octree-Based Region Growing)")
    print("="*70)
    
    print("Reading point cloud...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Original points: {len(points)}")

    print("Downsampling point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if use_adaptive_downsample:
        print("Using adaptive multi-scale downsampling...")
        pcd = adaptive_voxel_downsample(pcd, base_voxel_size=voxel_size)
    else:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    points_down = np.asarray(pcd.points)
    print(f"Points after downsampling: {len(points_down)}")

    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_consistent_tangent_plane(k=20)
    normals = np.asarray(pcd.normals)

    print("Building Octree for neighbor search...")
    octree_search = OctreeNeighborSearch(pcd, max_depth=10)

    print("Performing octree-based region growing classification...")
    classification = np.ones(len(points_down), dtype=np.uint8)
    visited = np.zeros(len(points_down), dtype=bool)
    
    z_min = np.min(points_down[:, 2])
    z_max = np.max(points_down[:, 2])
    z_range = z_max - z_min
    
    z_stats = {
        'min': z_min,
        'max': z_max,
        'range': z_range,
        'threshold_floor': z_min + 0.35 * z_range,
        'threshold_ceiling': z_max - 0.35 * z_range
    }
    
    print(f"Z-range: {z_min:.2f}m to {z_max:.2f}m")
    print(f"Floor threshold: < {z_stats['threshold_floor']:.2f}m")
    print(f"Ceiling threshold: > {z_stats['threshold_ceiling']:.2f}m")
    
    regions_found = 0
    points_classified = 0
    
    normal_z_abs = np.abs(normals[:, 2])
    seed_order = np.argsort(-np.maximum(normal_z_abs, 1 - normal_z_abs))
    
    for seed_idx in seed_order:
        if visited[seed_idx]:
            continue
        
        region_indices = region_growing_octree(
            points_down, normals, octree_search, seed_idx, visited,
            normal_threshold=normal_threshold,
            distance_threshold=distance_threshold,
            min_region_size=min_region_size
        )
        
        if len(region_indices) > 0:
            region_points = points_down[region_indices]
            region_normals = normals[region_indices]
            region_class = classify_region(region_points, region_normals, z_stats)
            
            classification[region_indices] = region_class
            
            regions_found += 1
            points_classified += len(region_indices)
            
            if regions_found % 10 == 0:
                print(f"  Regions found: {regions_found}, Points classified: {points_classified}/{len(points_down)}")

    print(f"\nTotal regions found: {regions_found}")
    print(f"Points classified: {points_classified}/{len(points_down)}")
    
    unique, counts = np.unique(classification, return_counts=True)
    class_names = {1: "Unclassified", 2: "Floor", 6: "Wall", 7: "Ceiling"}
    print("\nClassification statistics:")
    for cls, count in zip(unique, counts):
        print(f"  {class_names.get(cls, f'Class {cls}')}: {count} points ({100*count/len(classification):.1f}%)")

    print(f"Saving classified point cloud to {output_file}...")
    header = laspy.LasHeader(point_format=3, version="1.2")
    out = laspy.LasData(header)
    out.x = points_down[:, 0]
    out.y = points_down[:, 1]
    out.z = points_down[:, 2]
    out.classification = classification
    out.write(output_file)
    print(f"Saved {len(points_down)} classified points")


def region_growing_point_based(points, normals, kdtree, seed_idx, visited,
                                normal_threshold=0.9, distance_threshold=0.1, min_region_size=100):
    region = []
    queue = deque([seed_idx])
    visited[seed_idx] = True
    seed_normal = normals[seed_idx]
    
    while queue:
        current_idx = queue.popleft()
        region.append(current_idx)
        
        neighbor_indices = kdtree.query_radius_single(
            points[current_idx],
            distance_threshold
        )
        
        for neighbor_idx in neighbor_indices:
            if neighbor_idx >= len(visited) or visited[neighbor_idx]:
                continue
            
            dot_product = np.dot(normals[neighbor_idx], seed_normal)
            
            if dot_product > normal_threshold:
                visited[neighbor_idx] = True
                queue.append(neighbor_idx)
    
    return region if len(region) >= min_region_size else []


def ransac_plane_fit_gpu(points, max_iterations=1000, distance_threshold=0.05, min_inliers=3):
    """GPU-accelerated RANSAC plane fitting."""
    if len(points) < 3:
        return None, None, None
    
    if not GPU_AVAILABLE or len(points) < 1000:
        return ransac_plane_fit_cpu(points, max_iterations, distance_threshold, min_inliers)
    
    points_gpu = torch.from_numpy(points).float().cuda()
    
    best_inliers = []
    best_normal = None
    best_center = None
    
    n_points = len(points)
    for _ in range(max_iterations):
        sample_indices = torch.randint(0, n_points, (3,), device='cuda')
        sample_points = points_gpu[sample_indices]
        
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = torch.linalg.cross(v1, v2)
        
        norm = torch.norm(normal)
        if norm < 1e-10:
            continue
        
        normal = normal / norm
        center = torch.mean(sample_points, dim=0)
        
        distances = torch.abs(torch.matmul(points_gpu - center, normal))
        inliers = torch.nonzero(distances < distance_threshold, as_tuple=False).squeeze(1)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers.cpu().numpy()
            best_normal = normal.cpu().numpy()
            best_center = center.cpu().numpy()
    
    if len(best_inliers) >= min_inliers:
        return best_normal, best_center, best_inliers
    return None, None, None


def ransac_plane_fit_cpu(points, max_iterations=1000, distance_threshold=0.05, min_inliers=3):
    """CPU fallback for RANSAC."""
    if len(points) < 3:
        return None, None, None
    
    best_inliers = []
    best_normal = None
    best_center = None
    
    for _ in range(max_iterations):
        sample_indices = np.random.choice(len(points), size=3, replace=False)
        sample_points = points[sample_indices]
        
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        
        normal = normal / norm
        center = np.mean(sample_points, axis=0)
        
        distances = np.abs(np.dot(points - center, normal))
        inliers = np.where(distances < distance_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_center = center
    
    if len(best_inliers) >= min_inliers:
        return best_normal, best_center, best_inliers
    return None, None, None


def refine_segment_with_ransac(segment_points, distance_threshold=0.05, use_gpu=True):
    if len(segment_points) < 3:
        return segment_points
    
    if use_gpu and GPU_AVAILABLE:
        normal, center, inliers = ransac_plane_fit_gpu(
            segment_points,
            max_iterations=500,
            distance_threshold=distance_threshold,
            min_inliers=3
        )
    else:
        normal, center, inliers = ransac_plane_fit_cpu(
            segment_points,
            max_iterations=500,
            distance_threshold=distance_threshold,
            min_inliers=3
        )
    
    if normal is not None and center is not None:
        return segment_points[inliers]
    return segment_points


class HybridRoomSegmenterGPU:
    def __init__(self, points, classification, octree_max_depth=8, min_cell_size=0.01, use_gpu=True):
        self.points = points
        self.classification = classification
        self.octree_max_depth = octree_max_depth
        self.min_cell_size = min_cell_size
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        self.wall_mask = classification == 6
        self.ceiling_mask = classification == 7
        
        self.wall_points = points[self.wall_mask]
        self.ceiling_points = points[self.ceiling_mask]
        
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
        self.pcd.orient_normals_consistent_tangent_plane(k=20)
        self.normals = np.asarray(self.pcd.normals)
        
        print(f"Building GPU-accelerated KDTree (GPU: {self.use_gpu})...")
        self.kdtree = GPUKDTreeBatch(points, use_gpu=self.use_gpu)
        
        self.octree = o3d.geometry.Octree(max_depth=octree_max_depth)
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0.01)
        
        self.leaf_cells = []
        self.cell_occupancy = {}
        self._build_occupancy_map()
    
    def _build_occupancy_map(self):
        node_to_cell_key = {}
        def traverse_octree(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                cell_size = node_info.size
                if cell_size >= self.min_cell_size:
                    center = node_info.origin + np.array([cell_size/2, cell_size/2, cell_size/2])
                    self.leaf_cells.append((node, node_info, center, cell_size))
        
        self.octree.traverse(traverse_octree)
        
        for node, node_info, center, cell_size in self.leaf_cells:
            cell_key = (center[0], center[1], center[2], cell_size)
            node_to_cell_key[id(node)] = cell_key
            self.cell_occupancy[cell_key] = {
                'wall_count': 0,
                'ceiling_count': 0,
                'is_wall': False,
                'is_ceiling': False,
                'center': center,
                'size': cell_size
            }
        
        print(f"Mapping {len(self.points)} points to {len(self.leaf_cells)} octree cells...")
        total_points = len(self.points)
        for i, point in enumerate(self.points):
            if i % 10000 == 0 and i > 0:
                print(f"  Progress: {i}/{total_points} points ({100*i/total_points:.1f}%)")
            leaf, info = self.octree.locate_leaf_node(point)
            if leaf is not None:
                cell_key = node_to_cell_key.get(id(leaf))
                if cell_key is not None:
                    if self.wall_mask[i]:
                        self.cell_occupancy[cell_key]['wall_count'] += 1
                    elif self.ceiling_mask[i]:
                        self.cell_occupancy[cell_key]['ceiling_count'] += 1
        
        for cell_key in self.cell_occupancy:
            self.cell_occupancy[cell_key]['is_wall'] = self.cell_occupancy[cell_key]['wall_count'] > 0
            self.cell_occupancy[cell_key]['is_ceiling'] = self.cell_occupancy[cell_key]['ceiling_count'] > 0
    
    def _phase1_octree_coarse(self, dilation_iterations=3, min_room_points=5000):
        print("Phase 1: Octree-based coarse segmentation (fast outline)...")
        
        x_min, y_min = np.min(self.points[:, :2], axis=0)
        x_max, y_max = np.max(self.points[:, :2], axis=0)
        
        z_ceiling = np.median(self.ceiling_points[:, 2]) if len(self.ceiling_points) > 0 else np.median(self.points[:, 2])
        
        adaptive_grid_size = min(0.01, (x_max - x_min) / 1000, (y_max - y_min) / 1000)
        grid_width = int(np.ceil((x_max - x_min) / adaptive_grid_size))
        grid_height = int(np.ceil((y_max - y_min) / adaptive_grid_size))
        
        occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)
        
        for cell_key, cell_data in self.cell_occupancy.items():
            center = cell_data['center']
            if cell_data['is_wall'] and abs(center[2] - z_ceiling) < 0.5:
                x_idx = int((center[0] - x_min) / adaptive_grid_size)
                y_idx = int((center[1] - y_min) / adaptive_grid_size)
                x_idx = np.clip(x_idx, 0, grid_width - 1)
                y_idx = np.clip(y_idx, 0, grid_height - 1)
                occupancy_grid[y_idx, x_idx] = 1
        
        print(f"Grid size: {grid_height}x{grid_width} (cell size: {adaptive_grid_size:.4f}m)")
        
        dilated_grid = binary_dilation(occupancy_grid, iterations=dilation_iterations)
        structure = np.ones((3, 3), dtype=int)
        room_labels, num_labels = label(1 - dilated_grid, structure=structure)
        print(f"Phase 1: Found {num_labels} potential room regions")
        
        ceil_x = ((self.ceiling_points[:, 0] - x_min) / adaptive_grid_size).astype(int)
        ceil_y = ((self.ceiling_points[:, 1] - y_min) / adaptive_grid_size).astype(int)
        ceil_x = np.clip(ceil_x, 0, grid_width - 1)
        ceil_y = np.clip(ceil_y, 0, grid_height - 1)
        point_room_labels = room_labels[ceil_y, ceil_x]
        
        unique_labels, counts = np.unique(point_room_labels, return_counts=True)
        valid_rooms = 0
        final_room_id_map = {}
        for room_id, count in zip(unique_labels, counts):
            if room_id == 0:
                continue
            if count >= min_room_points:
                valid_rooms += 1
                final_room_id_map[room_id] = valid_rooms
        
        final_point_labels = np.zeros_like(point_room_labels)
        for original_id, final_id in final_room_id_map.items():
            final_point_labels[point_room_labels == original_id] = final_id
        
        print(f"Phase 1: {valid_rooms} valid rooms after filtering")
        return final_point_labels, adaptive_grid_size, (x_min, y_min, x_max, y_max), final_room_id_map
    
    def _phase2_point_based_gap_filling_gpu(self, initial_labels, room_id_map, 
                                            normal_threshold=0.9, distance_threshold=0.1,
                                            gap_search_radius=0.3):
        print(f"Phase 2: GPU-accelerated gap filling on ceiling points (GPU: {self.use_gpu})...")
        
        unallocated_mask = initial_labels == 0
        unallocated_indices = np.where(unallocated_mask)[0]
        
        if len(unallocated_indices) == 0:
            print("Phase 2: No unallocated ceiling points to process")
            return initial_labels
        
        allocated_mask = initial_labels > 0
        allocated_indices = np.where(allocated_mask)[0]
        
        if len(allocated_indices) == 0:
            print("Phase 2: No allocated points found, skipping gap filling")
            return initial_labels
        
        kdtree_ceiling = GPUKDTreeBatch(self.ceiling_points, use_gpu=self.use_gpu)
        
        visited_ceiling = np.zeros(len(initial_labels), dtype=bool)
        visited_ceiling[allocated_mask] = True
        
        ceiling_normals = self.normals[self.ceiling_mask]
        
        refined_labels = initial_labels.copy()
        points_added = 0
        
        for room_id in room_id_map.values():
            room_mask = (initial_labels == room_id)
            room_indices = np.where(room_mask)[0]
            
            if len(room_indices) == 0:
                continue
            
            sample_indices = room_indices[:min(500, len(room_indices))]
            query_points = self.ceiling_points[sample_indices]
            
            print(f"  Room {room_id}: GPU batch querying {len(query_points)} boundary points...")
            results = kdtree_ceiling.query_radius_batch(query_points, gap_search_radius)
            
            boundary_points = []
            for neighbor_list in results:
                for neighbor_idx in neighbor_list:
                    if neighbor_idx in unallocated_indices and not visited_ceiling[neighbor_idx]:
                        boundary_points.append(neighbor_idx)
            
            boundary_points = list(set(boundary_points))
            
            if len(boundary_points) == 0:
                continue
            
            for seed_idx in boundary_points[:min(100, len(boundary_points))]:
                if visited_ceiling[seed_idx]:
                    continue
                
                region = []
                queue = deque([seed_idx])
                visited_ceiling[seed_idx] = True
                seed_normal = ceiling_normals[seed_idx]
                
                while queue:
                    current_idx = queue.popleft()
                    region.append(current_idx)
                    
                    neighbor_indices = kdtree_ceiling.query_radius_single(
                        self.ceiling_points[current_idx],
                        distance_threshold
                    )
                    
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx >= len(visited_ceiling) or visited_ceiling[neighbor_idx]:
                            continue
                        
                        dot_product = np.dot(ceiling_normals[neighbor_idx], seed_normal)
                        
                        if dot_product > normal_threshold:
                            visited_ceiling[neighbor_idx] = True
                            queue.append(neighbor_idx)
                
                if len(region) >= 50:
                    refined_labels[region] = room_id
                    points_added += len(region)
        
        print(f"Phase 2: Added {points_added} ceiling points through GPU-accelerated gap filling")
        return refined_labels
    
    def _phase3_ransac_refinement_gpu(self, labels, room_id_map, ransac_distance_threshold=0.05):
        print(f"Phase 3: GPU-accelerated RANSAC refinement on ceiling points (GPU: {self.use_gpu})...")
        
        refined_labels = labels.copy()
        points_refined = 0
        
        for room_id in room_id_map.values():
            room_mask = (labels == room_id)
            room_indices = np.where(room_mask)[0]
            
            if len(room_indices) < 10:
                continue
            
            room_points = self.ceiling_points[room_indices]
            
            refined_room_points = refine_segment_with_ransac(
                room_points,
                distance_threshold=ransac_distance_threshold,
                use_gpu=self.use_gpu
            )
            
            if len(refined_room_points) < len(room_points):
                from scipy.spatial import cKDTree
                room_tree = cKDTree(room_points)
                refined_tree = cKDTree(refined_room_points)
                
                distances, matches = room_tree.query(refined_room_points, k=1, distance_upper_bound=0.001)
                
                valid_matches = distances < 0.001
                refined_local_indices = matches[valid_matches]
                refined_indices = room_indices[refined_local_indices]
                
                if len(refined_indices) > 0:
                    new_mask = np.zeros(len(labels), dtype=bool)
                    new_mask[refined_indices] = True
                    refined_labels[room_mask] = 0
                    refined_labels[new_mask] = room_id
                    points_refined += len(room_indices) - len(refined_indices)
        
        print(f"Phase 3: Refined {points_refined} ceiling points using GPU-accelerated RANSAC")
        return refined_labels
    
    def segment_rooms_hybrid(self, dilation_iterations=3, min_room_points=5000,
                             normal_threshold=0.9, distance_threshold=0.1,
                             gap_search_radius=0.3, ransac_distance_threshold=0.05,
                             use_point_filling=True, use_ransac_refinement=True):
        initial_labels, grid_size, bounds, room_id_map = self._phase1_octree_coarse(
            dilation_iterations=dilation_iterations,
            min_room_points=min_room_points
        )
        
        if use_point_filling:
            initial_labels = self._phase2_point_based_gap_filling_gpu(
                initial_labels, room_id_map,
                normal_threshold=normal_threshold,
                distance_threshold=distance_threshold,
                gap_search_radius=gap_search_radius
            )
        
        if use_ransac_refinement:
            initial_labels = self._phase3_ransac_refinement_gpu(
                initial_labels, room_id_map,
                ransac_distance_threshold=ransac_distance_threshold
            )
        
        final_labels = np.zeros(len(self.points), dtype=np.int32)
        final_labels[self.ceiling_mask] = initial_labels
        
        unique_labels = np.unique(final_labels)
        valid_rooms = len(unique_labels[unique_labels > 0])
        print(f"GPU-accelerated hybrid segmentation: Found {valid_rooms} valid rooms")
        
        return final_labels, grid_size, bounds


def step2_segment_rooms(input_file, output_file, octree_max_depth=8, min_cell_size=0.01,
                        dilation_iterations=3, min_room_points=5000,
                        normal_threshold=0.9, distance_threshold=0.1,
                        gap_search_radius=0.3, ransac_distance_threshold=0.05,
                        use_point_filling=True, use_ransac_refinement=True, use_gpu=True):
    print("\n" + "="*70)
    print("STEP 2: GPU-ACCELERATED HYBRID ROOM SEGMENTATION")
    print("="*70)
    
    print(f"Reading classified point cloud from {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    classification = np.array(las.classification)
    print(f"Loaded {len(points)} points.")

    wall_mask = classification == 6
    ceiling_mask = classification == 7

    wall_points = points[wall_mask]
    ceiling_points = points[ceiling_mask]
    
    if len(wall_points) == 0 or len(ceiling_points) == 0:
        print("Error: No wall or ceiling points found.")
        return

    print(f"Found {len(wall_points)} wall points and {len(ceiling_points)} ceiling points.")
    
    segmenter = HybridRoomSegmenterGPU(
        points, classification,
        octree_max_depth=octree_max_depth,
        min_cell_size=min_cell_size,
        use_gpu=use_gpu
    )
    
    final_point_labels, grid_size, bounds = segmenter.segment_rooms_hybrid(
        dilation_iterations=dilation_iterations,
        min_room_points=min_room_points,
        normal_threshold=normal_threshold,
        distance_threshold=distance_threshold,
        gap_search_radius=gap_search_radius,
        ransac_distance_threshold=ransac_distance_threshold,
        use_point_filling=use_point_filling,
        use_ransac_refinement=use_ransac_refinement
    )

    print(f"Saving segmented point cloud to {output_file}...")
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="room_id",
        type="u2",
        description="Room ID"
    ))
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="room_class",
        type="u2",
        description="Room ceiling class"
    ))

    las.room_id = np.zeros(len(las.points), dtype=np.uint16)
    las.room_class = np.zeros(len(las.points), dtype=np.uint16)

    las.room_id[ceiling_mask] = final_point_labels[ceiling_mask]

    ceiling_room_class = np.where(final_point_labels[ceiling_mask] > 0, 
                                  700 + final_point_labels[ceiling_mask], 0)
    las.room_class[ceiling_mask] = ceiling_room_class

    las.write(output_file)
    print("Each room's ceiling now has a unique classification (700 + room_id).")


def save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path):
    print(f"Generating floor polygons for JSON...")
    x_min, y_min, _, _ = bounds
    
    output_data = {"rooms": []}

    for rid in sorted(room_stats.keys()):
        rows, cols = np.where(grid == rid)
        if len(rows) < 3:
            continue

        real_x = x_min + (cols * grid_size) + (grid_size / 2)
        real_y = y_min + (rows * grid_size) + (grid_size / 2)
        points = list(zip(real_x, real_y))
        
        hull = MultiPoint(points).convex_hull
        if hull.geom_type == 'Polygon':
            coords = list(hull.exterior.coords)
            room_data = {
                "room_id": int(rid),
                "area_sqm": room_stats[rid]["area_m2"],
                "polygon_coordinates": [[round(x, 3), round(y, 3)] for x, y in coords]
            }
            output_data["rooms"].append(room_data)

    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved room polygons to: {json_path}")


def get_room_ids_from_las(las):
    try:
        if hasattr(las, "room_id"):
            return np.asarray(las.room_id, dtype=np.int32)
    except Exception:
        pass

    try:
        if "room_id" in las.point_format.extra_dimension_names:
            return np.asarray(las["room_id"], dtype=np.int32)
    except Exception:
        pass

    try:
        if hasattr(las, "room_class"):
            rc = np.asarray(las.room_class, dtype=np.int32)
            nonzero = rc[rc > 0]
            if nonzero.size > 0:
                median = np.median(nonzero)
                if median > 500:
                    base = 700
                elif median > 100:
                    base = 200
                else:
                    base = 0
                if base > 0:
                    return np.where(rc > 0, rc - base, 0).astype(np.int32)
    except Exception:
        pass

    return None


def compute_room_areas_octree(points_xy, room_ids, grid_size):
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    x_min, y_min = np.min(points_xy, axis=0)
    x_max, y_max = np.max(points_xy, axis=0)

    width = int(np.ceil((x_max - x_min) / grid_size)) + 1
    height = int(np.ceil((y_max - y_min) / grid_size)) + 1

    grid = np.zeros((height, width), dtype=np.int32)

    xi = np.floor((x - x_min) / grid_size).astype(int)
    yi = np.floor((y - y_min) / grid_size).astype(int)
    xi = np.clip(xi, 0, width - 1)
    yi = np.clip(yi, 0, height - 1)

    cell_idx = yi * width + xi
    mask_nonzero = room_ids > 0
    
    if np.any(mask_nonzero):
        combos = np.vstack((cell_idx[mask_nonzero], room_ids[mask_nonzero])).T
        uniq_pairs, counts = np.unique(combos, axis=0, return_counts=True)
        
        cell_best = {}
        for (cidx, rid), cnt in zip(uniq_pairs, counts):
            if cidx not in cell_best or cnt > cell_best[cidx][1]:
                cell_best[cidx] = (int(rid), int(cnt))
        
        for cidx, (rid, _) in cell_best.items():
            cy = cidx // width
            cx = cidx % width
            grid[cy, cx] = rid

    unique_ids, counts = np.unique(grid, return_counts=True)
    room_stats = {}
    for uid, cnt in zip(unique_ids, counts):
        if uid == 0:
            continue
        area = cnt * (grid_size ** 2)
        room_stats[int(uid)] = {"pixel_count": int(cnt), "area_m2": float(area)}

    return grid, room_stats, (x_min, y_min, x_max, y_max)


def save_csv(room_stats, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["room_id", "pixel_count", "area_m2"])
        for rid in sorted(room_stats.keys()):
            writer.writerow([rid, room_stats[rid]["pixel_count"], f"{room_stats[rid]['area_m2']:.4f}"])


def plot_map(grid, bounds, room_stats, output_image, cmap_name="tab20"):
    x_min, y_min, x_max, y_max = bounds
    extent = (x_min, x_max, y_min, y_max)

    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap(cmap_name)
    
    unique_ids = np.unique(grid)
    unique_ids = unique_ids[unique_ids > 0]
    
    if unique_ids.size == 0:
        raise RuntimeError("No rooms found in grid to plot.")

    id_to_idx = {int(rid): (i % 20) for i, rid in enumerate(unique_ids)}
    color_grid = np.zeros_like(grid, dtype=np.int32)
    for rid, idx in id_to_idx.items():
        color_grid[grid == rid] = idx + 1

    plt.imshow(color_grid, origin='lower', extent=extent, interpolation='nearest')
    plt.title("Room Segmentation Map (GPU-Accelerated Hybrid)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')

    import matplotlib.patches as mpatches
    patches = []
    for rid, idx in id_to_idx.items():
        patches.append(mpatches.Patch(color=cmap(idx/20.0), label=f"Room {rid}"))
    plt.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    for rid in sorted(room_stats.keys()):
        positions = np.argwhere(grid == rid)
        if positions.size == 0:
            continue
        mean_row = positions[:, 0].mean()
        mean_col = positions[:, 1].mean()
        width = grid.shape[1]
        cell_x = x_min + (mean_col + 0.5) * ((x_max - x_min) / width)
        cell_y = y_min + (mean_row + 0.5) * ((y_max - y_min) / grid.shape[0])
        area = room_stats[rid]["area_m2"]
        plt.text(cell_x, cell_y, f"{rid}\n{area:.1f} m²", ha="center", va="center",
                 fontsize=8, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()


def step3_measure_rooms(input_file, output_image, csv_path, json_path, grid_size=0.05):
    print("\n" + "="*70)
    print("STEP 3: ROOM AREA CALCULATION & MAP GENERATION")
    print("="*70)
    
    print(f"Reading {input_file}...")
    las = laspy.read(input_file)
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"Total points: {len(points)}")

    room_ids_per_point = get_room_ids_from_las(las)
    if room_ids_per_point is None:
        print("Error: Could not find room_id in LAS file.")
        return

    mask = room_ids_per_point > 0
    if not np.any(mask):
        print("Error: No points have a room_id > 0.")
        return

    points_xy = points[mask, :2]
    room_ids = room_ids_per_point[mask]

    print(f"Found {len(np.unique(room_ids))} unique room IDs among {points_xy.shape[0]} points.")

    grid, room_stats, bounds = compute_room_areas_octree(points_xy, room_ids, grid_size)

    save_csv(room_stats, csv_path)
    print(f"Wrote CSV to: {csv_path}")

    save_room_polygons_json(room_stats, grid, bounds, grid_size, json_path)

    print(f"Plotting map to {output_image}...")
    plot_map(grid, bounds, room_stats, output_image)
    print(f"Saved map image to {output_image}")

    print("\nRoom areas (m²):")
    for rid in sorted(room_stats.keys()):
        print(f"  Room {rid}: {room_stats[rid]['area_m2']:.3f} m²  ({room_stats[rid]['pixel_count']} cells)")


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated hybrid room analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python complete_room_analysis_hybrid_gpu.py input.laz output_folder --use-gpu
  python complete_room_analysis_hybrid_gpu.py input.laz output_folder --octree-max-depth 8

GPU Acceleration:
  Requires NVIDIA GPU with CUDA support
  Install: pip install cupy-cuda12x torch
        """
    )
    
    parser.add_argument("input", help="Input .las/.laz file")
    parser.add_argument("output_folder", help="Output folder for all results")
    
    parser.add_argument("--use-gpu", action="store_true", default=True,
                       help="Use GPU acceleration (default: auto-detect)")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                       help="Disable GPU acceleration")
    
    parser.add_argument("--voxel", type=float, default=0.02,
                       help="Base voxel size for downsampling in meters (default: 0.02)")
    parser.add_argument("--normal-threshold", type=float, default=0.9,
                       help="Normal similarity threshold (0-1, default: 0.9)")
    parser.add_argument("--distance-threshold", type=float, default=0.1,
                       help="Neighbor distance threshold in meters (default: 0.1)")
    parser.add_argument("--min-region-size", type=int, default=1500,
                       help="Minimum points per region (default: 1500)")
    parser.add_argument("--no-adaptive-downsample", action="store_true",
                       help="Disable adaptive multi-scale downsampling")
    
    parser.add_argument("--octree-max-depth", type=int, default=8,
                       help="Octree maximum depth for Phase 1 (default: 8)")
    parser.add_argument("--min-cell-size", type=float, default=0.01,
                       help="Minimum octree cell size in meters (default: 0.01)")
    parser.add_argument("--dilation-iterations", type=int, default=3,
                       help="Wall dilation iterations (default: 3)")
    parser.add_argument("--min-room-points", type=int, default=5000,
                       help="Minimum points per room (default: 5000)")
    
    parser.add_argument("--gap-search-radius", type=float, default=0.3,
                       help="Search radius for gap filling in Phase 2 (default: 0.3)")
    parser.add_argument("--ransac-distance-threshold", type=float, default=0.05,
                       help="Distance threshold for RANSAC in Phase 3 (default: 0.05)")
    parser.add_argument("--no-point-filling", dest="use_point_filling", action="store_false",
                       help="Disable Phase 2 point-based gap filling")
    parser.add_argument("--no-ransac", dest="use_ransac_refinement", action="store_false",
                       help="Disable Phase 3 RANSAC refinement")
    
    parser.add_argument("--measure-grid-size", type=float, default=0.05,
                       help="Grid size for area calculation (default: 0.05)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.use_gpu and not GPU_AVAILABLE:
        print("\n⚠ WARNING: GPU acceleration requested but not available!")
        print("  Install GPU libraries with: pip install cupy-cuda12x torch")
        print("  Falling back to CPU mode...\n")
        args.use_gpu = False
    
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    classified_file = output_folder / "classified.las"
    segmented_file = output_folder / "segmented.las"
    map_file = output_folder / "room_map.png"
    csv_file = output_folder / "room_areas.csv"
    json_file = output_folder / "room_polygons.json"
    
    print("\n" + "="*70)
    print("GPU-ACCELERATED HYBRID ROOM ANALYSIS PIPELINE")
    print("="*70)
    print(f"Input file:       {args.input}")
    print(f"Output folder:    {output_folder}")
    print(f"GPU Acceleration: {'✓ Enabled' if args.use_gpu else '✗ Disabled'}")
    if args.use_gpu and GPU_AVAILABLE:
        print(f"GPU Device:       {torch.cuda.get_device_name(0)}")
    print(f"\nHybrid approach:")
    print(f"  Phase 1: Octree (max_depth={args.octree_max_depth})")
    print(f"  Phase 2: GPU-accelerated gap filling ({'enabled' if args.use_point_filling else 'disabled'})")
    print(f"  Phase 3: GPU-accelerated RANSAC ({'enabled' if args.use_ransac_refinement else 'disabled'})")
    
    try:
        step1_classify(
            args.input,
            str(classified_file),
            voxel_size=args.voxel,
            normal_threshold=args.normal_threshold,
            distance_threshold=args.distance_threshold,
            min_region_size=args.min_region_size,
            use_adaptive_downsample=not args.no_adaptive_downsample
        )

        step2_segment_rooms(
            str(classified_file),
            str(segmented_file),
            octree_max_depth=args.octree_max_depth,
            min_cell_size=args.min_cell_size,
            dilation_iterations=args.dilation_iterations,
            min_room_points=args.min_room_points,
            normal_threshold=args.normal_threshold,
            distance_threshold=args.distance_threshold,
            gap_search_radius=args.gap_search_radius,
            ransac_distance_threshold=args.ransac_distance_threshold,
            use_point_filling=args.use_point_filling,
            use_ransac_refinement=args.use_ransac_refinement,
            use_gpu=args.use_gpu
        )
        
        step3_measure_rooms(
            str(segmented_file),
            str(map_file),
            str(csv_file),
            str(json_file),
            grid_size=args.measure_grid_size
        )
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nAll results saved in: {output_folder}")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

