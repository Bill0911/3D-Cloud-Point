import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
from collections import deque
from ..io.las_io import load_las_points, save_las_with_classification

def _region_grow(points, normals, kd_tree, seed, visited,
                 angle_thr, dist_thr, min_region, max_z_diff):
    region = []
    q = deque([seed])
    visited[seed] = True

    while q:
        idx = q.popleft()
        p = points[idx]
        n = normals[idx]
        region.append(idx)

        neighbors = kd_tree.query_radius(p.reshape(1, -1), r=dist_thr)[0]

        for nb in neighbors:
            if visited[nb]:
                continue

            if max_z_diff is not None:
                if abs(points[nb, 2] - p[2]) > max_z_diff:
                    continue

            dotv = np.dot(normals[nb], n)
            dotv = np.clip(dotv, -1.0, 1.0)
            angle = np.degrees(np.arccos(dotv))
            if angle <= angle_thr:
                visited[nb] = True
                q.append(nb)

    return region if len(region) >= min_region else None


def _classify_region(region_points, region_normals, z_stats):
    n = np.mean(region_normals, axis=0)
    n /= (np.linalg.norm(n) + 1e-9)
    nz = n[2]
    z = np.mean(region_points[:, 2])

    if abs(nz) > 0.85:
        if z < z_stats["threshold_floor"]:
            return 2
        if z > z_stats["threshold_ceiling"]:
            return 7
        return 7 if nz > 0 else 2

    if abs(nz) < 0.2:
        return 6

    return 1


def run_classification(input_file, output_file, cfg):
    print("STEP 1: Classification")

    pts = load_las_points(input_file)
    print(f"Loaded {len(pts)} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    pcd = pcd.voxel_down_sample(voxel_size=cfg["voxel_size"])
    points = np.asarray(pcd.points)
    print(f"Downsampled -> {len(points)} points")

    if len(points) == 0:
        raise ValueError("Downsampling removed all points.")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    pcd.orient_normals_consistent_tangent_plane(k=20)
    normals = np.asarray(pcd.normals)
    print("Normals done")

    tree = KDTree(points)

    z = points[:, 2]
    z_floor = float(np.percentile(z, cfg["floor_pct"]))
    z_ceil = float(np.percentile(z, cfg["ceil_pct"]))
    span = max(z_ceil - z_floor, 1e-6)

    z_stats = {
        "threshold_floor": z_floor + 0.10 * span,
        "threshold_ceiling": z_ceil - 0.10 * span
    }

    knn_tree = o3d.geometry.KDTreeFlann(pcd)
    curvature = np.zeros(len(points))
    for i in range(len(points)):
        _, idxs, _ = knn_tree.search_knn_vector_3d(pcd.points[i], 10)
        nn = normals[idxs]
        curvature[i] = np.mean(np.linalg.norm(nn - nn.mean(axis=0), axis=1))

    seed_order = np.argsort(curvature)

    visited = np.zeros(len(points), dtype=bool)
    labels = np.ones(len(points), dtype=np.uint8)

    angle_thr = cfg["angle_threshold"]
    dist_thr = cfg["distance_threshold"]
    min_region = cfg["min_region_size"]
    max_z_diff = cfg["max_z_diff"]

    region_count = 0
    classified_count = 0

    print("Region growing...")
    for seed in seed_order:
        if visited[seed]:
            continue

        region = _region_grow(
            points, normals, tree, seed, visited,
            angle_thr, dist_thr, min_region, max_z_diff
        )

        if region is None:
            continue

        r_pts = points[region]
        r_nrm = normals[region]
        cls = _classify_region(r_pts, r_nrm, z_stats)
        labels[region] = cls

        region_count += 1
        classified_count += len(region)

    print(f"Regions: {region_count}")
    print(f"Classified: {classified_count}/{len(points)}")

    have_floor = np.any(labels == 2)
    have_ceil = np.any(labels == 7)

    if not have_floor:
        print("[Fallback] Floor from Z band")
        labels[z < z_stats["threshold_floor"]] = 2

    if not have_ceil:
        print("[Fallback] Ceiling from Z band")
        labels[z > z_stats["threshold_ceiling"]] = 7

    mid = (z >= z_stats["threshold_floor"]) & (z <= z_stats["threshold_ceiling"])
    vertical = np.abs(normals[:, 2]) < 0.4
    wall_idx = mid & vertical

    before = np.sum(labels == 6)
    labels[wall_idx] = 6
    after = np.sum(labels == 6)
    print(f"Walls: {before} → {after}")

    save_las_with_classification(output_file, points, labels)
    print(f"Saved -> {output_file}")

    return {
        "points": points,
        "labels": labels,
        "z_stats": z_stats,
        "regions": region_count
    }