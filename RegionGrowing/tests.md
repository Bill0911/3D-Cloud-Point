# Benchmark Test Commands

PowerShell-ready commands for testing all room analysis scripts. All commands assume you are in the project root (`3D-Cloud-Point`) and input file is `RegionGrowing\appartement.las` (adjust as needed).

## Script Overview

- **`complete_room_analysis.py`** - Point-based (KD-Tree), tests noise threshold (min-region-size)
- **`complete_room_analysis_gpu.py`** - GPU-Accelerated point-based (fastest with NVIDIA GPU)
- **`complete_room_analysis_octree.py`** - Octree-based with research enhancements (slowest, highest accuracy)
- **`complete_room_analysis_hybrid.py`** - Hybrid (Octree → Point-based → RANSAC), balanced speed/accuracy
- **`complete_room_analysis_advanced.py`** - Advanced (all features + room classification)

---

## 1. Point-Based Script (complete_room_analysis.py)

**Focus: Minimum Cluster Size (Noise Threshold) Testing**

The `--min-region-size` parameter controls noise filtering. Setting it too low includes residual noise clusters that distort area calculations by inflating min/max coordinates.

### Test 1.1: Low Noise Threshold (Too Low - Includes Noise)
**Purpose**: Demonstrate how low threshold causes noise clusters to distort area calculations

```powershell
py -3.11 complete_room_analysis.py appartement.las output_point_T1 --voxel 0.015 --min-region-size 500 --min-room-points 3000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Issues**: 
- Small noise clusters included in results
- Inflated room areas due to noise points affecting boundaries
- Distorted min/max coordinates

### Test 1.2: Optimal Noise Threshold (Recommended)
**Purpose**: Demonstrate optimal threshold that filters noise while preserving real features

```powershell
py -3.11 complete_room_analysis.py appartement.las output_point_T2 --voxel 0.015 --min-region-size 1500 --min-room-points 5000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Results**:
- Noise clusters filtered out
- Accurate area calculations
- Clean room boundaries

### Test 1.3: High Noise Threshold (Conservative Filtering)
**Purpose**: Demonstrate very strict filtering - may remove small but valid features

```powershell
py -3.11 complete_room_analysis.py appartement.las output_point_T3 --voxel 0.015 --min-region-size 3000 --min-room-points 8000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Results**:
- Maximum noise filtering
- May remove small valid rooms/features
- Very clean results but potentially over-filtered

**Key Learning**: Compare area calculations across these three tests to see impact of `--min-region-size` on noise filtering and area accuracy.

---

## 2. GPU-Accelerated Script (complete_room_analysis_gpu.py) 🚀

**Focus: Maximum Speed with GPU Acceleration**

Requires NVIDIA GPU with CUDA support. Install dependencies:
```powershell
py -3.11 -m pip install cupy-cuda12x torch-cluster
```

### Test 2.1: GPU - Low Noise Threshold
**Purpose**: Fast processing with GPU acceleration, low noise filtering

```powershell
py -3.11 complete_room_analysis_gpu.py appartement.las output_gpu_T1 --use-gpu --voxel 0.015 --min-region-size 500 --min-room-points 3000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Speed**: ~5-15 minutes for 2.7M points (5-10x faster than CPU)  
**GPU Memory**: ~2-4 GB VRAM usage

### Test 2.2: GPU - Optimal Noise Threshold (Recommended)
**Purpose**: GPU-accelerated with optimal noise filtering for production use

```powershell
py -3.11 complete_room_analysis_gpu.py appartement.las output_gpu_T2 --use-gpu --voxel 0.015 --min-region-size 1500 --min-room-points 5000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Speed**: ~5-15 minutes for 2.7M points  
**GPU Memory**: ~2-4 GB VRAM usage  
**Best for**: Production use with RTX 4080 or similar GPU

### Test 2.3: GPU - High Accuracy Fine Voxel
**Purpose**: Maximum detail with GPU acceleration

```powershell
py -3.11 complete_room_analysis_gpu.py appartement.las output_gpu_T3 --use-gpu --voxel 0.01 --min-region-size 3000 --min-room-points 8000 --grid-size 0.005 --dilation-iterations 3 --measure-grid-size 0.04
```

**Expected Speed**: ~10-25 minutes for 2.7M points  
**GPU Memory**: ~4-8 GB VRAM usage  
**Best for**: High-end GPUs (RTX 4080/4090)

### Test 2.4: CPU Fallback (No GPU)
**Purpose**: Test same script without GPU acceleration

```powershell
py -3.11 complete_room_analysis_gpu.py appartement.las output_gpu_cpu --no-gpu --voxel 0.015 --min-region-size 1500 --min-room-points 5000 --grid-size 0.01 --dilation-iterations 3 --measure-grid-size 0.05
```

**Expected Speed**: ~30-90 minutes (same as CPU-only script)

**Performance Comparison**: Compare output_gpu_T2 vs output_gpu_cpu to see GPU speedup.

---

## 3. Octree Script (complete_room_analysis_octree.py)

**Focus: Accuracy and Research Features**

### Test 2.1: Medium Dataset (Balanced)
**Purpose**: Standard configuration for medium-sized apartment scans

```powershell
py -3.11 complete_room_analysis_octree.py appartement.las output_octree_T1 --voxel 0.015 --max-depth 10 --residual-threshold 0.04 --curvature-threshold 0.08 --angle-threshold 9.0 --min-room-points 5000 --filter-outlier --improve-boundaries --measure-grid-size 0.05
```

**Processing Time**: ~2-6 hours for 2.7M points  
**Expected Results**: High accuracy, detailed segmentation

### Test 2.2: Fast Configuration (Lower Depth)
**Purpose**: Faster processing with reduced octree depth

```powershell
py -3.11 RegionGrowing\complete_room_analysis_octree.py RegionGrowing\appartement.las output_octree_T2 `
  --voxel 0.02 `
  --max-depth 9 `
  --residual-threshold 0.05 `
  --curvature-threshold 0.1 `
  --angle-threshold 10.0 `
  --min-room-points 6000 `
  --measure-grid-size 0.055
```

**Processing Time**: ~1-3 hours for 2.7M points  
**Expected Results**: Good accuracy, faster processing

### Test 2.3: Maximum Accuracy (High Depth)
**Purpose**: Maximum accuracy with highest octree depth (very slow)

```powershell
py -3.11 RegionGrowing\complete_room_analysis_octree.py RegionGrowing\appartement.las output_octree_T3 `
  --voxel 0.01 `
  --max-depth 11 `
  --residual-threshold 0.03 `
  --curvature-threshold 0.06 `
  --angle-threshold 8.0 `
  --min-room-points 4000 `
  --filter-outliers `
  --improve-boundaries `
  --measure-grid-size 0.04
```

**Processing Time**: ~4-12+ hours for 2.7M points (Phase B can take many hours)  
**Expected Results**: Highest accuracy, finest detail  
**Warning**: Phase B refinement is very slow with high depth and many segments

---

## 4. Hybrid Script (complete_room_analysis_hybrid.py)

**Focus: Balanced Speed and Accuracy**

### Test 4.1: Fast Hybrid (Recommended for Most Users)
**Purpose**: Optimal speed/accuracy balance for production use

```powershell
py -3.11 RegionGrowing\complete_room_analysis_hybrid.py RegionGrowing\appartement.las output_hybrid_T1 `
  --voxel 0.015 `
  --octree-max-depth 8 `
  --gap-search-radius 0.3 `
  --ransac-distance-threshold 0.05 `
  --min-room-points 5000 `
  --measure-grid-size 0.05
```

**Processing Time**: ~1-3 hours for 2.7M points  
**Expected Results**: Good accuracy, much faster than octree-only script

### Test 4.2: Balanced Hybrid (Default Configuration)
**Purpose**: Standard hybrid configuration with all phases enabled

```powershell
py -3.11 complete_room_analysis_hybrid.py appartement.las output_hybrid_T2 --voxel 0.015 --octree-max-depth 9 --gap-search-radius 0.25 --ransac-distance-threshold 0.04 --normal-threshold 0.92 --min-room-points 5000 --measure-grid-size 0.05
```

**Processing Time**: ~2-4 hours for 2.7M points  
**Expected Results**: High accuracy with gap filling and RANSAC refinement

### Test 4.3: Maximum Accuracy Hybrid (All Features)
**Purpose**: Maximum accuracy with strict parameters

```powershell
py -3.11 RegionGrowing\complete_room_analysis_hybrid.py RegionGrowing\appartement.las output_hybrid_T3 `
  --voxel 0.01 `
  --octree-max-depth 9 `
  --gap-search-radius 0.25 `
  --ransac-distance-threshold 0.03 `
  --normal-threshold 0.92 `
  --distance-threshold 0.08 `
  --min-room-points 4000 `
  --measure-grid-size 0.04
```

**Processing Time**: ~3-6 hours for 2.7M points  
**Expected Results**: Maximum accuracy with all refinement phases

### Test 4.4: GPU-Accelerated Hybrid (Fast) 🚀
**Purpose**: GPU-accelerated hybrid for maximum speed

```powershell
py -3.11 complete_room_analysis_hybrid_gpu.py appartement.las output_hybrid_gpu_T1 --use-gpu --voxel 0.015 --octree-max-depth 8 --gap-search-radius 0.3 --ransac-distance-threshold 0.05 --min-room-points 5000 --measure-grid-size 0.05
```

**Processing Time**: ~30-90 minutes for 2.7M points (2-3x faster than CPU)  
**GPU Memory**: ~3-5 GB VRAM usage  
**Speedup**: Phase 2 gap filling and Phase 3 RANSAC accelerated

### Test 4.5: GPU-Accelerated Hybrid (Maximum Accuracy) 🚀
**Purpose**: GPU-accelerated with strict parameters

```powershell
py -3.11 complete_room_analysis_hybrid_gpu.py appartement.las output_hybrid_gpu_T2 --use-gpu --voxel 0.01 --octree-max-depth 9 --gap-search-radius 0.25 --ransac-distance-threshold 0.03 --normal-threshold 0.92 --min-room-points 4000 --measure-grid-size 0.04
```

**Processing Time**: ~1-2 hours for 2.7M points (3-4x faster than CPU)  
**GPU Memory**: ~4-6 GB VRAM usage  
**Best for**: RTX 4080/4090 with high accuracy requirements

---

## 5. Advanced Script (complete_room_analysis_advanced.py)

**Focus: Full Features with Room Classification**

### Test 5.1: Standard Advanced (Recommended)
**Purpose**: Full feature set with attachment detection, door frames, and room classification

```powershell
py -3.11 complete_room_analysis_advanced.py appartement.las output_advanced_T1 --voxel 0.015 --octree-max-depth 8 --attachment-threshold 0.08 --door-height-min 0.7 --door-height-max 2.1 --door-width-min 0.6 --door-width-max 1.2 --gap-search-radius 0.3 --ransac-distance-threshold 0.05 --min-room-points 5000 --measure-grid-size 0.05
```

**Processing Time**: ~2-4 hours for 2.7M points  
**Expected Results**: Full features including room type classification

### Test 4.2: Fast Advanced (Essential Features Only)
**Purpose**: Faster processing with essential features, disabled RANSAC

```powershell
py -3.11 RegionGrowing\complete_room_analysis_advanced.py RegionGrowing\appartement.las output_advanced_T2 `
  --voxel 0.02 `
  --octree-max-depth 7 `
  --attachment-threshold 0.1 `
  --door-height-min 0.7 `
  --door-height-max 2.1 `
  --door-width-min 0.6 `
  --door-width-max 1.2 `
  --no-ransac `
  --min-room-points 6000 `
  --measure-grid-size 0.06
```

**Processing Time**: ~1-2 hours for 2.7M points  
**Expected Results**: Attachment detection and door frames, faster processing

### Test 4.3: Maximum Features Advanced (Production Quality)
**Purpose**: All features enabled with strict parameters for production use

```powershell
py -3.11 RegionGrowing\complete_room_analysis_advanced.py RegionGrowing\appartement.las output_advanced_T3 `
  --voxel 0.015 `
  --octree-max-depth 9 `
  --attachment-threshold 0.06 `
  --normal-parallel-threshold 0.88 `
  --door-height-min 0.7 `
  --door-height-max 2.1 `
  --door-width-min 0.6 `
  --door-width-max 1.2 `
  --gap-search-radius 0.25 `
  --ransac-distance-threshold 0.04 `
  --normal-threshold 0.92 `
  --min-room-points 5000 `
  --filter-outliers `
  --improve-boundaries `
  --measure-grid-size 0.04
```

**Processing Time**: ~3-6 hours for 2.7M points  
**Expected Results**: Production-quality results with all features and maximum accuracy

### Test 5.4: GPU-Accelerated Advanced (Fast) 🚀🔥
**Purpose**: GPU-accelerated advanced with MASSIVE attachment detection speedup

```powershell
py -3.11 complete_room_analysis_advanced_gpu.py appartement.las output_advanced_gpu_T1 --use-gpu --voxel 0.015 --octree-max-depth 8 --attachment-threshold 0.08 --door-height-min 0.7 --door-height-max 2.1 --door-width-min 0.6 --door-width-max 1.2 --gap-search-radius 0.3 --ransac-distance-threshold 0.05 --min-room-points 5000 --measure-grid-size 0.05
```

**Processing Time**: ~20-40 minutes for 2.7M points (5-10x faster than CPU!)  
**GPU Memory**: ~4-6 GB VRAM usage  
**CRITICAL SPEEDUP**: Attachment detection (150k-600k KDTree queries) accelerated 10-50x  
**Best for**: Production use with RTX 4080 or better

### Test 5.5: GPU-Accelerated Advanced (Maximum Features) 🚀🔥
**Purpose**: All features with GPU acceleration and strict parameters

```powershell
py -3.11 complete_room_analysis_advanced_gpu.py appartement.las output_advanced_gpu_T2 --use-gpu --voxel 0.015 --octree-max-depth 9 --attachment-threshold 0.06 --normal-parallel-threshold 0.88 --door-height-min 0.7 --door-height-max 2.1 --door-width-min 0.6 --door-width-max 1.2 --gap-search-radius 0.25 --ransac-distance-threshold 0.04 --normal-threshold 0.92 --min-room-points 5000 --filter-outliers --improve-boundaries --measure-grid-size 0.04
```

**Processing Time**: ~30-60 minutes for 2.7M points (4-8x faster than CPU)  
**GPU Memory**: ~5-8 GB VRAM usage  
**Expected Results**: Production-quality with room classification - significantly faster  
**Note**: Attachment detection with 200k unclassified points: ~2 minutes (GPU) vs ~20+ minutes (CPU)

---

## Test Comparison Guide

### Noise Threshold Impact (Point-Based Script)

Compare the three point-based tests to understand noise threshold impact:

| Test | --min-region-size | Expected Issue |
|------|-------------------|----------------|
| 1.1 | 500 (Too Low) | Noise clusters included → Distorted areas |
| 1.2 | 1500 (Optimal) | Balanced noise filtering → Accurate areas |
| 1.3 | 3000 (High) | Over-filtering → May remove valid features |

**Key Metric**: Compare `room_areas.csv` across tests to see area calculation differences.

### Speed Comparison

| Script | Test Configuration | Approx. Time (2.7M points) | GPU Version Time |
|--------|-------------------|----------------------------|------------------|
| Point-Based | Test 1.2 (Optimal) | 30-90 minutes | 5-15 min (GPU script) |
| GPU Point-Based | Test 2.2 (GPU Optimal) | - | 5-15 minutes |
| Octree | Test 3.1 (Balanced) | 2-6 hours | N/A |
| Hybrid | Test 4.1 (Fast) | 1-3 hours | 30-90 min (GPU) |
| Hybrid GPU | Test 4.4 (GPU Fast) | - | 30-90 minutes |
| Advanced | Test 5.1 (Standard) | 2-4 hours | 20-40 min (GPU) |
| Advanced GPU | Test 5.4 (GPU Fast) | - | 20-40 minutes |

### Accuracy vs Speed Trade-off

| Priority | Recommended Script & Test |
|----------|---------------------------|
| **Maximum Accuracy** | Octree Test 3.3 or Advanced Test 5.3 |
| **Balanced** | Hybrid Test 4.1 or Advanced Test 5.1 |
| **Speed** | GPU Point-Based Test 2.2 or GPU Hybrid Test 4.4 |
| **Maximum Speed + Accuracy** | GPU Advanced Test 5.4 (🔥 BEST) |
| **Noise Analysis** | Point-Based Tests 1.1-1.3 (compare areas) |

### GPU Acceleration Impact 🚀

| Script | CPU Time | GPU Time | Speedup | Primary Bottleneck Accelerated |
|--------|----------|----------|---------|-------------------------------|
| **Point-Based GPU** | 30-90 min | 5-15 min | 2-6x | KDTree queries, region growing |
| **Hybrid GPU** | 1-3 hours | 30-90 min | 2-3x | Phase 2 gap filling, Phase 3 RANSAC |
| **Advanced GPU** 🔥 | 2-4 hours | 20-40 min | **5-10x** | **Attachment detection (150k-600k queries!)** |

**Why Advanced GPU is Fastest:**
- Original attachment detection: 3 sequential KDTree queries × 50k-200k points = **150k-600k queries!**
- GPU-accelerated: All queries in 1-2 GPU batches
- Result: **Attachment detection: 2 min (GPU) vs 20+ min (CPU)**

---

## Usage Notes

1. **Replace file path**: Change `RegionGrowing\appartement.las` to your actual input file
2. **Output folders**: Will be created automatically (`output_*_T#`)
3. **Processing time**: Times are estimates for 2.7M points on Intel i7 / 16GB RAM
4. **PowerShell**: Commands use backticks (`` ` ``) for line continuation
5. **Memory**: Ensure 32GB RAM available for larger point clouds
6. **Noise threshold**: For point-based script, always compare results across different `--min-region-size` values

---

## Expected Output Files (Per Test)

Each test generates:
- `classified.las` - Classified point cloud
- `segmented.las` - Segmented rooms
- `room_map.png` - 2D visualization
- `room_areas.csv` - Area measurements (with room types for Advanced script)
- `room_polygons.json` - Polygon coordinates (with types for Advanced script)

---

## Troubleshooting Test Results

### Issue: Areas are inflated/too large
**Cause**: Noise clusters not filtered (too low `--min-region-size`)  
**Solution**: Increase `--min-region-size` (compare Test 1.1 vs 1.2)

### Issue: Missing small rooms
**Cause**: Over-filtering (too high `--min-region-size` or `--min-room-points`)  
**Solution**: Decrease `--min-region-size` or `--min-room-points`

### Issue: Processing too slow
**Solution**: Use Hybrid script (Test 3.1) instead of Octree script

### Issue: Need room classification
**Solution**: Use Advanced script (Tests 4.1-4.3)
