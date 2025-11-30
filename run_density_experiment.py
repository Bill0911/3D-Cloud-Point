import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
import re
import shutil

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# The LAS file to test
INPUT_FILE = r"C:\Users\freetii\Downloads\emit-it_appartement_sor_noise_filtered-laz_2025-11-11_1332\appartement_SOR_NoiseFiltered_5mm.las"

# The script to run (Assuming regionGrowing.py is your main pipeline)
PIPELINE_SCRIPT = "Algorithm/regionGrowing.py"

# The voxel sizes to test (in meters)
VOXEL_SIZES = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1]

# Base folder for experiment results
BASE_OUTPUT_DIR = "experiment_results"

# ==============================================================================
# EXPERIMENT LOOP
# ==============================================================================

results = []

print(f"--- Starting Density Experiment ---")
print(f"Input File: {INPUT_FILE}")
print(f"Voxel Sizes: {VOXEL_SIZES}\n")

for voxel_size in VOXEL_SIZES:
    print(f"Running for Voxel Size: {voxel_size}...")
    
    # 1. Setup Output Folder (Use Absolute Path)
    run_output_folder = os.path.abspath(os.path.join(BASE_OUTPUT_DIR, f"voxel_{voxel_size}"))
    
    if os.path.exists(run_output_folder):
        shutil.rmtree(run_output_folder) # Clean previous run
    os.makedirs(run_output_folder, exist_ok=True)

    # 2. Construct Command
    # Matches: python regionGrowing.py <input> <output> --voxel <size>
    cmd = [
        "python", 
        PIPELINE_SCRIPT, 
        INPUT_FILE, 
        run_output_folder,
        "--voxel", str(voxel_size)
    ]

    # 3. Execute and Time
    start_time = time.time()
    try:
        # Capture stdout to find point counts if printed
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_log = process.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running voxel {voxel_size}: {e}")
        print(e.stderr)
        continue
    
    duration = time.time() - start_time

    # 4. Parse Results
    
    # A. Try to find point count in logs (Adjust regex based on your script's print statements)
    # Looking for something like "Decimated to 12345 points" or "Points: 12345"
    point_count = 0
    match = re.search(r"Decimated to (\d+) points", output_log)
    if match:
        point_count = int(match.group(1))
    else:
        # Fallback: try to find just a large number followed by "points"
        match = re.search(r"(\d+) points", output_log)
        if match:
            point_count = int(match.group(1))

    # B. Read the Area CSV
    total_area = 0
    csv_path = os.path.join(run_output_folder, "room_areas.csv")
    if os.path.exists(csv_path):
        try:
            df_area = pd.read_csv(csv_path)
            # Summing all rooms to get total apartment area
            total_area = df_area['area_m2'].sum()
        except Exception as e:
            print(f"  Failed to read CSV: {e}")
    else:
        print("  Warning: room_areas.csv not found.")

    print(f"  -> Time: {duration:.2f}s | Points: {point_count} | Area: {total_area:.2f} m2")

    # 5. Store Data
    results.append({
        "voxel_size": voxel_size,
        "point_count": point_count,
        "total_area_m2": total_area,
        "computation_time_s": duration
    })

# ==============================================================================
# AGGREGATION & VISUALIZATION
# ==============================================================================

# Create DataFrame
df_results = pd.DataFrame(results)

# Save Summary CSV
summary_path = os.path.join(BASE_OUTPUT_DIR, "density_experiment_summary.csv")
df_results.to_csv(summary_path, index=False)
print(f"\nExperiment finished. Summary saved to: {summary_path}")

# Plotting
if not df_results.empty:
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Voxel Size (m)')
    ax1.set_ylabel('Total Calculated Area (m2)', color=color)
    ax1.plot(df_results['voxel_size'], df_results['total_area_m2'], marker='o', color=color, label='Area')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a second y-axis for computation time
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Computation Time (s)', color=color)  
    ax2.plot(df_results['voxel_size'], df_results['computation_time_s'], marker='x', linestyle='--', color=color, label='Time')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Impact of Voxel Density on Area Calculation & Performance')
    fig.tight_layout()
    
    plot_path = os.path.join(BASE_OUTPUT_DIR, "density_impact_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()