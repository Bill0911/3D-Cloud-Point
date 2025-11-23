import laspy
import numpy as np

# Load raw XYZ points from a LAS/LAZ file
def load_las_points(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    return pts

# Save a LAS with classification for downsampled points
def save_las_with_classification(path, points, labels):
    # Create minimal LAS header (same point format 3 -> XYZ + classification)
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = labels.astype(np.uint8)

    las.write(path)
