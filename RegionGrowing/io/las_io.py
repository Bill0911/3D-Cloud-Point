import laspy
import numpy as np

# Load only XYZ
def load_las_points(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    return pts

# Load XYZ + classification
def load_las_with_classification(path):
    las = laspy.read(path)

    if "classification" not in las.point_format.dimension_names:
        raise RuntimeError(f"LAS file {path} has no 'classification' field")

    points = np.vstack((las.x, las.y, las.z)).T
    classification = np.asarray(las.classification, dtype=np.uint8)

    return las, points, classification

# Save downsampled LAS with classification labels
def save_las_with_classification(path, points, labels):
    header = laspy.LasHeader(point_format=3, version="1.4")
    las = laspy.LasData(header)

    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = labels.astype(np.uint8)

    las.write(path)

# Save LAS with room_id + room_class assigned to ceiling points
def save_las_with_room_ids(las, ceiling_mask, room_ids, path):
    # Add extra dimensions if missing
    extra = set(las.point_format.extra_dimension_names)

    if "room_id" not in extra:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="room_id",
                type="u2",
                description="Room ID"
            )
        )

    if "room_class" not in extra:
        las.add_extra_dim(
            laspy.ExtraBytesParams(
                name="room_class",
                type="u2",
                description="Room Class (700 + room_id)"
            )
        )

    # Init arrays
    las.room_id = np.zeros(len(las.points), dtype=np.uint16)
    las.room_class = np.zeros(len(las.points), dtype=np.uint16)

    # Assign only to ceiling points
    las.room_id[ceiling_mask] = room_ids
    las.room_class[ceiling_mask] = np.where(room_ids > 0, 700 + room_ids, 0)

    las.write(path)
