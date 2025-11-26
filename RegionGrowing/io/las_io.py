import laspy
import numpy as np

def load_las_points(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    return pts, las.header


def load_las_with_classification(path):
    las = laspy.read(path)
    pts = np.vstack((las.x, las.y, las.z)).T
    cls = np.asarray(las.classification, dtype=np.uint8)
    return las, pts, cls


def save_las_with_classification(path, points, labels, src_header):
    header = laspy.LasHeader(
        point_format=src_header.point_format.id,
        version=src_header.version
    )
    header.scales = src_header.scales
    header.offsets = src_header.offsets

    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.classification = labels.astype(np.uint8)
    las.write(path)


def save_las_with_room_ids(las, room_ids, output_path):
    if "room_id" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="room_id", type="u2", description="Room ID"
        ))

    if "room_class" not in las.point_format.extra_dimension_names:
        las.add_extra_dim(laspy.ExtraBytesParams(
            name="room_class", type="u2", description="Room Class"
        ))

    las.room_id = room_ids.astype(np.uint16)
    las.room_class = np.where(
        room_ids > 0,
        700 + room_ids,
        0
    ).astype(np.uint16)

    las.write(output_path)
