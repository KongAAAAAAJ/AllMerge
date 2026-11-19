"""Drivable road union from the raw sampled lane edges, in the ego frame."""
import numpy as np
import shapely


def road_area(features):
    geometry = np.asarray(features['road_lane_geometry'], dtype=np.float64)
    points = np.asarray(features['road_lane_point_mask'], dtype=bool)
    lanes = np.asarray(features['road_lane_mask'], dtype=bool)
    polygons = []
    for lane in np.flatnonzero(lanes):
        edge = geometry[lane, points[lane]]
        polygons.append(shapely.Polygon(np.concatenate((edge[:, 2:4], edge[::-1, 4:6]))))
    return shapely.union_all(polygons)


def road_clearance(area, xy):
    """Signed distance to the union boundary: positive inside, negative outside."""
    points = shapely.points(np.asarray(xy, dtype=np.float64))
    distance = shapely.distance(points, area.boundary)
    return np.where(shapely.covers(area, points), distance, -distance)
