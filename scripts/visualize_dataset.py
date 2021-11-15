import sys
import numpy as np
import open3d as o3d


sys.path.insert(1, '../src')

from dataset import DatasetConfig, DatasetBuilder

if __name__ == "__main__":
    points = np.load('lines.npy')
    num_points = points.shape[0]
    lines = np.array(range(num_points)).reshape(-1, 2)

    colors = np.load('colors.npy')
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.6, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([line_set, mesh_frame])