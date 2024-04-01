
import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("Crankshaft-HD.ply")
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

xzy = np.random.random((100,3))*10
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xzy)
o3d.visualization.draw_geometries([pcd])