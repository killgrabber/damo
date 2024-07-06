import sys

import open3d as o3d
import pc2img
import numpy as np
import cv2


def convert_stl(filepath: str):
    mesh = o3d.io.read_triangle_mesh(filepath)
    print(f"Loading stl done")
    pcd = mesh.sample_points_poisson_disk(number_of_points=100000, init_factor=5)
    pc2img.show_pointcloud(pcd)
    print(f"Sampling pointcloud done")
    stl_points = np.asarray(pcd.points)
    max_h, min_h = pc2img.anaylseZ(stl_points)
    image = pc2img.convert_chunk(stl_points, max_h, min_h)
    print(f"Converting to image done, size: {image.shape}")
    small_image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    cv2.imshow("stl_viewer", small_image)
    cv2.imwrite("test.png", image)
    cv2.waitKey(0)
