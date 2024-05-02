import open3d as o3d
import cv2 as cv2
import numpy as np
import sys
from scipy.spatial import procrustes

DATA_DIR = "daten/240405_Messungen_LL/"

FILES = [
    DATA_DIR + "FDM2_SP2_1.ply",
    DATA_DIR + "FDM2_SP2_2.ply",
]

def showPointClouds(pcds):
    o3d.visualization.draw_geometries(pcds,
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])

def loadPointClouds(vis = False) :
    pcds = [None] * len(FILES)
    pcds[0] = o3d.io.read_point_cloud(FILES[0])
    print("Loaded file: ", FILES[0] + " with " +str(len(pcds[0].points)) + " points.")
    pcds[1] = o3d.io.read_point_cloud(FILES[1])
    print("Loaded file: ", FILES[1] + " with " +str(len(pcds[1].points)) + " points.")
    return pcds

def translatePcd2Zero(pcds):
    for pcd in pcds:
        aabb = pcd.get_axis_aligned_bounding_box()
        minPoint = aabb.get_min_bound()
        minPointX = 0 - minPoint[0]
        minPointY = 0 - minPoint[1]
        minPointZ = 0 - minPoint[2]
        pcd.translate(np.array([minPointX, minPointY, minPointZ]))
        print(pcd.get_axis_aligned_bounding_box())

# source is moved to end of target (x axis)
def alignPointCloudsToEachOther(source, target):
    #align target point cloud under source
    aabb = target.get_axis_aligned_bounding_box()
    maxPoints = aabb.get_max_bound()
    minPointX = maxPoints[0]
    minPointZ = maxPoints[2]
    print("x cutting axis: ", minPointX)
    lines = drawBox(minPointX, minPointZ)
    source.translate(np.array([minPointX, 0, 0]))
    showPointClouds([lines, source, target])

def drawBox(xAxis, zAxis):
    print("Let's draw a box using o3d.geometry.LineSet.")
    points = [
        [xAxis, 0, zAxis],
        [xAxis, 100, zAxis],
    ]
    lines = [
        [0, 1],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
                                                height=1.0,
                                                depth=1.0)
    return line_set

def getDistanceSum(source, target):
    dists = source.compute_point_cloud_distance(target)
    dists = np.asarray(dists)
    return np.sum(dists)

def alignXAxis(source, target):
    sourcePoints = np.asarray(source.points)
    print(sourcePoints[0][0])

def movePointCloudX(source, target, by):
    oldSum = getDistanceSum(source, target)
    source.translate(np.array([by, 0, 0]))
    newSum = getDistanceSum(source, target)
    change = newSum - oldSum
    #print("Distance changed by: ", change)
    return newSum
    
def alignPointClouds(target, source):
    alignPointCloudsToEachOther(source, target)
    alignXAxis(source, target)

    
    showPointClouds([source, target])


if __name__ == "__main__":
    print("Starting easyStitch...")

    # Align point_cloud2 to point_cloud1 along the x-axis
    pcds = loadPointClouds()
    translatePcd2Zero(pcds)
    aligned_point_cloud2 = alignPointClouds(pcds[0], pcds[1])
    
    showPointClouds(aligned_point_cloud2)
    print("Original Point Cloud 1:\n", pcds[0])
    print("Original Point Cloud 2:\n", pcds[1])
    print("Aligned Point Cloud 2:\n", aligned_point_cloud2)

    sys.exit()
    pcds = loadPointClouds()
    translatePcd2Zero(pcds)
    alignPointClouds(pcds[0], pcds[1])

    #showPointClouds(pcds)