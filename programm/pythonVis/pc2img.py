import open3d as o3d
import numpy as np
import cv2 as cv
import collections
from dataclasses import dataclass

class imageStructure:
    x: int
    y: int
    depth: float
    color: int

DATA_DIR = "daten/240405_Messungen_LL/"

FILES = {
    DATA_DIR + "AM_SP0_1.ply",
    #DATA_DIR + "FDM_2.ply",
}

def loadPointClouds(vis = False) :
    pcds = []
    for file in FILES:
        pcd = o3d.io.read_point_cloud(file)
        pcds.append(pcd)
        print("Loaded file: ", file + " with " +str(len(pcd.points)) + " points.")
    if(vis):
        visPcds = []
        for pcd in pcds:
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (1, 0, 0)
            visPcds.append(aabb)
            visPcds.append(pcd)
        o3d.visualization.draw_geometries(visPcds,
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    return pcds  

def removeOutliers(pcds):
    print("Statistical oulier removal")
    cleared = []
    for pcd in pcds:
        oldAmount = len(pcd.points)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=2.0)
        newAmount = 0# len(cl.select_by_index(ind).points)
        print("Reduced from: " + str(oldAmount) + " to " + str(newAmount) + " points")
        cleared.append(cl.select_by_index(ind))
    return cleared

def anaylseZ(pcd) :
    MIN_AMOUNT = 5
    z = []
    for point in pcd.points:
        z.append(point[2])
        
    counter = collections.Counter(z)
    mostCommon = counter.most_common(500)
    minFound = 10000000000000
    maxFound =  0
    for key, val in mostCommon:
        if(val > MIN_AMOUNT):
            maxFound = max(key, maxFound)
            minFound = min(key, minFound)
    return maxFound, minFound

def conversion(old_value, old_min, old_max, new_min = 10, new_max = 255):
    return ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min

def pcNeighbours(pcd):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print("Find its 200 nearest neighbors, and paint them blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[5000], 10)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd],
                                  zoom=0.5599,
                                  front=[-0.4958, 0.8229, 0.2773],
                                  lookat=[2.1126, 1.0163, -1.8543],
                                  up=[0.1007, -0.2626, 0.9596])
    
def cropPointCloud(pcd, pointAmount = 100000):
    if(len(pcd.points) < pointAmount):
        print("not enough points in cloud")
        return
    firstPoint = 0
    lastPoint = 900000
    oriented_bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(pcd.points[firstPoint:lastPoint])
    cropped = pcd.crop(oriented_bounding_box)
    pointCloudToImage(cropped)
    return cropped

def pointCloudToImage(pcd, resolution = 1) :
    pcd.translate((0, 0.1, 0))
    maxFound, minFound = anaylseZ(pcd)
    scale = 100
    aabb = pcd.get_axis_aligned_bounding_box()
    size = aabb.get_extent()
    height = int(size[0]*scale +100)
    width = int(size[1]*scale +100)
    depth = int(size[2]*scale +100)
    minX = np.amin(np.asarray(aabb.get_box_points()))
    offsetX  = abs(int(minX*scale))
    print("Dimensions are: ", 
          height, 
          width, 
          depth)
    image = np.zeros((height,width,3), np.uint8)
    i = -1
    amountPoints = len(pcd.points)
    amountPixel = -1
    for point in pcd.points:
        amountPixel += 1
        i += 1
        if(i%resolution != 0 or point[2] < minFound or point[2] > maxFound):
            continue
        x = int(point[0]*scale) + offsetX #negativ offset
        y = int(point[1]*scale)
        r = int(pcd.colors[i][0]*255)
        g = int(pcd.colors[i][1]*255)
        b = int(pcd.colors[i][1]*255)
        brightness = conversion(point[2]*1000, minFound*1000, maxFound*1000)
        image[x, y] = (b, r, g)
        if(i%10000 == 0):
            print("{:.4f}%".format(round((i/amountPoints)*100, 4)), end="\r")
    print("Amount of pixel:" + str(amountPixel))
    small = cv.resize(image, (0,0), fx=0.3, fy=0.3) 
    #cv.imshow('Image from point cloud', small)
    #cv.waitKey(0)
    name = "pcd_" + str(len(pcd.points)) + ".png"
    cv.imwrite(name, image)
    # add wait key. window waits until user presses a key

if __name__ == "__main__":
    pcds = loadPointClouds(vis = False)
    for pcd in pcds:
        print("Converting cloud...")
        pointCloudToImage(pcd, 5)
    #cropPointCloud(pcds[0])
    #pcNeighbours(pcds[0])