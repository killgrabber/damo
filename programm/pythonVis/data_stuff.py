import math
import sys

import pc2img
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import copy


def do_other_stuff():
    pc_file_fdm = 'C:/Users/nikla/Documents/Bachleorarbeit/damo/daten/240405_Messungen_LL/FDM_1.ply'
    pc_file_m = 'C:/Users/nikla/Documents/Bachleorarbeit/damo/daten/AM1/AM_SP0_1.ply'
    progress = [""]
    result = []
    #pc2img.convert_point_cloud([pc_file], progress, result)

    print(f"Loading pointcloud...")
    pcd = o3d.io.read_point_cloud(pc_file_m)
    np_points = np.asarray(pcd.points)
    z_points = np_points[:, 2]
    print(f"Generating histogram...")
    font = {'size': 30}
    plt.rc('font', **font)
    max_ma = np.amax(z_points)
    min_ma = np.amin(z_points)

    print(f"Loading pointcloud...")
    pcd = o3d.io.read_point_cloud(pc_file_fdm)
    np_points_fdm = np.asarray(pcd.points)
    z_points_fdm = np_points_fdm[:, 2]
    max_fdm = np.amax(z_points_fdm)
    min_fdm = np.amin(z_points_fdm)

    all_z = [z_points, z_points_fdm]

    max_ma = max(max_ma, max_fdm)
    min_ma = min(min_ma, min_fdm)
    print(f"Range: {min_ma}, {max_ma}")
    plt.xlim(xmin=math.floor(min_ma), xmax=math.ceil(max_ma))
    #plt.ylim(ymax=0.1)
    plt.hist(all_z, density=True, bins=500, log=False)  # density=False would make counts
    plt.ylabel('Percentage of heights in each group')
    plt.xlabel('Height')
    plt.legend(['Metal', 'FDM'])
    plt.show()


def get_data_from_file(filename):
    file = open(filename, "r")
    x_val = []
    both_val = []
    for line in file:
        splitted = line.replace(" ", "").split(',').copy()
        x_val.append(int(splitted[0]))
        both_val.append(int(splitted[1]))
    file.close()

    new_x = []
    new_y = []
    for i in range(len(both_val)):
        if not both_val[i] == 0:
            new_x.append(i)
            new_y.append((both_val[i]))

    # create rolling average
    step_size = 10
    rolling_x = []
    rolling_y = []
    average_y = np.mean(new_y)
    for i in range(0, len(new_y), step_size):
        mean_x = np.median(new_x[i:i + step_size])
        mean_y = np.median(new_y[i:i + step_size])
        #if abs(mean_y) < abs(average_y * 10):
        rolling_x.append(mean_x)
        rolling_y.append(mean_y)

    # Adjust data
    mean_y = np.median(rolling_y)
    print(f"Mean y is: {mean_y}")
    for i in range(len(rolling_y)):
        rolling_y[i] = rolling_y[i] - mean_y

    return new_x, new_y, rolling_x, rolling_y


def do_stuff_with_deformation_data(filenames: []):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', ]
    plt.rc('font', size=50)
    for index in range(len(filenames)):
        new_x, new_y, rolling_x, rolling_y = get_data_from_file(filenames[index])
        filename = filenames[index].split('/')[-1].replace("_stitched", "").split('.')[0]
        #all_rolling_x.append(rolling_x)
        #all_rolling_y.append(rolling_y)
        plt.plot(rolling_x, rolling_y, label=filename, color=colors[(index + 2) % len(filenames)],
                 linewidth=1)
        #plt.scatter(x_val, y1_val, label="top_distance", s=1)
        #plt.scatter(x_val, y2_val, label="bot_distance", s=1)
        #plt.scatter(new_x, new_y, s=1)
    plt.grid()
    plt.axhline(0, color='black')
    plt.legend()
    plt.ylabel('Deformation in Pixel')
    plt.xlabel('Pixel - X-Achse')
    plt.ylim((-100, 100))
    plt.show()
