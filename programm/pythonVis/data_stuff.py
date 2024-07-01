import math

import pc2img
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

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
