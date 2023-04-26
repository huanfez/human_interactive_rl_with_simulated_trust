#! /usr/bin/env python

import rospkg
import numpy as np
import tifffile as tf

rospack = rospkg.RosPack()
rospack.list()  # list all packages, equivalent to rospack list
pkg_dir = rospack.get_path('trust_rl_mrs')  # get the file path for rospy_tutorials
map_dir = pkg_dir + '/src/map'  # map directory
data_dir = pkg_dir + '/src/data'  # data directory

#####################################################################################
# dem_file = pkg_dir + '/src/map/yazoo_500m_dem.tif'
# dsm_file = pkg_dir + '/src/map/yazoo_500m_dsm.tif'
dem_img = tf.imread(map_dir + '/yazoo_500m_dem.tif')
dsm_img = tf.imread(map_dir + '/yazoo_500m_dsm.tif')

# traversability_file = pkg_dir + '/src/map/slope_yazoo_500m.tif'
# visibility_file = pkg_dir + '/src/map/Minus_yazoo_500m.tif'
traversability_img = tf.imread(map_dir + '/slope_yazoo_500m.tif')
visibility_img = tf.imread(map_dir + '/Minus_yazoo_500m.tif')
trust_temp_tif = data_dir + '/trust_temp.tif'

gis_traversability = np.copy(np.asarray(traversability_img))
gis_visibility = np.copy(np.asarray(visibility_img))

# normalize data
r1_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 1.8)),
                                      (1 + np.exp(gis_traversability * 100.0 - 1.8)))
r1_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)), (1 + np.exp(gis_visibility - 1.5)))

r2_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 1.8)),
                                      (1 + np.exp(gis_traversability * 100.0 - 1.8)))
r2_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)), (1 + np.exp(gis_visibility - 1.5)))

r3_dynamic_traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 1.8)),
                                      (1 + np.exp(gis_traversability * 100.0 - 1.8)))
r3_dynamic_visibility = np.divide((1 - np.exp(gis_visibility - 1.5)), (1 + np.exp(gis_visibility - 1.5)))

#####################################################################################
# define grid environment parameters
env_width, env_height = 500.0, 500.0  # meters
# cell_width, cell_height = 25, 25  # pixels
img_width, img_height = traversability_img.shape  # pixels
# grid_width, grid_height = int(img_width / cell_width), int(img_height / cell_height)  # discrete environment size

# global iteration
# iteration = 0

######################################################################################

# obs_list = [(14, 0), (11, 3), (0, 7), (14, 7), (2, 8), (3, 8), (4, 8), (5, 8), (7, 8), (8, 8), (17, 8),
#             (1, 9), (4, 10), (6, 10), (7, 10), (9, 10), (12, 11), (1, 12), (3, 12), (5, 12),
#             (6, 12), (8, 12), (10, 12), (14, 12), (16, 12), (0, 13), (12, 13), (0, 14),
#             (4, 14), (5, 14), (10, 14), (0, 15), (2, 15), (0, 16), (6, 16), (7, 16), (0, 17),
#             (3, 17), (4, 17), (17, 17), (18, 18), (12, 19), (13, 19), (14, 19), (18, 19)]
######################################################################################


# Common used functions:
def imgPos2envPos(imgPos, img_width=img_width, img_height=img_height, env_width=env_width, env_height=env_height):
    transMat = np.array([[img_width / 2.0], [img_height / 2.0]])
    scalMat = np.array([[env_width / img_width, 0], [0, -env_height / img_height]])

    array1 = imgPos - transMat
    envPos = np.dot(scalMat, array1)
    return envPos


def envPos2imgPos(envPos, img_width=img_width, img_height=img_height, env_width=env_width, env_height=env_height):
    transMat = np.array([[img_width / 2.0], [img_height / 2.0]])
    scalMat = np.array([[env_width / img_width, 0], [0, -env_height / img_height]])

    array1 = np.dot(np.linalg.inv(scalMat), np.array([envPos]).T)
    imgPos = array1 + transMat
    return imgPos.flatten().astype(int)


def imgpath2traj(localPath):
    localTraj = [imgPos2envPos(np.array([[imgPos[0]], [imgPos[1]]])) for imgPos in localPath]
    return localTraj
