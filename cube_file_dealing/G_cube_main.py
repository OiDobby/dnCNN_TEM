import numpy as np
import math
import time
import os
import shutil
import matplotlib.pyplot as plt
from G_cube_util import make_cube_map 

#################################################################################
# This code makes to deal with the gaussian cube format files.                  #
# This code makes lage-scale charge density map from combining cube files       #
# This code operates in ractangular cell only. (11/28/2023)                     #
# Thic code work in .tden.cube, which is output of openMX package.              #
#################################################################################

base_path = os.getcwd()   # Now, current directory
dir_name = 'ori_cube_files/'

tip_pos = 3  # z position of the charge density map in Angstrom unit.

num_cube_matrix = 3   # We make N*N matrix from randomly-choiced cube files.
num_rand_cube = 1  # We make N randomly-choiced cube files for making N images.

sat_level_max = 2.0e-6  # Max value of satulation level. 
                        #We will confirm that this is a reasonable value.

make_cube_map(base_path, dir_name, tip_pos, num_cube_matrix, num_rand_cube, sat_level_max)

print('Program done')
