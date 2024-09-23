import numpy as np
import math
import time
import os
import shutil
import random
import matplotlib.pyplot as plt
#import tensorflow as tf

########################################################################################
# This code makes to deal with the gaussian cube format files.                         #
# This code makes lage-scale charge density map from combining cube files              #
# This code operates in ractangular cell only. (11/28/2023)                            #
# This code work in .tden.cube, which is output of openMX package.                     #
# This code plot charge density map, which are randomly extended supercell sutructure. #
########################################################################################

def make_cube_map(base_path, dir_name, tip_pos, num_cube_matrix, num_rand_cube, sat_level_max):
    file_dir_path = os.path.join(base_path, dir_name)
    file_list = os.listdir(file_dir_path)

    print('-------------------------------------------------------------------')
    print('Searching files.')

    ori_file_num = len(file_list)

    file_list_name = file_list.copy()
    for i in range(ori_file_num):
        file_list_name[i] = file_list_name[i].replace('.tden', '') 
        file_list_name[i] = file_list_name[i].replace('.cube', '')

    N_grid = []
    N_atom_info = []
    N_map_data = []
    for x in range(ori_file_num):
        file_path = os.path.join(file_dir_path, file_list[x])
        
        print('-------------------------------------------------------------------')
        print('Reading files. File:', x)

        file_ori = open(file_path, 'r')

        for i in range(0,2):
            line = file_ori.readline()

        atom_list = []
        tmp1 = file_ori.readline()
        tmp2 = ' '.join(tmp1.split())
        atom_list = tmp2.split(' ')

        atom_num = int(atom_list[0])
        
        cell_grid = []
        for i in range(0,3):
            a = file_ori.readline().rstrip('\n')
            b = ' '.join(a.split())
            c = b.split(' ')
            cell_grid.append(c)

        nx_grid = int(cell_grid[0][0])
        ny_grid = int(cell_grid[1][0])
        nz_grid = int(cell_grid[2][0])

        tmp_grid = [nx_grid, ny_grid, nz_grid]
        N_grid.append(tmp_grid)

        z_lattice = nz_grid*float(cell_grid[2][3]) * 0.529
        
        atom_info = []
        for i in range(atom_num):
            a = file_ori.readline().rstrip('\n')
            b = ' '.join(a.split())
            c = b.split(' ')
            atom_info.append(c)
        N_atom_info.append(atom_info)

        z_coor_list = []
        for i in range(atom_num):
            z_coor_list.append(atom_info[i][4])

        z_coor_min = min(z_coor_list)
        z_coor_min_frac = (float(z_coor_min)-float(atom_list[3])) / (nz_grid*float(cell_grid[2][3]))
        map_pos = math.ceil((z_coor_min_frac + tip_pos/z_lattice)*nz_grid)

        tmp1_vol_data = []
        while True:
            a = file_ori.readline().rstrip('\n')
            b = " ".join(a.split())
            c = b.split(' ')
            if not a: break
            tmp1_vol_data.append(c)

        file_ori.close()

        print('Completely read file:', x)
        print('Sorting file:', x)

        tmp0_vol_data = []
        for i in range(len(tmp1_vol_data)):
            for j in range(0,6):
                tmp0_vol_data.append(tmp1_vol_data[i][j])
        tmp1_vol_data = []

        print('Making 2D map in file:', x)

        tmp0_map_data = []
        tmp1_map_data = []
        for i in range(nx_grid):
            for j in range(ny_grid):
                tmp1_map_data.append(tmp0_vol_data[ny_grid*nz_grid*i+nz_grid*j+map_pos-1])
            tmp0_map_data.append(tmp1_map_data)
            tmp1_map_data = []
        tmp0_vol_data = []

        map_data1 = []
        for i in range(nx_grid):
            map_data2 = []
            for j in range(ny_grid):
                map_data2.append(tmp0_map_data[i][j])
            map_data1.append(map_data2)
        N_map_data.append(map_data1)

    print('-------------------------------------------------------------------')
    print('Read all files.')
    print('Making randomly extended charge density.')

    num_domain = num_cube_matrix*num_cube_matrix

    fig_name_list = [i+1 for i in range(num_rand_cube)]
    fig_name_list = list(map(str, fig_name_list))
    np_name_list = [i+1 for i in range(num_rand_cube)]
    np_name_list = list(map(str, np_name_list))
    for i in range(num_rand_cube):
        fig_name_list[i] = 'Rand' + fig_name_list[i] + '.png'
        np_name_list[i] = 'Rand' + np_name_list[i] + '.npy'

    for x in range(num_rand_cube):
        fig_path = os.path.join(base_path, fig_name_list[x])
        np_path = os.path.join(base_path, np_name_list[x])

        domain_data_list = []
        for i in range(num_domain):
            rand_num = random.randint(0,(len(N_map_data)-1))
            domain_data_list.append(N_map_data[rand_num])

        tmp_data = []
        domain_data = []
        for k in range(num_cube_matrix):
            for j in range(nx_grid): 
                tmp_data = []
                for i in range(num_cube_matrix): 
                    tmp_data.extend(domain_data_list[num_cube_matrix*k+i][j])
                domain_data.append(tmp_data)
        tmp_data = []

        #print('nx grid', nx_grid, 'ny grid', ny_grid)
        #print(len(domain_data), len(domain_data[0]))
        #print(domain_data[223][0], domain_data[223][224])

        for i in range(nx_grid*num_cube_matrix):
            for j in range(ny_grid*num_cube_matrix):
                domain_data[i][j] = float(domain_data[i][j])

        max_level = max(map(max, domain_data))

        if x == 0:
            print('We confirmed saturation level.')
            print('your input:', sat_level_max, 'confirmed:', max_level)
        
        if sat_level_max/max_level >= 1.5:
            sat_level_max = max_level
            if x == 0:
                print('The sat_level_max is too large. We set sat_level_max by the confirmed level.')
        elif sat_level_max/max_level <= 0.5:
            sat_level_max = max_level
            if x == 0:
                print('The sat_level_max is too small. We set sat_level_max by the confirmed level.')
            else :
                sat_level_max = sat_level_max
                if x==0:
                    print('sat_level_max is proper. We set sat_level_max by your input value.')

        map_data0 = np.array(domain_data)
        map_data1 = np.transpose(map_data0)

        for i in range (ny_grid*num_cube_matrix):
            for j in range(nx_grid*num_cube_matrix):
                    map_data1[i][j] = map_data1[i][j]/sat_level_max

        for i in range (ny_grid*num_cube_matrix):
            for j in range(nx_grid*num_cube_matrix):
                if map_data1[i][j] >= 1:
                    map_data1[i][j] = 1
                elif map_data1[i][j] < 0:
                    map_data1[i][j] = 0
                else :
                    map_data1[i][j] = map_data1[i][j]

        print('-------------------------------------------------------------------')
        print('Saving numpy file:', np_name_list[x])
        
        np.save(np_path, map_data1) 

        print('-------------------------------------------------------------------')
        print('Plotting file:', fig_name_list[x])

        fig = plt.imshow(map_data1)
        plt.axis('off'), plt.xticks([]), plt.yticks([])
        plt.tight_layout()
        plt.gray()
        plt.savefig(fig_path, bbox_inches = 'tight', pad_inches = 0, dpi = 300)

        domain_data_list = []
        domain_data = []
            
    print('===================================================================')
    print('We plotted all files.')
