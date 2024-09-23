import numpy as np
import math
import time
import os
import shutil
import matplotlib.pyplot as plt

#################################################################################
# This code makes the rearranged atomic configuration for random positions.     #
# This code operates in ractangular cell only. (11/28/2023)                     #
# This code work in POSCAR, which is input of vasp package.                     #
# openMX input file option incorporated. (12/05/2023)                           #
#################################################################################

def make_vasp(file_name, base_bond_len, Max_delta, base_path, num_new_conf):
    file_path = os.path.join(base_path, file_name)

    file_ori = open(file_path, 'r')

    print('===================================================================')
    print('Starting to make POSCAR files')
    print('-------------------------------------------------------------------')
    print('Reading file.')
    
    lattice_para = []
    atom_pos=[]
    atom_spec=[]
    atom_num_list=[]
    coor_type=[]
    
    for i in range(0,2):
        line = file_ori.readline()
    
    for i in range(0,3):
        a = file_ori.readline().rstrip('\n')
        b = ' '.join(a.split())
        c = b.split(' ')
        lattice_para.append(c)
    
    atom_spec = file_ori.readline().rstrip('\n')
    atom_num_list = file_ori.readline().rstrip('\n')
    coor_type = file_ori.readline().rstrip('\n')
    
    while True:
        a = file_ori.readline().rstrip('\n')
        b = ' '.join(a.split())
        c = b.split(' ')
        if not a: break
        atom_pos.append(c)
    
    print('-------------------------------------------------------------------')
    print('Post-processing.')
    
    atom_num_list = ' '.join(atom_num_list.split())
    atom_num_list = atom_num_list.split(' ')
    atom_num_list = list(map(int, atom_num_list))
    atom_num = sum(atom_num_list)
    
    for i in range(0,3):
        lattice_para[i] = list(map(float, lattice_para[i]))
    
    for i in range(len(atom_pos)):
        atom_pos[i] = list(map(float, atom_pos[i]))
    
    
    if coor_type == 'Direct':
        Max_delta_x = Max_delta/lattice_para[0][0]
        Max_delta_y = Max_delta/lattice_para[1][1]
    elif coor_type == 'Cartesian':
        Max_delta_x = Max_delta
        Max_delta_y = Max_delta

    ##################Re-formating for write the files##################
    
    atom_num_list = list(map(str, atom_num_list))

    for j in range(0,3):
        for i in range(0,3):
            lattice_para[i][j] = format(lattice_para[i][j], '.12f')
    
    ##################random position from gaussian dist.##################

    print('-------------------------------------------------------------------')
    print('Make random position.')

    re_atom_pos = []
    
    for k in range(num_new_conf):
        rand_num = np.random.randn(atom_num,2)
        
        for j in range(0,2):
            for i in range(atom_num):
                if rand_num[i][j] > 1 or rand_num[i][j] < -1:
                    if rand_num[i][j] > 2 or rand_num[i][j] < -2:
                        rand_num[i][j] = rand_num[i][j]/3
                    else :
                        rand_num[i][j] = rand_num[i][j]/4
                else :
                    rand_num[i][j] = rand_num[i][j]/5
        
        for j in range(0,2):
            for i in range(atom_num):
                if rand_num[i][j] > 1:
                    rand_num[i][j] = 1
                elif rand_num[i][j] < -1:
                    rand_num[i][j] = -1
                else :
                    rand_num[i][j] = rand_num[i][j]
        
        tmp_atom_pos = [[0 for j in range(0,3)] for i in range(atom_num)]
        for i in range(atom_num):
            tmp_atom_pos[i][0] = (rand_num[i][0] * Max_delta_x) + atom_pos[i][0]
            tmp_atom_pos[i][1] = (rand_num[i][1] * Max_delta_y) + atom_pos[i][1]

        tmp1_atom_pos = []
        for i in range(atom_num):
            tmp2_atom_pos = []
            for j in range(0,3):
                tmp2_atom_pos.append(tmp_atom_pos[i][j])
            tmp1_atom_pos.append(tmp2_atom_pos)
        re_atom_pos.append(tmp1_atom_pos)


    for k in range(num_new_conf):
        for j in range(0,3):
            for i in range(atom_num):
                re_atom_pos[k][i][j] = format(re_atom_pos[k][i][j], '.12f')


        ##################file wirte##################

    new_poscar_name = [i+1 for i in range(num_new_conf)]
    new_poscar_name = list(map(str, new_poscar_name))

    for i in range(num_new_conf):
        new_poscar_name[i] = 'POSCAR_rand' + new_poscar_name[i]

    for k in range(num_new_conf):
        new_file_path = os.path.join(base_path, new_poscar_name[k])

        print('-------------------------------------------------------------------')
        print('Writing POSCAR file in', new_file_path)
        
        file_new = open(new_file_path, 'w')
        file_new.write('POSCAR\n')
        file_new.write('1.0\n')
        
        for i in range(0,3):
            for j in range(0,3):
                file_new.write('     ')
                file_new.write(lattice_para[i][j])
                file_new.write('\t')
            file_new.write('\n')
        
        for i in range(len(atom_spec)):
            file_new.write(atom_spec[i])
        
        file_new.write('\n')
        
        for i in range(len(atom_num_list)):
            file_new.write('   ')
            file_new.write(atom_num_list[i])
        
        file_new.write('\n')
        file_new.write(coor_type)
        file_new.write('\n')
        
        for j in range(atom_num):
            for i in range(0,3):
                file_new.write('    ')
                file_new.write(re_atom_pos[k][j][i])
            file_new.write('\n')

    print('-------------------------------------------------------------------')

    return re_atom_pos

##################openMX input file##################

def make_openMX(file_name, base_path, num_new_conf, openMX_option, val_elec_up, val_elec_dn, re_atom_pos):
    if openMX_option == 0:
        return 0
    elif openMX_option == 1:
        
        file_path = os.path.join(base_path, file_name)
        
        file_ori = open(file_path, 'r')
        
        print('\n')
        print('===================================================================')
        print('The option of writing input files for OpenMX code is turned on!')
        print('Starting to make input files for OpenMX code')
        print('-------------------------------------------------------------------')
        print('Checking data.')
        
        lattice_para = []
        atom_spec=[]
        atom_num_list=[]
        coor_type=[]
        
        for i in range(0,2):
            line = file_ori.readline()
        
        for i in range(0,3):
            a = file_ori.readline().rstrip('\n')
            b = ' '.join(a.split())
            c = b.split(' ')
            lattice_para.append(c)
        
        atom_spec = file_ori.readline().rstrip('\n')
        atom_num_list = file_ori.readline().rstrip('\n')
        coor_type = file_ori.readline().rstrip('\n')

        print('-------------------------------------------------------------------')
        print('Sorting data.')      

        ##################Sorting data##################
        atom_spec = " ".join(atom_spec.split())
        atom_spec = atom_spec.split(' ')

        atom_num_list = " ".join(atom_num_list.split())
        atom_num_list = atom_num_list.split(' ')
        atom_num_list = list(map(int, atom_num_list))
        atom_num = sum(atom_num_list)

        atom_num_integ = [0 for i in range(len(atom_num_list))]
        for i in range(1,len(atom_spec)):
            atom_num_integ[i] = atom_num_integ[i-1] + atom_num_list[i-1]
        
        if coor_type == 'Direct':
            coor_type = 'FRAC'
        elif coor_type == 'Cartesian':
            coor_type = 'Ang'

        if len(atom_num_list) != len(val_elec_up) or len(atom_num_list) != len(val_elec_dn):
            print('You miss val_elec option in input.')
            return 0

        atom_num_str = str(atom_num)

        print('-------------------------------------------------------------------')
        print('Writing files.')

        ##################file writing##################

        MXinput_name = [i+1 for i in range(num_new_conf)]
        MXinput_name = list(map(str, MXinput_name))

        for i in range(num_new_conf):
            MXinput_name[i] = 'Rand' + MXinput_name[i] +'.in'

        input_name=[]
        for i in MXinput_name:
            tmp = i.replace('.in', '')
            input_name.append(tmp)

        for k in range(num_new_conf):
            new_file_path = os.path.join(base_path, MXinput_name[k])
            
            print('-------------------------------------------------------------------')
            print('Writing openMX file in', new_file_path)
            
            file_new = open(new_file_path, 'w')
            file_new.write('# File Name\n')
            file_new.write('System.CurrrentDirectory  ./    # default=./\n')
            file_new.write('System.Name               ')
            file_new.write(input_name[k])
            file_new.write('\n')
            file_new.write('DATA.PATH                 /TGM/Apps/OpenMX/3.9/openmx3.9.9/DFT_DATA19\n')
            file_new.write('level.of.stdout           1    # default=1 (1-3)\n')
            file_new.write('level.of.fileout          2    # default=1 (0-2)\n')
            file_new.write('\n')

            file_new.write('# Definition of Atomic Species\n')
            file_new.write('Species.Number       ')
            file_new.write(str(len(atom_spec)))
            file_new.write('\n')
            file_new.write('<Definition.of.Atomic.Species\n')
            for i in range(len(atom_spec)):
                file_new.write('  ')
                file_new.write(atom_spec[i])
                file_new.write('\t')
                file_new.write(atom_spec[i])
                file_new.write('6.0-s2p2d1')
                file_new.write('\t')
                file_new.write(atom_spec[i])
                file_new.write('_PBE19')
                file_new.write('\n')
            file_new.write('Definition.of.Atomic.Species>\n')
            file_new.write('\n')

            ##################atomic config##################
            file_new.write('# Atoms\n')
            file_new.write('Atoms.Number\t')
            file_new.write(atom_num_str)
            file_new.write('\n')
            file_new.write('Atoms.SpeciesAndCoordinates.Unit\t')
            file_new.write(coor_type)
            file_new.write('\n')
            file_new.write('<Atoms.SpeciesAndCoordinates\n')
            for x in range(len(atom_spec)):
                for j in range(atom_num_list[x]):
                    file_new.write(str(atom_num_integ[x]+j+1))
                    file_new.write('\t')
                    file_new.write(atom_spec[x])
                    for i in range(0,3):
                        file_new.write('\t')
                        file_new.write(re_atom_pos[k][atom_num_integ[x]+j][i])
                    file_new.write('\t')
                    file_new.write(val_elec_up[x])
                    file_new.write('\t')
                    file_new.write(val_elec_dn[x])
                    file_new.write('\n')
            file_new.write('Atoms.SpeciesAndCoordinates>\n')

            file_new.write('Atoms.UnitVectors.Unit             Ang #  Ang|AU\n')
            file_new.write('<Atoms.UnitVectors                     # unit=Ang.\n')
            for i in range(0,3):
                for j in range(0,3):
                    file_new.write('\t')
                    file_new.write(lattice_para[i][j])
                file_new.write('\n')
            file_new.write('Atoms.UnitVectors>\n')
            file_new.write('\n')

            ##################scf options##################
            file_new.write('# SCF or Electronic System\n')
            file_new.write('scf.XcType                 GGA-PBE     # LDA|LSDA-CA|LSDA-PW\n')
            file_new.write('scf.SpinPolarization       off         # On|Off\n')
            file_new.write('scf.ElectronicTemperature  300.0       # default=300 (K)\n')
            file_new.write('scf.energycutoff           300.0       # default=150 (Ry)\n')
            file_new.write('scf.maxIter                200         # default=40\n')
            file_new.write('scf.EigenvalueSolver       band        # Recursion|Cluster|Band\n')
            file_new.write('scf.lapack.dste            dstevx      # dstegr|dstedc|dstevx, default=dstegr\n')
            file_new.write('scf.Kgrid                 18 18 1      # means nk1xnk2xnk3\n')
            file_new.write('scf.Mixing.Type           rmm-diisk    # Simple|Rmm-Diis|Gr-Pulay\n')
            file_new.write('scf.Init.Mixing.Weight     0.010       # default=0.30\n')
            file_new.write('scf.Min.Mixing.Weight      0.001       # default=0.001\n')
            file_new.write('scf.Max.Mixing.Weight      0.200       # default=0.40\n')
            file_new.write('scf.Mixing.History         15          # default=5\n')
            file_new.write('scf.Mixing.StartPulay       5          # default=6\n')
            file_new.write('scf.criterion             1.0e-8       # default=1.0e-6 (Hartree)\n')
            file_new.write('scf.Ngrid               240 240 240    # x*y*z grids should be able to divide by 6.\n')
            file_new.write('\n')

            ##################MD options##################
            file_new.write('# MD or Geometry Optimization\n')
            file_new.write('MD.Type                     EF         # Nomd|Opt|NVE|NVT_VS|NVT_NH\n')
            file_new.write('                                       # Constraint_Opt|DIIS2|Constraint_DIIS2\n')
            file_new.write('MD.Opt.DIIS.History          5         # default=4\n')
            file_new.write('MD.Opt.StartDIIS             6         # default=5\n')
            file_new.write('MD.Opt.EveryDIIS         10000         # default=10\n')
            file_new.write('MD.maxIter                   1         # default=1\n')
            file_new.write('MD.TimeStep                1.0         # default=0.5 (fs)\n')
            file_new.write('MD.Opt.criterion           1.0e-4      # default=1.0e-4 (Hartree/bohr)\n')
            file_new.write('\n')

            ##################Band and DOS options are turned off##################
            file_new.write('Band.dispersion             off       # on|off, default=off\n')
            file_new.write('\n')
            file_new.write('Dos.fileout                 off       # on|off, default=off\n')

        print('-------------------------------------------------------------------')
        print('All files are written!')
        print('-------------------------------------------------------------------')
        print('\n')
        print('===================================================================\n')
        print('You should be changed the input files!')
        print('Especially, "Definition.of.Atomic.Specie" block!\n')
        print('===================================================================\n')
