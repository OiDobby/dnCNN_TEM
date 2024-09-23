import numpy as np
import math
import time
import os
import shutil
import matplotlib.pyplot as plt
from ran_atomic_util import make_vasp, make_openMX

#################################################################################
# This code makes the rearranged atomic configuration for random positions.     #
# This code operates in ractangular cell only. (11/28/2023)                     #
# Thic code work in POSCAR, which is input of vasp package.                     #
#################################################################################

file_name = 'POSCAR-Gr_mid_supercell'

base_bond_len = 1.42   # This number indicates the average bond length in atomic configuration.
                       # It sets the range of random positions for rearranged atomic configuration.

Max_delta = base_bond_len * 0.2   # It will work in both x and y positions.

base_path = os.getcwd()   # Current directory

num_new_conf = 4   # How many POSCAR you need?

openMX_option = 1  # You need to make openMX input? 1 = yes, 0 = no

#These two lines need to make openMX input. Please, make it in string list.
val_elec_up = ['2.0']  # number of val_electron in spin-up; [atom_species1, atom_species2...]
val_elec_dn = ['2.0']  # number of val_electron in spin-down; [atom_species1, atom_species2...]

#example in 3 atom species
#val_elec_up = ['2.0','3.0', '2.5']
#val_elec_dn = ['2.0', '2.0', '3.0']

re_atom_pos = make_vasp(file_name, base_bond_len, Max_delta, base_path, num_new_conf)

make_openMX(file_name, base_path, num_new_conf, openMX_option, val_elec_up, val_elec_dn, re_atom_pos)

print('Program done')
