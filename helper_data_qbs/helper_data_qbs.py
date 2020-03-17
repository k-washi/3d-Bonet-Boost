import glob
import numpy as np
import random
import copy
from random import shuffle
import h5py

class Data_Configs:
    sem_names = ['otherwise', 'wood']
    sem_ids = [0, 1]

    points_cc = 6  # [x, y, z, x_normal, y_normal, z_normal #  color情報なし[, c_R, c_G, c_B]
    sem_num = len(sem_names)
    ins_max_num = 24
    train_pts_num = 4096
    test_pts_num = 4096



