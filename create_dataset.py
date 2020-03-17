"""
pickleデータから、BoNetに入力するh5pyデータ形式へ変更する

example:
    python create_dataset.py -i ../../data/qbs-data -o ../../data/qbs-h5

"""

import argparse
import numpy as np
import glob
import os
import sys

from helper_data_qbs.load_pickle import get_qbs_data
from helper_data_qbs.point_areas import area_to_block, save_batch_h5

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='path to input directory: pickle形式')
parser.add_argument('-o', help='path to output directory: h5形式')
parser.add_argument('-seed', type=int, default=42, help='random number seed')
parser.add_argument('-num_point', type=int, default=4092, help='点群データの点数')

args = parser.parse_args()

input = args.i
output = args.o

if not os.path.isdir(input):
    print('[WARNING] Cannot find {}'.format(input))
    sys.exit(-1)

if not os.path.isdir(output):
    print('[WARNING] Cannot find {}'.format(output))
    sys.exit(-1)

# データの読み込みに関するシード
seed = args.seed
np.random.seed(seed)

num_points = args.num_point


#データの読み込み
input_files = files = sorted(glob.glob(os.path.join(input, '*')))
print(input_files)

for i, data_file in enumerate(input_files):
    point, label = get_qbs_data(data_file)
    batch = area_to_block(point, num_points=num_points)

    output_file = os.path.join(output, "area_{}.h5".format(i))
    print('> Saving batch to {} ...'.format(output_file))
    if not os.path.exists(output_file):
        save_batch_h5(output_file, batch)
    else:
        print("Already exsist: {}".format(output_file))


