"""
pickleデータから、BoNetに入力するh5pyデータ形式へ変更する

example:
     python create_dataset.py -i ../../data/qbs-data -o ../../data/qbs-h5 -area_dist 10
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
parser.add_argument('-area_dist', type=float, default=10.0, help='分割するエリアのサイズ(m)')
parser.add_argument('-dataset_boost', type=int, default=0, help='データセットを増やす')


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
dist = args.area_dist

dataset_boost = args.dataset_boost
if dataset_boost != 0:
    import math
    dists = []
    ldist = int(math.floor(dist/2))
    for d in range(ldist, int(dist) + 1):
        dists.append(d)
    # print("Dataset boosting: {}".format(dists))
else:
    dists = [dist]

# データの読み込み
input_files = files = sorted(glob.glob(os.path.join(input, '*')))



for i, data_file in enumerate(input_files):
    point, label = get_qbs_data(data_file)
    batches = []
    for d in dists:
        batch = area_to_block(point, num_points=num_points, label=label, dist=d, threshold=1000) #  data_boost=dataset_boost

        if len(dists) == 1:
            output_file = os.path.join(output, "area_{0}.h5".format(i+1))
        else:
            output_file = os.path.join(output, "area_{0}{1}.h5".format(i+1, d))

        print('> Saving batch to {} ...'.format(output_file))
        if not os.path.exists(output_file):
            save_batch_h5(output_file, batch)
        else:
            print("Already exsist: {}".format(output_file))
