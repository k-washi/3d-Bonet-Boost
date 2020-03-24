# coding:utf-8
"""
python pc_pcd2pkl.py -i ../../data/cut_50_pcd/xmin_-392_ymin_137_delta_50.pcd -o ../../data/test.pkl

"""
import pickle
import pandas as pd


def load_pcd(file):
    data = pd.read_table(file, header=None)
    #head = data[:10]
    data = data[10:].reset_index(drop=True)
    data = data.iloc[:, 0].apply(lambda x: pd.Series(x.split()))

    data2 = data.copy()
    data2[3] = ((data[3].astype(int) < 7) & (data[3] != '0')) * 1
    return data2


def save_h5(data, path):
    points, ids = [], []
    for i in range(len(data)):
        x, y, z, label, obj = data.iloc[i]
        points.append((x, y, z))
        ids.append((label, obj))
    pcd_dict = {'point': points, 'id': ids}
    with open(path, 'wb') as f:
        pickle.dump(pcd_dict, f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to input directory: pcd')
    parser.add_argument('-o', help='file name of pickle')

    args = parser.parse_args()

    input = args.i
    output = args.o

    data = load_pcd(input)
    print(data.head(20))
    save_h5(data, output)
