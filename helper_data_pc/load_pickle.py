import pickle
import os
import sys
import logging
import numpy as np


def _open_pickle(data_path):
    with open(data_path, 'br') as f:
        return pickle.load(f)
    print("data is not loaded")
    return None


def _read_data(data_path):
    data = _open_pickle(data_path)
    try:
        point = data['point']
        label = data['id']
        return point, label
    except Exception as e:
        logging.error("Error: data can not get: {0}".format(e))
        sys.exit(-1)


def get_pc_data(data_path):
    """

    :param data_path: pickle data path
    :return:
        point: (point num, xyz)
        label: (point num, [seg label, ins label])
    """
    point, label = _read_data(data_path)

    # convert numpy
    point = np.array(point).astype(np.float)
    label = np.array(label).astype(np.int)

    point, label = data_Fix(point, label)

    print("データの修正----------------------------")
    _, _ = data_checker(point, label)
    return point, label


def data_checker(point, label):
    """
    インスタンスラベルと分類ラベルの不一致
    :param point:
    :param label:
    :return:
    """
    other_count = 0
    forest_count = 0
    for lab, ins in label:
        # 樹木以外は分類ラベル(0), インスタンスラベル(-1)
        if lab == 0 and ins != -1:
            other_count += 1

        # 樹木は分類ラベル(1), インスタンスラベル(0以上)
        if lab != 0 and ins == -1:
            forest_count += 1
    print("樹木以外の物体に、樹木のインスタンス(0以上)が割り当てられている割合 {0} / {1}".format(other_count, label.shape[0]))
    print("樹木に、樹木以外のインスタンス(-1）が割り当てられている割合 {0} / {1}".format(forest_count, label.shape[0]))
    return other_count, forest_count


def data_Fix(point, label):
    """
    インスタンスラベルにより分類ラベルを修正
    :param point:
    :param label:
    :return:
    """
    other_count, forest_count = data_checker(point, label)
    if other_count > 0 or forest_count > 0:
        label[label[:, 1] == -1, 0] = 0
        label[label[:, 1] >= 0, 0] = 1
    return point, label


if __name__ == "__main__":
    data_path = '/Users/washizakikai/dev/work/pc/data/pc-data'
    file_name = 'xmin_-242_ymin_37_delta_50.pkl'
    pickle_path = os.path.join(data_path, file_name)

    point, label = get_pc_data(pickle_path)
    # print("Data length: ", len(point), len(label))
    # print("First daata: ", point[0], label[0])

    print("END")
