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


def get_qbs_data(data_path):
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

    return point, label




if __name__ == "__main__":
    data_path = '/Users/washizakikai/dev/work/qbs/data/qbs-data'
    file_name = 'xmin_-242_ymin_37_delta_50.pkl'
    pickle_path = os.path.join(data_path, file_name)

    
    point, label = _read_data(pickle_path)
    print("Data length: ", len(point), len(label))
    print("First daata: ", point[0], label[0])


    print("END")
