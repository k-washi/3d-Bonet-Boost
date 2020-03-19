import h5py
import numpy as np

def get_file_list(dir_path):
    import glob
    files = sorted(glob.glob( dir_path+ '*.h5'))
    return files

def read_h5py(file):
    data = h5py.File(file, 'r')
    #print(data)
    coords = data['coords']
    points = data['points']
    semIns_labels = data['labels']

    new_points = np.zeors((points.shape[0], points.shape[1]))

    #print(coords.shape, points.shape, semIns_labels.shape)


if __name__ == '__main__':
    dir_path = '../../../data/Data_S3DIS/'
    files = get_file_list(dir_path)
    read_h5py(files[0])
