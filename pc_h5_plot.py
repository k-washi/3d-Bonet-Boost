import h5py
from eval.pc_plot_helper import Plot
import numpy as np
def load_h5py(path):
    return h5py.File(path, 'r')


if __name__ == '__main__':
    #path = "/Users/washizakikai/dev/work/pc/data/pc-h5/area_1.h5"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='path to input directory: h5')

    args = parser.parse_args()

    path = args.i

    data = load_h5py(path)

    cd = data['coords']
    pc = data['points']
    #Plot.draw_pc(np.concatenate(pc[:, :, 0:3], axis=0))
    Plot.draw_pc(np.concatenate(cd[:, :, 0:3], axis=0))
    #print(cd.shape, pc.shape)