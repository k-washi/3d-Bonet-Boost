import os
import numpy as np
import math
import h5py


def _xyz_min_max(point):
    mins = np.amin(point[:, 0:3], axis=0)
    maxs = np.amax(point[:, 0:3], axis=0)
    return mins, maxs


def _areas_sort(areas):
    """
    areasを昇順に並べ替える
    :param areas:
    :return:
        result: list[a1, a2, ...], a1 = (min, max)
    """

    areas_min = []
    result = []
    for amin, _ in areas:
        areas_min.append(amin)
    areas_min = np.array(areas_min)
    inds = np.argsort(areas_min)
    for ind in inds:
        result.append(areas[ind])

    return result


def _areas_value(vmin, vmax, dist=5.):
    """
    distごとに区切られたvmin, vmaxの間の領域を作成
    example:
        input(vmin=-5.5, vmax=12.0, dist=5.)
        output: [(-5.5, -0.5), (-2.375, 2.625), (0.75, 5.75), (3.875, 8.875), (7.0, 12)]
    """
    areas = []
    if vmax - vmin <= dist:
        areas.append((vmin, vmax))
        return areas
    # 端を除去する処理
    areas.append((vmin, vmin + dist))
    areas.append((vmax - dist, vmax))
    areas = _areas_sort(areas)

    while True:
        new_areas = []
        for i, _ in enumerate(areas):
            if len(areas) - 1 == i:
                break

            if areas[i + 1][0] > areas[i][1]:
                center = (areas[i + 1][0] - areas[i][1]) / 2. + areas[i][1]
                new_areas.append((center - dist / 2., center + dist / 2.))
                break

        areas += new_areas
        areas = _areas_sort(areas)

        if len(new_areas) == 0:
            break
    return areas


def get_areas(point, dist=5.):
    pmins, pmaxs = _xyz_min_max(point)
    xmin, xmax = pmins[0], pmaxs[0]
    ymin, ymax = pmins[1], pmaxs[1]

    xareas = _areas_value(xmin, xmax, dist=dist)
    yareas = _areas_value(ymin, ymax, dist=dist)

    num_xarea = len(xareas)
    num_yarea = len(yareas)
    print("Create Area widht: x:{0}, y:{1}, prod:{2}".format(num_xarea, num_yarea, num_xarea * num_yarea))

    return xareas, yareas

def _concate_point_labal(point, label):
    return np.concatenate([point, label], axis=1)

def sample_point(cloud, num_samples):
    """
    cloudからデータをサンプリングする
    :param cloud:
    :param num_samples:
    :return:
    """
    n = cloud.shape[0]
    if n >= num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
    else:
        print("Info point sample: data num < sample num")
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled



def area_to_block(point, num_points, label = None, dist=15, threshold=100, size = 1.0):
    if label is not None:
        point = _concate_point_labal(point, label)

    limit = np.amax(point[:, 0:3], axis=0)
    xareas, yareas = get_areas(point, dist)
    cells = [(xi, yi) for xi in range(len(xareas)) for yi in range(len(yareas))]

    blocks = []
    for xi, yi in cells:
        xcond = (xareas[xi][0] <= point[:, 0]) & (point[:, 0] <= xareas[xi][1])
        ycond = (yareas[yi][0] <= point[:, 1]) & (point[:, 1] <= yareas[yi][1])
        cond = xcond & ycond

        if np.sum(cond) < threshold:
            continue
        block = point[cond, :]

        for _ in range(math.floor(block.shape[0] / num_points)):
            # データが少ないので複数回サンプリングを繰り返す
            block = sample_point(block, num_points)
            blocks.append(block)

    blocks = np.stack(blocks, axis=0)
    print("Create {0} point area block: {1}".format(num_points, blocks.shape[0]))

    # 以下の形式のbatchを生成 BxNx11
    # [0:3] - global coordinates
    # [3:6] - block normalized coordinates (centered at Z-axis) #blockのx, yのminを0に
    # [6:9] - all normalized coordinates #データ全体に対する正規化
    # [9:11] - semantic and instance labels
    num_blocks = blocks.shape[0]
    batch = np.zeros((num_blocks, num_points, 11))
    for b in range(num_blocks):
        minx = min(blocks[b, :, 0])
        miny = min(blocks[b, :, 1])
        batch[b, :, 3] = blocks[b, :, 0] - (minx + size * 0.5)
        batch[b, :, 4] = blocks[b, :, 1] - (miny + size * 0.5)
        batch[b, :, 6] = blocks[b, :, 0] / limit[0]
        batch[b, :, 7] = blocks[b, :, 1] / limit[1]
        batch[b, :, 8] = blocks[b, :, 2] / limit[2]
    batch[:, :, 0:3] = blocks[:, :, 0:3]
    # batch[:, :, 5:9] = blocks[:, :, 2:6]
    if label is not None:
        batch[:, :, 9:11] = blocks[:, :, 3:5]
    return batch

def save_batch_h5(fname, batch):
    fp = h5py.File(fname)
    coords = batch[:, :, 0:3]
    points = batch[:, :, 3:9]
    labels = batch[:, :, 9:11]
    fp.create_dataset('coords', data=coords, compression='gzip', dtype='float32')
    fp.create_dataset('points', data=points, compression='gzip', dtype='float32')
    fp.create_dataset('labels', data=labels, compression='gzip', dtype='int64')
    fp.close()


if __name__ == "__main__":
    import sys

    data_path = '/Users/washizakikai/dev/work/qbs/data/qbs-data'
    file_name = 'xmin_-242_ymin_37_delta_50.pkl'
    pickle_path = os.path.join(data_path, file_name)

    from load_pickle import get_qbs_data

    # point, label = _read_data(pickle_path)
    # print("Data length: ", len(point), len(label))
    # print("First daata: ", point[0], label[0])

    point, label = get_qbs_data(pickle_path)
    print(point.shape, label.shape)
    #cloud = _concate_point_labal(point, label)
    #print(cloud.shape)
    # get_areas(point, dist=5)
    batch = area_to_block(point, 4096)
    print(batch.shape)
    print("END")
