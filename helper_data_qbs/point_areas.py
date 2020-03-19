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

def boost_arera(areas, dist):
    """
    areaをスライド
    :param areas:
    :param dist:
    :return:
    """
    if dist < 2:
        return areas
    if len(areas) < 2:
        return areas

    temp = []
    for area in areas[:-1]:
        temp.append(area)
        for i in range(2,4):
            a_min, a_max = area[0] + int(dist/i), area[1] + int(dist/i)
            temp.append((a_min, a_max))

    temp.append(areas[-1])
    return temp

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
        indices = np.random.choice(n, num_samples, replace=True)  # replaceをTrueにして、重複は許可するものとする。
    else:
        # print("Info point sample: data num {0} < sample num {1}".format(n, num_samples))
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled


def area_to_block(point, num_points, label=None, dist=15, threshold=100, size=1.0):
    if label is not None:
        point = _concate_point_labal(point, label)

    limit = np.amax(point[:, 0:3], axis=0)
    for i in range(3):
        if limit[i] < 0:
            limit[i] = np.min(point[:, i])

    xareas, yareas = get_areas(point, dist)
    #print(xareas)
    cells = [(xi, yi) for xi in range(len(xareas)) for yi in range(len(yareas))]

    blocks = []
    count = 0
    skip_count = 0
    for xi, yi in cells:
        xcond = (xareas[xi][0] <= point[:, 0]) & (point[:, 0] <= xareas[xi][1])
        ycond = (yareas[yi][0] <= point[:, 1]) & (point[:, 1] <= yareas[yi][1])
        cond = xcond & ycond

        if np.sum(cond) < threshold:
            skip_count += 1
            continue
        block = point[cond, :]

        # 　ニューラルネットワークの入力に対して、データセットの点群を多いのでサンプリングする。
        # train_ls/img/cover_rate2にデータの9割以上をサンプリングできる回数を示す
        # 今回は、(全体の点数/ サンプリング点数) x 2 + 1とする。
        # ただし, 全体の点数 <= サンプリング点数の場合、1回のみサンプリングする。

        if block.shape[0] <= num_points:
            sampling_num = 1
            count += 1
        else:
            sampling_num = math.floor(block.shape[0] / num_points) * 2 + 1

        for _ in range(sampling_num):
            block = sample_point(block, num_points)
            blocks.append(block)
    print("点数が少なすぎるためスキップしました。：{}".format(skip_count))
    print("点数が足りないブロック数 {0} / {1}".format(count, len(blocks)))
    blocks = np.stack(blocks, axis=0)
    print("Create {0} point area block: {1}".format(num_points, blocks.shape[0]))

    # 以下の形式のbatchを生成 BxNx11
    # [0:3] - global coordinates
    # [3:6] - block normalized coordinates (centered at Z-axis) #blockのx, yのminを0に
    # [6:9] - all normalized coordinates #データ全体に対する正規化
    # [9:11] - semantic and instance labels
    num_blocks = blocks.shape[0]
    batch = np.zeros((num_blocks, num_points, 11))
    count = 0

    for b in range(num_blocks):
        minx = min(blocks[b, :, 0])
        miny = min(blocks[b, :, 1])
        batch[b, :, 3] = blocks[b, :, 0] - (minx + size * 0.5)
        batch[b, :, 4] = blocks[b, :, 1] - (miny + size * 0.5)
        batch[b, :, 5] = blocks[b, :, 2]
        batch[b, :, 6] = blocks[b, :, 0] / limit[0]
        batch[b, :, 7] = blocks[b, :, 1] / limit[1]
        batch[b, :, 8] = blocks[b, :, 2] / limit[2]

        uq = np.unique(blocks[b, :, 4])
        if uq.shape[0] > 40:
            # print(uq.shape[0])
            count += 1

    print("1ブロックに含まれるインスタンスの数が40を超えたブロック数 {0} / {1}".format(count, num_blocks))

    #batch[:, :, 0:3] = blocks[:, :, 0:3]

    if label is not None:
        batch[:, :, 9:11] = blocks[:, :, 3:5]

    # batch[:, :, :] = batch[0, :, :]

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
    # cloud = _concate_point_labal(point, label)
    # print(cloud.shape)
    # get_areas(point, dist=5)
    batch = area_to_block(point, 4096, label=label, dist=10)
    print(batch.shape)
    print("END")
