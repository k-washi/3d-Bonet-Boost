import os
import numpy as np

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


def get_areas(point, dist = 5.):
    pmins, pmaxs = _xyz_min_max(point)
    xmin, xmax = pmins[0], pmaxs[0]
    ymin, ymax = pmins[1], pmaxs[1]

    xareas = _areas_value(xmin, xmax, dist=dist)
    yareas = _areas_value(ymin, ymax, dist=dist)

    num_xarea = len(xareas)
    num_yarea = len(yareas)
    print("Create Area widht: x:{0}, y:{1}, prod:{2}".format(num_xarea, num_yarea, num_xarea*num_yarea))

if __name__ == "__main__":
    data_path = '/Users/washizakikai/dev/work/qbs/data/qbs-data'
    file_name = 'xmin_-242_ymin_37_delta_50.pkl'
    pickle_path = os.path.join(data_path, file_name)

    from helper_data_qbs.load_pickle import get_qbs_data

    # point, label = _read_data(pickle_path)
    # print("Data length: ", len(point), len(label))
    # print("First daata: ", point[0], label[0])

    point, label = get_qbs_data(pickle_path)
    get_areas(point, dist=5)

    print("END")