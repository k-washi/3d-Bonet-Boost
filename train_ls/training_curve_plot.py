import pickle
import matplotlib.pyplot as plt


def pickle_load(path):
    data = None
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data

def list_plot(datas, names, figname = "test"):
    plt.figure()

    if len(datas[0]) != len(names[0]):
        raise Exception("Args error: 入力の長さが違う")
    for i in range(len(datas)):
        step = range(0, len(datas[i]))
        plt.plot(step, datas[i], label=names[i], alpha=0.5)

    plt.legend()
    plt.ylim([-0.5, 2])
    plt.savefig('img/' + figname + '.png')

def one_list_plot(data, name, y_min = -1., y_max = 2.):
    plt.figure()

    step = range(0, len(data))
    plt.plot(step, data)
    plt.ylim([y_min, y_max])
    plt.savefig('img/' + name + '.png')


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

if __name__ == "__main__":
    datas = []
    smdatas = []
    names = []
    file_name = "area1_ep10/"

    names.append("psemce")
    data_path = file_name + "ls_psemce.pickle"
    ls_psemce = pickle_load(data_path)
    if ls_psemce is not None:
        datas.append(ls_psemce)
    smls_psemce = smooth_curve(ls_psemce)
    smdatas.append(smls_psemce)

    one_list_plot(ls_psemce, names[-1], y_min=-0.1, y_max=1.)

    names.append("bbscore")
    data_path = file_name + "ls_bbscore.pickle"
    ls_bbscore = pickle_load(data_path)
    if ls_bbscore is not None:
        datas.append(ls_bbscore)
    smls_bbscore = smooth_curve(ls_bbscore)
    smdatas.append(smls_bbscore)
    one_list_plot(ls_bbscore, names[-1], y_min=-0.1, y_max=0.3)

    names.append("bbvert")
    data_path = file_name + "ls_bbvert.pickle"
    ls_bbvert = pickle_load(data_path)
    if ls_bbvert is not None:
        datas.append(ls_bbvert)
    smls_bbvert = smooth_curve(ls_bbvert)
    smdatas.append(smls_bbvert)
    one_list_plot(ls_bbvert, names[-1], y_min=-0.1, y_max=0.7)

    names.append("pmask")
    data_path = file_name + "ls_pmask.pickle"
    ls_pmask = pickle_load(data_path)
    if ls_pmask is not None:
        datas.append(ls_pmask)
    smls_pmaks = smooth_curve(ls_pmask)
    smdatas.append(smls_pmaks)
    one_list_plot(ls_pmask, names[-1], y_min=-0.1, y_max=0.7)

    #list_plot(datas, names, figname="test")
    #list_plot(smdatas, names, figname="smooth")