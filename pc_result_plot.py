"""
python ./pc_result_plot.py -i ../../data/pc_train_data/test_res/area_1/res_by_scene/area_1.h5.mat

"""
import scipy.io
import numpy as np

from eval.pc_plot_helper import Plot, Eval_Tools

def mat_data_lead(path):
    return scipy.io.loadmat(path,  verify_compressed_data_integrity=False)

def create_plot_data(scene_result, block = None, h5path = None):
    pc_all = [];
    ins_gt_all = [];
    sem_pred_all = [];
    sem_gt_all = []
    gap = 5e-3
    volume_num = int(1. / gap) + 2
    volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)
    volume_sem = -1 * np.ones([volume_num, volume_num, volume_num]).astype(np.int32)

    if block is not None:
        b0 = block
        block_num = block + 1
    else:
        b0 = 0
        block_num = len(scene_result)


    #if h5path:
    #    from pc_h5_plot import load_h5py
    #    pc_data = load_h5py(h5path)


    for i in range(b0,block_num):
        block = 'block_' + str(i).zfill(4)
        #print(block)
        if block not in scene_result: continue
        """
        
        if h5path:
            pc = scene_result[block][0]['pc'][0]
            pc[:, 0:3] = pc_data['coords'][i]
        else:
        """

        pc = scene_result[block][0]['pc'][0]
            
        ins_gt = scene_result[block][0]['ins_gt'][0][0]
        sem_gt = scene_result[block][0]['sem_gt'][0][0]
        bbscore_pred_raw = scene_result[block][0]['bbscore_pred_raw'][0][0]
        pmask_pred_raw = scene_result[block][0]['pmask_pred_raw'][0]
        sem_pred_raw = scene_result[block][0]['sem_pred_raw'][0]

        sem_pred = np.argmax(sem_pred_raw, axis=-1)
        pmask_pred = pmask_pred_raw * np.tile(bbscore_pred_raw[:, None], [1, pmask_pred_raw.shape[-1]])
        ins_pred = np.argmax(pmask_pred, axis=-2)
        ins_sem_dic = Eval_Tools.get_sem_for_ins(ins_by_pts=ins_pred, sem_by_pts=sem_pred)
        Eval_Tools.BlockMerging(volume, volume_sem, pc[:, 3:6], ins_pred, ins_sem_dic, gap)

        pc_all.append(pc)
        ins_gt_all.append(ins_gt)
        sem_pred_all.append(sem_pred)
        sem_gt_all.append(sem_gt)
        
    #print(len(pc_all), pc_all[0].shape)
    pc_all = np.concatenate(pc_all, axis=0)
    ins_gt_all = np.concatenate(ins_gt_all, axis=0)
    sem_pred_all = np.concatenate(sem_pred_all, axis=0)
    sem_gt_all = np.concatenate(sem_gt_all, axis=0)

    pc_xyz_int = (pc_all[:, 3:6] / gap).astype(np.int32)
    ins_pred_all = volume[tuple(pc_xyz_int.T)]
    return pc_all, ins_gt_all, ins_pred_all, sem_gt_all, sem_pred_all

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='data path to mat file')
    parser.add_argument('-b', help='block id')
    parser.add_argument('-v', type=int, default=0, help='viewablility priority')
    parser.add_argument('-h5', help='h5 original data')
    

    args = parser.parse_args()

    input = args.i
    if not input:
        raise Exception("Error: 引数 -i (plotするmatファイルへのパス) を設定して下さい")

    block = args.b
    if not block:
        block = None
    else:
        block = int(block)

    vp = args.v
    if vp == 0:
        vp = False
    else:
        vp = True

    h5path = args.h5
        
    scene_res = mat_data_lead(input)


    #print(scene_res)

    pc_all, ins_gt_all, ins_pred_all, sem_gt_all, sem_pred_all = create_plot_data(scene_res, block = block)
    
    #Plot.draw_pc(pc_all[:,6:9])
    Plot.draw_pc_semins(pc_xyz=pc_all[:, 6:9], pc_semins=ins_gt_all, draw_bb=True)
    Plot.draw_pc_semins(pc_xyz=pc_all[:, 6:9], pc_semins=ins_pred_all, draw_bb=True, viewability=vp)
    """
    elif not block:
        Plot.draw_pc_semins(pc_xyz=pc_all[:, 0:3], pc_semins=ins_pred_all, draw_bb=True, viewability=vp)
    else:
        Plot.draw_pc_semins(pc_xyz=pc_all[:, 0:3], pc_semins=ins_pred_all, draw_bb=True, viewability=vp)
    """
    #Plot.draw_pc_semins(pc_xyz=pc_all[:, 6:9], pc_semins=sem_gt_all)
    #Plot.draw_pc_semins(pc_xyz=pc_all[:, 6:9], pc_semins=sem_pred_all)