import numpy as np
# import os
import scipy.io
import scipy.stats
# import copy
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import open3d  ## version 0.8
import random
import colorsys
import math
import glob


class Plot(object):
    @staticmethod
    def random_colors(N, bright=True, seed=0):
        brightness = 1.0 if bright else 0.7
        hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.seed(seed)
        random.shuffle(colors)
        return colors

    @staticmethod
    def draw_pc(pc_xyzrgb):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.utility.Vector3dVector(pc_xyzrgb[:, 0:3])
        if pc_xyzrgb.shape[1] == 3:
            open3d.visualization.draw_geometries([pc])
            return 0
        if np.max(pc_xyzrgb[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_xyzrgb[:, 3:6])
        open3d.visualization.draw_geometries([pc])
        return 0

    @staticmethod
    def draw_ins_bb(pc_semins, bbs, viewability = False, outlier_width = 0.):
        """
        o3d.visualization.draw_geometries(
        [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

        print("Let\'s draw a cubic using o3d.geometry.LineSet")
        points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
                  [0, 1, 1], [1, 1, 1]]
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]
        colors = [[1, 0, 0] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        """
        pc = open3d.geometry.PointCloud()

        mins = np.amin(pc_semins[:, 0:3], axis=0)
        maxs = np.amax(pc_semins[:, 0:3], axis=0)
        #print(mins, maxs)
        pc.points = open3d.utility.Vector3dVector(pc_semins[:, 0:3])
        if np.max(pc_semins[:, 3:6]) > 20:  ## 0-255
            pc.colors = open3d.utility.Vector3dVector(pc_semins[:, 3:6] / 255.)
        else:
            pc.colors = open3d.utility.Vector3dVector(pc_semins[:, 3:6])

        bb_pcs = []
        bb_inds = []
        bb_c = []
        lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
                 [0, 4], [1, 5], [2, 6], [3, 7]]

        for i, bb in enumerate(bbs[:]):
            #  [xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]

            if viewability and int(i % 2) == 0:
                print("None Vis")
                continue

            if bb[1][0] < bb[0][0] or bb[1][1] < bb[0][1] or bb[1][2] < bb[0][2]:
                print(bb)
                continue
            if bb[1][0] - bb[0][0] > 30 or bb[1][1] - bb[0][1] > 30 or bb[1][2] - bb[0][2] > 30:
                #print("検出物体が大きすぎる")
                continue

            if bb[1][0] - bb[0][0] < 0.5 or bb[1][1] - bb[0][1] < 0.5 or bb[1][2] - bb[0][2] < 0.5:
                #print("検出物体が小さすぎる")
                continue

            if bb[2][0] == 0 and bb[2][1] == 0 and bb[2][2] == 0:
                print("include zero")
                continue

            if bb[0][0] < (mins[0] - outlier_width) or bb[1][0] > (maxs[0] + outlier_width) \
                    or bb[0][1] < (mins[1] - outlier_width) or bb[1][1] > (maxs[1] + outlier_width) \
                    or bb[0][2] < (mins[2] - outlier_width) or bb[1][2] > (maxs[2] + outlier_width):
                print("Get outlier")
                continue

            if bb[1][2] - bb[0][2] < bb[1][0] - bb[0][0] or bb[1][2] - bb[0][2] < bb[1][1] - bb[0][1]:
                continue



            bb_pc = [bb[0],
                     [bb[1][0], bb[0][1], bb[0][2]],
                     [bb[0][0], bb[1][1], bb[0][2]],
                     [bb[1][0], bb[1][1], bb[0][2]],
                     [bb[0][0], bb[0][1], bb[1][2]],
                     [bb[1][0], bb[0][1], bb[1][2]],
                     [bb[0][0], bb[1][1], bb[1][2]],
                     bb[1]]
            bb_pcs.extend(bb_pc)

            for line in lines:
                bb_inds.append([line[0] + len(bb_pc) * i, line[1] + len(bb_pc) * i])
                bb_c.append(bb[2])

        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(bb_pcs)
        line_set.lines = open3d.utility.Vector2iVector(bb_inds)
        line_set.colors = open3d.utility.Vector3dVector(bb_c)

        open3d.visualization.draw_geometries([pc, line_set])


    @staticmethod
    def draw_pc_semins(pc_xyz, pc_semins, fix_color_num=None, draw_bb=False, viewability = False):
        if fix_color_num is not None:
            ins_colors = Plot.random_colors(fix_color_num + 1, seed=2)
        else:
            ins_colors = Plot.random_colors(len(np.unique(pc_semins)) + 1, seed=2)  # cls 14

        ##############################
        semins_labels = np.unique(pc_semins)
        semins_bbox = []
        Y_colors = np.zeros((pc_semins.shape[0], 3))
        for id, semins in enumerate(semins_labels):
            valid_ind = np.argwhere(pc_semins == semins)[:, 0]
            if semins <= -1:
                tp = [0, 0, 0]
                continue
            else:
                if fix_color_num is not None:
                    tp = ins_colors[semins]
                else:
                    tp = ins_colors[id]

            Y_colors[valid_ind] = tp

            ### bbox
            valid_xyz = pc_xyz[valid_ind]

            xmin = np.min(valid_xyz[:, 0])
            xmax = np.max(valid_xyz[:, 0])
            ymin = np.min(valid_xyz[:, 1])
            ymax = np.max(valid_xyz[:, 1])
            zmin = np.min(valid_xyz[:, 2])
            zmax = np.max(valid_xyz[:, 2])
            #print(xmin, ymin, zmin, xmax, ymax, zmax)
            """
            if xmin <= 0. or 1. <= xmax or ymin <= 0. or 1. <= ymax or zmin <= 0. or 1. <= zmax:
                #print(xmin, ymin, zmin,xmax, ymax, zmax)
                continue
            if math.fabs(xmax - xmin) < 10e-5 or math.fabs(ymax - ymin) < 10e-5 or math.fabs(zmax - zmin) < 10e-5:
                continue

            if math.fabs(ymin - xmin) < 10e-5 or math.fabs(zmin - xmin) < 10e-5 or math.fabs(zmin - ymin) < 10e-5:
                #print(math.fabs(ymin - xmin),math.fabs(zmin - xmin),math.fabs(zmin - ymin))
                continue
            if math.fabs(ymax - xmax) < 10e-5 or math.fabs(zmax - xmax) < 10e-5 or math.fabs(zmax - ymax) < 10e-5:
                #print(math.fabs(ymax - xmax), math.fabs(zmax - xmax), math.fabs(zmax - ymax))
                continue
            """
            semins_bbox.append(
                [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

        Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)

        if draw_bb:
            Plot.draw_ins_bb(Y_semins, semins_bbox, viewability)
            return Y_semins
        Plot.draw_pc(Y_semins)
        return Y_semins


class Eval_Tools:
    @staticmethod
    def get_scene_list(res_blocks):
        scene_list_dic = {}
        for b in res_blocks:
            scene_name = b.split('/')[-1][0:-len('_0000')]
            if scene_name not in scene_list_dic: scene_list_dic[scene_name] = []
            scene_list_dic[scene_name].append(b)
        if len(scene_list_dic) == 0:
            print('scene len is 0, error!')
            exit()
        return scene_list_dic

    @staticmethod
    def get_sem_for_ins(ins_by_pts, sem_by_pts):
        ins_cls_dic = {}
        ins_idx, cnt = np.unique(ins_by_pts, return_counts=True)
        for ins_id, cn in zip(ins_idx, cnt):
            if ins_id == -1: continue  # empty ins
            temp = sem_by_pts[np.argwhere(ins_by_pts == ins_id)][:, 0]
            sem_for_this_ins = scipy.stats.mode(temp)[0][0]
            ins_cls_dic[ins_id] = sem_for_this_ins
        return ins_cls_dic

    @staticmethod
    def BlockMerging(volume, volume_seg, pts, grouplabel, groupseg, gap=1e-3):
        overlapgroupcounts = np.zeros([100, 1000])
        groupcounts = np.ones(100)
        x = (pts[:, 0] / gap).astype(np.int32)
        y = (pts[:, 1] / gap).astype(np.int32)
        z = (pts[:, 2] / gap).astype(np.int32)
        for i in range(pts.shape[0]):
            xx = x[i]
            yy = y[i]
            zz = z[i]
            if grouplabel[i] != -1:
                if volume[xx, yy, zz] != -1 and volume_seg[xx, yy, zz] == groupseg[grouplabel[i]]:
                    overlapgroupcounts[grouplabel[i], volume[xx, yy, zz]] += 1
            groupcounts[grouplabel[i]] += 1

        groupcate = np.argmax(overlapgroupcounts, axis=1)
        maxoverlapgroupcounts = np.max(overlapgroupcounts, axis=1)
        curr_max = np.max(volume)
        for i in range(groupcate.shape[0]):
            if maxoverlapgroupcounts[i] < 7 and groupcounts[i] > 12:
                curr_max += 1
                groupcate[i] = curr_max

        finalgrouplabel = -1 * np.ones(pts.shape[0])
        for i in range(pts.shape[0]):
            if grouplabel[i] != -1 and volume[x[i], y[i], z[i]] == -1:
                volume[x[i], y[i], z[i]] = groupcate[grouplabel[i]]
                volume_seg[x[i], y[i], z[i]] = groupseg[grouplabel[i]]
                finalgrouplabel[i] = groupcate[grouplabel[i]]
        return finalgrouplabel

