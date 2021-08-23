
import math
import json
import os.path as osp
import copy
from collections import defaultdict

import numpy as np
import torch

from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes
from mmdet3d.core.bbox import mono_cam_box2vis, points_cam2img

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']

cat_map = dict(zip(range(len(class_names)), class_names))


def save_json(boxes_3d, boxes_2d, categories, intrinsics, cf, output_path):
    dic_out = {
        'cuboids': defaultdict(list),
        'intrinsics': intrinsics
    }
    centers = intrinsics_shift(boxes_3d, cf)
    if len(centers) > 0:
        for idx, center in enumerate(centers):
            cuboid = {}
            center = center.tolist()
            cuboid['x'], cuboid['y'], cuboid['z'] = center[0], center[1], center[2]
            dim = boxes_3d.dims[idx]
            cuboid['h'], cuboid['w'], cuboid['l'] = float(dim[1]), float(dim[2]), float(dim[0])
            cuboid['corrective_factor'] = cf
            cuboid['category'] = cat_map[categories[idx]]  # 23 categories
            box = boxes_2d[idx].tolist()
            dic_out['box'] = box[:-1]
            dic_out['confidence'] = box[-1]
            alpha = float(boxes_3d.yaw[idx])
            yaw = alpha + math.atan2(center[0], center[2])
            alpha = normalize_angle(alpha)  # Networks predict allocentric angle alpha
            yaw = normalize_angle(yaw)
            cuboid['alpha'] = alpha
            cuboid['yaw'] = yaw
            dic_out['cuboids'].append(cuboid)
    with open(output_path, 'w') as ff:
        json.dump(dic_out, ff)
    print(f'Saved file {output_path}')


def generate_txt(boxes_3d, boxes_2d, categories, out_dir, filename):

    path_txt = osp.join(out_dir, osp.splitext(osp.basename(filename))[0] + '.txt')
    with open(path_txt, "w+") as ff:
        if not boxes_3d:
            return
        locs = boxes_3d.gravity_center
        dims = boxes_3d.dims
        alphas = boxes_3d.yaw
        yaws = alphas + torch.atan2(locs[:, 0], locs[:, 2])
        for idx, box in enumerate(boxes_2d):
            loc = locs[idx].tolist()
            dim = dims[idx].tolist()
            hwl = [dim[1], dim[2], dim[0]]
            alpha = normalize_angle((alphas[idx]))  # KITTI Convention
            yaw = normalize_angle((yaws[idx]))
            conf = float(box[-1])
            box = box[:-1].tolist()
            cat = cat_map[categories[idx]]
            output_list = [alpha] + box + hwl + loc + [yaw, conf]
            ff.write("%s " % cat)
            ff.write("%i %i " % (-1, -1))
            for el in output_list:
                ff.write("%f " % el)
            ff.write("\n")
    print(f"Saved file {path_txt}")


def postprocess(data, result, score_thr=0.0):
    """
    Postprocess data for evaluation and predictions
    """

    if 'pts_bbox' in result[0].keys():
        result[0] = result[0]['pts_bbox']
    elif 'img_bbox' in result[0].keys():
        result[0] = result[0]['img_bbox']
    pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
    pred_scores = result[0]['scores_3d'].numpy()
    categories = result[0]['labels_3d'].numpy()
    # filter out low score bboxes for visualization

    if score_thr > 0:
        inds = pred_scores > score_thr
        # inds = np.logical_and(pred_scores > score_thr, categories == 0)
        pred_bboxes = pred_bboxes[inds]
        pred_scores = pred_scores[inds]
        categories = categories[inds]

    if 'cam_intrinsic' not in data['img_metas'][0][0]:
        raise NotImplementedError(
            'camera intrinsic matrix is not provided')

    show_bboxes = CameraInstance3DBoxes(
        pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))
    show_bboxes = mono_cam_box2vis(show_bboxes)

    # NMS based on the frontal face
    boxes_3d, boxes_2d, indices = filter_boxes(show_bboxes, pred_scores, data['img_metas'][0][0]['cam_intrinsic'])
    return boxes_3d, boxes_2d, categories[indices]


def filter_boxes(bboxes3d, scores, cam_intrinsic):
    if not bboxes3d:
        return [], [], []
    cam_intrinsic = copy.deepcopy(cam_intrinsic)
    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    if not isinstance(cam_intrinsic, torch.Tensor):
        cam_intrinsic = torch.from_numpy(np.array(cam_intrinsic))
    cam_intrinsic = cam_intrinsic.reshape(3, 3).float().cpu()

    # project to 2d to get image coords (uv)
    uv_origin = points_cam2img(points_3d, cam_intrinsic)
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()
    boxes = np.array(
        [[np.min(pt[:, 0]), np.min(pt[:, 1]), np.max(pt[:, 0]), np.max(pt[:, 1]), scores[idx]]
         for idx, pt in enumerate(imgfov_pts_2d)]
    )
    indices = nms(boxes, thresh=0.5)
    return bboxes3d[indices], boxes[indices], indices


def nms(dets, thresh):

    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def intrinsics_shift(bboxes3d, cf):
    if not bboxes3d:
        return torch.empty(0)
    centers = copy.deepcopy(bboxes3d.gravity_center)
    centers[:, 0] *= cf
    return centers


def normalize_angle(ori):
    while ori > np.pi:
        ori -= 2 * np.pi
    while ori < -np.pi:
        ori += 2 * np.pi
    assert -np.pi <= ori <= np.pi
    return ori