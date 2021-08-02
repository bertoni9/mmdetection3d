
import numpy as np
import torch

from mmdet3d.core.bbox.structures.cam_box3d import CameraInstance3DBoxes

# class_names = [
#     'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
#     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']


def preprocess(data, result, score_thr=0.0):
    """
    Preprocess data for evaluation and predictions
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

    from mmdet3d.core.bbox import mono_cam_box2vis
    show_bboxes = CameraInstance3DBoxes(
        pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))
    show_bboxes = mono_cam_box2vis(show_bboxes)

    # NMS based on the frontal face
    if show_bboxes:
        show_bboxes, _ = filter_boxes(show_bboxes,  pred_scores, data['img_metas'][0][0]['cam_intrinsic'])
    return show_bboxes


def filter_boxes(bboxes3d, scores, cam_intrinsic):
    import copy
    from mmdet3d.core.bbox import points_cam2img
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
    return bboxes3d[indices], indices


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