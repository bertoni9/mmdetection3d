from argparse import ArgumentParser
import glob
import os
from os import path as osp

import mmcv
from mmdet3d.apis import (inference_mono_3d_detector, init_model)
from mmdet3d.core import show_multi_modality_result
from process import postprocess, generate_txt, save_json


def main():
    parser = ArgumentParser()
    parser.add_argument('images', nargs='*', help='input images')
    parser.add_argument('--ann', help='ann file', default='demo/data/wv/test.json')
    parser.add_argument('--config', help='Config file',
                        default='configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth')
    parser.add_argument(
        '--predict', action='store_true', help='run predictions')
    parser.add_argument(
        '--generate', action='store_true', help='create txt file in KITTI format')
    parser.add_argument(
        '--save', action='store_true', help='save json files')
    parser.add_argument(
        '--cf', type=float, default=1., help='corrective factor to scale annotations')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--glob', help='glob expression for input images (for many images)')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visuliaztion results')
    args = parser.parse_args()

    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")
    assert args.generate or args.predict or args.save, "expected either inference or generation of txt/json files"
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image

    for image in args.images:
        print(f'Running inference on image {image}')
        result, data = inference_mono_3d_detector(model, image, args.ann)
        assert 'img' in data.keys(), 'image data is not provided for visualization'

        img_filename = data['img_metas'][0][0]['filename']
        file_name = osp.split(img_filename)[-1].split('.')[0]
        # read from file because img in data_dict has undergone pipeline transform
        img = mmcv.imread(img_filename)
        boxes_3d, boxes_2d, categories = postprocess(data, result, score_thr=args.score_thr)

        if args.generate:
            generate_txt(boxes_3d, boxes_2d, categories, args.out_dir, args.cf, img_filename)
        # show the results
        if args.save:
            output_path = osp.join(args.out_dir, file_name + '.json')
            save_json(boxes_3d, boxes_2d, categories, data['img_metas'][0][0]['cam_intrinsic'], args.cf, output_path)
        if args.predict and boxes_3d:
            show_multi_modality_result(
                img,
                None,
                boxes_3d,
                data['img_metas'][0][0]['cam_intrinsic'],
                args.out_dir,
                file_name,
                box_mode='camera',
                show=args.show)


if __name__ == '__main__':
    main()
