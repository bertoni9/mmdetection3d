from argparse import ArgumentParser
import glob
from os import path as osp

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)


def main():
    parser = ArgumentParser()
    parser.add_argument('images', nargs='*', help='input images')
    parser.add_argument('--ann', help='ann file', default='demo/data/wayve/test.json')
    parser.add_argument('--config', help='Config file',
                        default='configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default='checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--glob', help='glob expression for input images (for many images)')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show', action='store_true', help='show online visuliaztion results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visuliaztion results')
    args = parser.parse_args()

    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image

    for image in args.images:
        print(f'Running inference on image {image}')
        result, data = inference_mono_3d_detector(model, image, args.ann)
        # show the results
        show_result_meshlab(
            data,
            result,
            args.out_dir,
            args.score_thr,
            show=args.show,
            snapshot=args.snapshot,
            task='mono-det')


if __name__ == '__main__':
    main()
