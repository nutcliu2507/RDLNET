# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmedit.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[250, 250],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    elif len(args.shape) in [3, 4]:  # 4 for video inputs (t, c, h, w)
        input_shape = tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported '
            f'with {model.__class__.__name__}')


    import time
    import torch

    # t = 10
    t = 10
    repeat_time = 10
    warm_up = 5
    infer_time = 0

    with torch.no_grad():
        # x = torch.rand(1, t, 3, 180, 320).cuda()
        x = torch.rand(1, t, 3, 128, 128).cuda()

        for i in range(repeat_time):
            if i < warm_up:
                infer_time = 0
            start_time = time.time()
            y = model(x)
            infer_time += (time.time() - start_time)

    print(y.shape, infer_time / (repeat_time - warm_up + 1) / t)


if __name__ == '__main__':
    main()
