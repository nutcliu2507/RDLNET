# Copyright (c) OpenMMLab. All rights reserved.
from .matting_inference import init_model
from .restoration_video_inference import restoration_video_inference
from .test import multi_gpu_test, single_gpu_test
from .train import set_random_seed, train_model

__all__ = [
    'train_model', 'set_random_seed', 'init_model',
    'multi_gpu_test', 'single_gpu_test', 'restoration_video_inference'
]
