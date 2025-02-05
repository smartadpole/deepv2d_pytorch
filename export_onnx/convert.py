#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: convert_onnx.py
@time: 2023/3/9 下午6:32
@desc:
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))
import numpy as np
import torch
import os
import argparse
from utils.file import MkdirSimple
from export_onnx.onnx_test import test_dir
from models.depth_net import DepthNet

W = 640
H = 480


class OptAttributes:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX format")
    parser.add_argument("--model", type=str, required=False, help="Path to the trained model.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    parser.add_argument("--image", type=str, required=False, help="Path to the input image.")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="Device to run the model on.")
    parser.add_argument("--width", type=int, default=W, help="Width of the input image.")
    parser.add_argument("--height", type=int, default=H, help="Height of the input image.")
    parser.add_argument("--test", action="store_true", help="test model")
    return parser.parse_args()


def get_config():
    inv_K_pool = {
        (480, 640): [[0.0017, 0.0000, -0.5, 0.0000],
                     [0.0000, 0.0017, -0.5, 0.0000],
                     [0.0000, 0.0000,  1.0, 0.0000],
                     [0.0000, 0.0000,  0.0, 1.0000]],
        (240, 320): [[0.0035, 0.0000, -0.5, 0.0000],
                     [0.0000, 0.0035, -0.5, 0.0000],
                     [0.0000, 0.0000,  1.0, 0.0000],
                     [0.0000, 0.0000,  0.0, 1.0000]],
        (120, 160): [[0.0069, 0.0000, -0.5, 0.0000],
                     [0.0000, 0.0069, -0.5, 0.0000],
                     [0.0000, 0.0000,  1.0, 0.0000],
                     [0.0000, 0.0000,  0.0, 1.0000]],
        (60, 80): [[0.0139, 0.0000, -0.556, 0.0000],
                   [0.0000, 0.0139, -0.5, 0.0000],
                   [0.0000, 0.0000,  1.0, 0.0000],
                   [0.0000, 0.0000,  0.0, 1.0000]],
        (30, 40): [[0.0278, 0.0000, -0.556, 0.0000],
                   [0.0000, 0.0278, -0.5, 0.0000],
                   [0.0000, 0.0000,  1.0, 0.0000],
                   [0.0000, 0.0000,  0.0, 1.0000]],
        (15, 20): [[0.0556, 0.0000, -0.556, 0.0000],
                   [0.0000, 0.0556, -0.5, 0.0000],
                   [0.0000, 0.0000,  1.0, 0.0000],
                   [0.0000, 0.0000,  0.0, 1.0000]]
    }

    proj_mats = [
        {
            (480, 640): [[-239.94, 0.0000, 0.0000, 0.0000],
                         [0.0000, -239.94, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (240, 320): [[-119.97, 0.0000, 0.0000, 0.0000],
                         [0.0000, -119.97, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (120, 160): [[-5.9984e+01, 0.0000, 0.0000, 0.0000],
                         [0.0000, -5.9984e+01, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (60, 80): [[-2.9992e+01, 0.0000, 0.0000, 0.0000],
                       [0.0000, -2.9992e+01, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (30, 40): [[-14.9960, 0.0000, 0.0000, 0.0000],
                       [0.0000, -14.9960, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (15, 20): [[-7.4980, -18.4777, -5.1143, 123.7495],
                       [-2.3994, -10.9992, -3.5678, 92.3475],
                       [-1.1997, -5.9984, -1.8765, 63.8457],
                       [0.0000, 0.0000, 0.0000, 1.0000]]
        },
        {
            (480, 640): [[-239.94, 0.0000, 0.0000, 0.0000],
                         [0.0000, -239.94, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (240, 320): [[-119.97, 0.0000, 0.0000, 0.0000],
                         [0.0000, -119.97, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (120, 160): [[-5.9984e+01, 0.0000, 0.0000, 0.0000],
                         [0.0000, -5.9984e+01, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (60, 80): [[-2.9992e+01, 0.0000, 0.0000, 0.0000],
                       [0.0000, -2.9992e+01, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (30, 40): [[-14.9960, 0.0000, 0.0000, 0.0000],
                       [0.0000, -14.9960, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (15, 20): [[-7.4980, -18.4777, -5.1143, 123.7495],
                       [-2.3994, -10.9992, -3.5678, 92.3475],
                       [-1.1997, -5.9984, -1.8765, 63.8457],
                       [0.0000, 0.0000, 0.0000, 1.0000]]
        },
        {
            (480, 640): [[-239.94, 0.0000, 0.0000, 0.0000],
                         [0.0000, -239.94, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (240, 320): [[-119.97, 0.0000, 0.0000, 0.0000],
                         [0.0000, -119.97, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (120, 160): [[-5.9984e+01, 0.0000, 0.0000, 0.0000],
                         [0.0000, -5.9984e+01, 0.0000, 0.0000],
                         [0.0000, 0.0000, 1.0, 0.0000],
                         [0.0000, 0.0000, 0.0, 1.0000]],
            (60, 80): [[-2.9992e+01, 0.0000, 0.0000, 0.0000],
                       [0.0000, -2.9992e+01, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (30, 40): [[-14.9960, 0.0000, 0.0000, 0.0000],
                       [0.0000, -14.9960, 0.0000, 0.0000],
                       [0.0000, 0.0000, 1.0, 0.0000],
                       [0.0000, 0.0000, 0.0, 1.0000]],
            (15, 20): [[-7.4980, -18.4777, -5.1143, 123.7495],
                       [-2.3994, -10.9992, -3.5678, 92.3475],
                       [-1.1997, -5.9984, -1.8765, 63.8457],
                       [0.0000, 0.0000, 0.0000, 1.0000]]
        }
    ]

    return proj_mats, inv_K_pool


class WarpPoseModel(torch.nn.Module):
    def __init__(self, file, width, height):
        super(WarpPoseModel, self).__init__()

        self.model = DepthNet(image_size=(width, height), backbone={'extractor': 'down_sample'},
                              mode='avg', seq_len=2, downscale=1).cuda()
        device = torch.device("cuda")
        self.model.to(device)
        self.model.eval()

    def forward(self, image):
        proj_mats, inv_K_pool = get_config()
        inv_K_pool = {k: torch.tensor(v).unsqueeze(0).cuda() for k, v in inv_K_pool.items()}
        proj_mats = [{k: torch.tensor(v).unsqueeze(0).cuda() for k, v in pm.items()} for pm in proj_mats]

        data = {}
        data['poses'] = proj_mats
        data['images'] = [image, image, ]
        data['intrinsics'] = inv_K_pool[480, 640]

        outputs = self.model(image, [image, image, ], proj_mats[0], proj_mats[1:], inv_K_pool)
        depth_pred = outputs
        return depth_pred


def export_to_onnx(model_path, onnx_file, width=W, height=H, device="cuda", version=12):
    opt = OptAttributes(
        multi_view_agg=1,
        robust=False,
        att_rate=4,
        depth_embedding="learned",
        nlabel=32,
        use_skip=1,
        input_scale=0,
        use_unet=True,
        unet_channel_mode="v0",
        num_depth_regressor_anchor=512,
        max_depth=32.0,
        min_depth=0.5,
        inv_depth=1,
        num_frame=2,
        nhead=1,
        height=height,
        width=width,
        pred_conf=0,
        output_scale=2,
    )

    model = WarpPoseModel(model_path, opt)

    # Create dummy input for the model
    dummy_np = np.random.randn(3, height, width).astype(np.float32)
    dummy_input = torch.from_numpy(dummy_np).unsqueeze(0).float().cuda()
    # dummy_input = torch.randn(1, 3, height, width)


    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")
        break
    print(f"dummy_input is on device {dummy_input.device}")
    # Export the model
    torch.onnx.export(model, dummy_input, onnx_file,
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=version,  # the ONNX version to export the model to
                      do_constant_folding=True)

    print(f"Model exported to {onnx_file}")


def main():
    args = parse_args()
    version = 16

    if args.model:
        model_name = "_".join(args.model.split("/")[-3:]).replace("ckpts", "").replace("=", "-").strip('_')
        model_name = os.path.splitext(model_name)[0].split("-val")[0]
    else:
        model_name = "model"
    output = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output, f'MVS2D_{args.width}_{args.height}_{model_name}_{version}.onnx')
    MkdirSimple(output)

    export_to_onnx(args.model, onnx_file, args.width, args.height,
                   args.device, version=version)  # Replace 'vitl' with the desired encoder

    print("export onnx to {}".format(onnx_file))
    if args.test:
        test_dir(onnx_file, [args.image, ], output)


if __name__ == "__main__":
    main()
