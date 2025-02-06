#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
File: convert.py
Description: Convert DepthNet model to ONNX format.
Author: Sun Hao
Contact: smartadpole@163.com
Time: 2023/3/9 18:32
"""

import sys, os
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

import argparse
import numpy as np
import torch
from export_onnx.onnx_test import test_dir
from utils.file import MkdirSimple

# Adjust the import path according to your project structure
from models.depth_net import DepthNet
import time

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

class WarpPoseModel(torch.nn.Module):
    """
    A wrapper model to adapt the input format for DepthNet.
    Accepts a single image tensor of shape [B, 3, H, W] (values in [0, 255])
    and constructs the required inputs for DepthNet (poses, image sequence, intrinsics).
    """
    def __init__(self, ckpt_dir, width, height):
        super(WarpPoseModel, self).__init__()
        self.image_width = width
        self.image_height = height
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = DepthNet(
            ckpt_dir=ckpt_dir,
            device=device,
            image_size=(height, width),
            backbone={'extractor': 'down_sample'},
            mode='avg',
            seq_len=2,
            downscale=1
        )
        self.model.to(device)
        self.model.eval()

    def forward(self, image):
        batch_size = 1
        seq_len = 2  # Must match the seq_len provided during initialization
        device = image.device

        # Create fake poses (zeros imply identity transformation)
        poses = torch.zeros(batch_size, seq_len, 7, device=device)
        poses[..., 3] = 1.0  # 设置四元数的实部为 1
        # Replicate the input image seq_len times
        images = image.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        # Construct intrinsics: fx = width/2, fy = height/2, cx = width/2, cy = height/2
        fx = self.image_width / 2.0
        fy = self.image_height / 2.0
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        intrinsics = torch.tensor([fx, fy, cx, cy], device=device).unsqueeze(0).repeat(batch_size, 1)

        data = {
            'poses': poses,
            'images': images,
            'intrinsics': intrinsics
        }

        outputs = self.model(data)
        depth = outputs['depths'][-1]
        return depth

def export_to_onnx(model_path, onnx_file, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, device="cuda", opset_version=16):
    """
    Export the DepthNet model to ONNX format.
    :param model_path: Path to the trained model weights (passed to DepthNet as ckpt_dir)
    :param onnx_file: File path to save the ONNX model.
    :param width: Input image width.
    :param height: Input image height.
    :param device: Device to run the model.
    :param opset_version: ONNX opset version.
    """
    device = torch.device(device)
    model = WarpPoseModel(model_path, width, height).to(device)
    model.eval()

    for name, param in model.named_parameters():
        print(f"Parameter {name} is on device {param.device}")
        break
    dummy_input = (torch.rand(1, 3, height, width, device=device) * 255.0).float()
    print(f"dummy_input is on device {dummy_input.device}, shape={dummy_input.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['depth_output']
    )
    print(f"Model exported to {onnx_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Export DepthNet model to ONNX format")
    parser.add_argument("--model", type=str, default="",
                        help="Path to the trained model directory or weight file")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the exported ONNX model")
    parser.add_argument("--image", type=str, required=False,
                        help="Path to an input image for testing the model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH,
                        help="Input image width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT,
                        help="Input image height")
    parser.add_argument("--test", action="store_true",
                        help="Test the exported ONNX model")
    return parser.parse_args()

def main():
    args = parse_args()
    opset_version = 16

    if args.model:
        model_name = "_".join(args.model.split("/")[-3:]).replace("ckpts", "").replace("=", "-").strip('_')
        model_name = os.path.splitext(model_name)[0].split("-val")[0]
    else:
        model_name = "model"
    output_dir = os.path.join(args.output, model_name, f'{args.width}_{args.height}')
    onnx_file = os.path.join(output_dir, f'DepthNet_{args.width}_{args.height}_{model_name}_{opset_version}.onnx')
    MkdirSimple(output_dir)

    export_to_onnx(args.model, onnx_file, args.width, args.height, args.device, opset_version=opset_version)
    if args.test:
        test_dir(onnx_file, [args.image], output_dir)

if __name__ == "__main__":
    main()
