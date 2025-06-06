# Copyright 2025 by Andrey Ignatov. All Rights Reserved.

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import torch.nn as nn
import torch

# Install ai_edge_torch_plugin
# https://pypi.org/project/ai-edge-torch/
# This plugin is developed by TFLite team and allows direct Torch to TFLite model conversion
# Note: this plugin is available only for Linux systems.
# However, it works perfectly under Windows WSL2 simply create a new conda environment
# and install this package along with torch / torchvision libs

import ai_edge_torch
from Enhanceformer_arch import Enhanceformer

if __name__ == '__main__':

    # Creating / loading pre-trained UNet model

    model = Enhanceformer()
    model.load_state_dict(torch.load(r"net_g_latest.pth", weight_only=True)['params'])

    # Converting model to TFLite

    sample_input = (torch.randn(1, 3, 1024, 1024),)

    edge_model = ai_edge_torch.convert(model.eval(), sample_input)
    edge_model.export("enhanceformer.tflite")