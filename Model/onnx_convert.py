import os
import sys

import torch
from Enhanceformer_arch import Enhanceformer


def main_worker():
    model = Enhanceformer()
    load_path = r"net_g_latest.pth"
    load_net = torch.load(load_path,weights_only=True)
    model.load_state_dict(load_net['params'])
    model.eval()

    dummy_input = torch.rand((1, 3, 224, 224))

    onnx_path = "enhanceformer.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        dynamic_axes={"input": {2: "height", 3: "width"}},
    )
    print(f"Model exported to {onnx_path}")

if __name__ == '__main__':
    main_worker()
