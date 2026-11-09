#!/usr/bin/env python3
"""Convert Silent-Face-Anti-Spoofing (MiniFASNet, Apache-2.0) to CoreML.

Inputs are pinned and checksum-verified by scripts/fetch-model.sh. Weights are
loaded with weights_only=True, so the .pth pickle cannot execute code.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import subprocess

import coremltools as ct
import numpy as np
import torch

MODELS = [
    ("FASV2", "MiniFASNetV2", "2.7_80x80_MiniFASNetV2.pth"),
    ("FASV1SE", "MiniFASNetV1SE", "4_0_0_80x80_MiniFASNetV1SE.pth"),
]


class WithSoftmax(torch.nn.Module):
    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("minifasnet", args.source / "MiniFASNet.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name, constructor, weights in MODELS:
        net = getattr(module, constructor)(conv6_kernel=(5, 5))
        state = torch.load(args.source / weights, map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        wrapped = WithSoftmax(net).eval()

        example = torch.rand(1, 3, 80, 80) * 255
        model = ct.convert(
            torch.jit.trace(wrapped, example),
            inputs=[ct.TensorType(name="image", shape=(1, 3, 80, 80), dtype=np.float32)],
            outputs=[ct.TensorType(name="probs")],
            minimum_deployment_target=ct.target.macOS14,
            compute_precision=ct.precision.FLOAT32,
        )
        package = args.out / f"{name}.mlpackage"
        model.save(str(package))

        difference = np.abs(wrapped(example).detach().numpy() - model.predict({"image": example.numpy()})["probs"]).max()
        print(f"{name}: CoreML vs PyTorch max difference {difference:.2e}")
        if difference > 1e-3:
            raise SystemExit(f"{name}: parity check failed")
        subprocess.run(["xcrun", "coremlc", "compile", str(package), str(args.out)], check=True, capture_output=True)


if __name__ == "__main__":
    main()
