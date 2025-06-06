#!/usr/bin/env python3
"""
infer_onnx.py

Usage:
    python infer_onnx.py \
        --onnx_path path/to/adaattn.onnx \
        --context_path path/to/context.jpg \
        --style_path path/to/style.jpg \
        --output_path path/to/output.jpg

Loads a context image and a style image, runs the ONNX AdaAttN model with ONNX Runtime,
and saves the stylized output image.
"""

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on an AdaAttN ONNX model"
    )
    parser.add_argument(
        "--onnx_path", required=True,
        help="Path to the exported ONNX model file"
    )
    parser.add_argument(
        "--context_path", required=True,
        help="Path to the context (content) image"
    )
    parser.add_argument(
        "--style_path", required=True,
        help="Path to the style image"
    )
    parser.add_argument(
        "--output_path", required=True,
        help="Where to save the stylized output image"
    )
    return parser.parse_args()


def load_image(path):
    """
    Load an image from disk, convert to RGB, normalize to [0,1],
    and reshape to (1,3,H,W) float32 numpy.
    """
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    # H x W x C -> C x H x W -> 1 x C x H x W
    img_np = np.transpose(img_np, (2, 0, 1))[None, ...]
    return img_np


def save_image(img_np, path):
    """
    img_np: H x W x C, float32 in [0,1]
    Saves to disk as uint8 JPEG/PNG.
    """
    img_np = np.clip(img_np, 0.0, 1.0)
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)
    img_pil = Image.fromarray(img_uint8)
    img_pil.save(path)


def main():
    args = parse_args()

    # 1. Load ONNX model into ONNX Runtime
    sess_options = ort.SessionOptions()
    # Enable optimizations if you like:
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Try GPU first, fallback to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.onnx_path, sess_options, providers=providers)

    # 2. Prepare inputs
    context_np = load_image(args.context_path)
    style_np = load_image(args.style_path)

    # 3. Map inputs by name
    input_meta = session.get_inputs()
    # Expect exactly two inputs: "context" and "style"
    name0 = input_meta[0].name
    name1 = input_meta[1].name
    feed = {
        name0: context_np,
        name1: style_np
    }

    # 4. Run inference
    outputs = session.run(None, feed)
    # Assume single output
    out_np = outputs[0]  # shape: (1, 3, H, W)

    # 5. Postprocess and save
    # N x C x H x W -> C x H x W -> H x W x C
    out_img = np.squeeze(out_np, axis=0)
    out_img = np.transpose(out_img, (1, 2, 0))
    save_image(out_img, args.output_path)

    print(f"Stylized image written to {args.output_path}")


if __name__ == "__main__":
    main()
