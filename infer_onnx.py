#!/usr/bin/env python3

import argparse
import numpy as np
from PIL import Image
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on an AdaAttN ONNX model"
    )
    parser.add_argument(
        "--onnx_path", default="adaattn.onnx", required=False,
        help="Path to the exported ONNX model file"
    )
    parser.add_argument(
        "--content_path", default="datasets/contents/0420a8ec521813e6f13c4a89cd20761a.jpg", required=False,
        help="Path to the content (content) image"
    )
    parser.add_argument(
        "--style_path", default="datasets/styles/96fc8a810fb4f1607c252128aec5b563f99b438d.jpg@600w_600h_1c.png", required=False,
        help="Path to the style image"
    )
    parser.add_argument(
        "--output_path", default="result/output.jpg", required=False,
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

    session = ort.InferenceSession(args.onnx_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])

    content_np = load_image(args.content_path)
    style_np = load_image(args.style_path)

    input_meta = session.get_inputs()

    name0 = input_meta[0].name
    name1 = input_meta[1].name
    feed = {
        name0: content_np,
        name1: style_np
    }

    outputs = session.run(None, feed)
    out_np = outputs[0]  # shape: (1, 3, H, W)

    # N x C x H x W -> C x H x W -> H x W x C
    out_img = np.squeeze(out_np, axis=0)
    out_img = np.transpose(out_img, (1, 2, 0))
    save_image(out_img, args.output_path)

    print(f"Stylized image written to {args.output_path}")

if __name__ == "__main__":
    main()
