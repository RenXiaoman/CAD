import argparse
from pathlib import Path

import torch

from model_from_plans import (
    DEFAULT_PLANS,
    build_generic_unet_from_plans,
    output_shapes,
    set_inference_mode,
)


def parse_shape(text):
    values = [int(i) for i in text.replace(",", " ").split()]
    if len(values) != 5:
        raise argparse.ArgumentTypeError("shape must be 5 integers: B C D H W")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", default=str(DEFAULT_PLANS))
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--shape", type=parse_shape, default=None, help="example: '1 1 16 256 256'")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deep-supervision", action="store_true")
    args = parser.parse_args()

    network, meta, _ = build_generic_unet_from_plans(
        plans_file=Path(args.plans),
        stage=args.stage,
        deep_supervision=args.deep_supervision,
    )

    if not args.deep_supervision:
        set_inference_mode(network)
    network.to(args.device)

    if args.shape is None:
        shape = [args.batch_size, meta["input_channels"], *meta["patch_size"]]
    else:
        shape = args.shape

    x = torch.randn(shape, device=args.device)
    with torch.no_grad():
        y = network(x)

    print("output shape(s):", output_shapes(y))


if __name__ == "__main__":
    main()
