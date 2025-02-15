from os import path
from argparse import ArgumentParser

import torch

from torchvision.io import decode_image
from torchvision.transforms.v2 import ToDtype, ToPILImage

from model import SuperCool


def main():
    parser = ArgumentParser(description="Super-resolution upscaling script")

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--device", default="cuda", type=str)

    args = parser.parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        raise RuntimeError("Cuda is not available.")

    checkpoint = torch.load(
        args.checkpoint_path, map_location=args.device, weights_only=True
    )

    model = SuperCool(**checkpoint["model_args"])

    print("Compiling model")
    model = torch.compile(model)

    model = model.to(args.device)

    model.load_state_dict(checkpoint["model"])

    print("Model checkpoint loaded successfully")

    model.remove_weight_norms()

    image_to_tensor = ToDtype(torch.float32, scale=True)
    tensor_to_image = ToPILImage()

    image = decode_image(args.image_path, mode="RGB")

    x = image_to_tensor(image).unsqueeze(0).to(args.device)

    model.eval()

    print("Upscaling ...")

    with torch.no_grad():
        y_pred = model(x)

    image = tensor_to_image(y_pred.squeeze())

    image.show()


if __name__ == "__main__":
    main()
