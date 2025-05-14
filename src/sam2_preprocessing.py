from napari_dinosim.utils import SAM2Processor, load_image, get_nhwc_image
import os
import torch
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from glob import glob


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM2 masks for images."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sam2_masks",
        help="Directory to save the generated masks (default: sam2_masks).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="tiny",
        help="SAM2 model type (e.g., 'tiny', 'base', 'large') (default: 'tiny').",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=16,
        help="Points per side for SAM2 model (default: 16).",
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="0",
        help="CUDA device to use (e.g., '0', '1', or 'cpu') (default: '0').",
    )

    args = parser.parse_args()

    if args.cuda_device == "cpu":
        device = torch.device("cpu")
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = (
            torch.device(f"cuda:{args.cuda_device}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    print(f"Using device: {device}")

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = []
    if input_path.is_file():
        image_files.append(input_path)
    elif input_path.is_dir():
        print(f"Searching for images in {input_path}...")
        image_files.extend(
            [
                Path(img_path)
                for img_path in glob(os.path.join(str(input_path), "*.*"))
            ]
        )
    else:
        print(
            f"Error: Input path {input_path} is not a valid file or directory."
        )
        return

    if not image_files:
        print(f"No image files found in {input_path}.")
        return

    print(f"Found {len(image_files)} images to process.")

    try:
        sam2_processor = SAM2Processor(device)
        sam2_processor.load_model(
            args.model_type, points_per_side=args.points_per_side
        )
    except Exception as e:
        print(f"Error initializing SAM2Processor or loading model: {e}")
        return

    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            img_filename = img_path.name
            img = load_image(str(img_path)).astype(np.uint8)
            img_hwc = get_nhwc_image(img)[0]

            sam2_processor.generate_sam_masks(img_hwc)

            output_mask_name = f"sam2_mask_{img_filename}"
            sam2_processor.save_masks(str(output_dir / output_mask_name))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"Mask generation complete. Masks saved in {output_dir}")


if __name__ == "__main__":
    main()
