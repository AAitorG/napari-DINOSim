import os
import torch
import numpy as np
from typing import Union
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


class SAM2Processor:
    """
    A class to handle SAM2 model loading, mask generation and processing.
    """

    DEFAULT_CHECKPOINTS = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "base": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    }

    CONFIGS = {
        "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "base": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }

    def __init__(
        self,
        device: Union[str, torch.device] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Initialize the SAM2Processor without loading a model.

        Args:
            device: Device to run the model on
        """
        self.device = (
            device
            if isinstance(device, torch.device)
            else torch.device(device)
        )

        # Initialize model to None, will be loaded on demand
        self.model_type = None
        self.model_cfg = None
        self.checkpoint_path = None
        self.sam2_model = None
        self.mask_generator = None

        self.sam2_predictions = None

    def _delete_predictions(self):
        self.sam2_predictions = None

    def _download_checkpoint(
        self, model_type: str = None, models_dir: str = "sam2_models"
    ):
        """
        Download the SAM2 checkpoint.

        Args:
            model_type: Type of the model to download (one of the keys in DEFAULT_CHECKPOINTS)
            models_dir: Directory to save the model checkpoint
        """
        import urllib.request
        from tqdm import tqdm

        # Use the instance's model_type if none is specified
        if model_type is None:
            model_type = self.model_type

        # Create the checkpoints directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)

        # Set checkpoint path
        checkpoint_path = os.path.join(models_dir, f"{model_type}.pt")

        # Get URL for the specified model type
        url = self.DEFAULT_CHECKPOINTS.get(model_type)
        if not url:
            raise ValueError(
                f"Model type '{model_type}' not recognized. Available types: {list(self.DEFAULT_CHECKPOINTS.keys())}"
            )

        print(
            f"Downloading checkpoint for {model_type} from {url} to {checkpoint_path}"
        )

        # Create a progress bar
        class DownloadProgressBar(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        # Download with progress bar
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(
                url, checkpoint_path, reporthook=t.update_to
            )

        print(f"\nDownload complete. Checkpoint saved to {checkpoint_path}")

        return checkpoint_path

    def exist_predictions(self):
        """Check if SAM2 predictions (masks) are available.

        Returns
        -------
        bool
            True if SAM2 predictions are loaded, False otherwise
        """
        return self.sam2_predictions is not None

    def load_model(
        self, model_type: str = "large", models_dir: str = "sam2_models"
    ):
        """
        Load the SAM2 model and create the mask generator.

        Args:
            model_type: Type of the model to use (one of the keys in CONFIGS)
            models_dir: Directory to load/save the model checkpoint
        """
        # Clean up existing model if loaded
        if self.sam2_model is not None:
            # Delete references to existing model and mask generator
            del self.mask_generator
            del self.sam2_model

            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            self.sam2_model = None
            self.mask_generator = None

        # Set model type and config
        self.model_type = model_type
        if model_type not in self.CONFIGS:
            raise ValueError(
                f"Model type '{model_type}' not recognized. Available types: {list(self.CONFIGS.keys())}"
            )
        self.model_cfg = self.CONFIGS[model_type]

        # Determine checkpoint path
        self.checkpoint_path = os.path.join(models_dir, f"{model_type}.pt")

        # Check if checkpoint exists and download if needed
        if not os.path.exists(self.checkpoint_path):
            self.checkpoint_path = self._download_checkpoint(
                model_type, models_dir
            )

        print(f"Loading SAM2 model {model_type} from {self.checkpoint_path}")
        self.sam2_model = build_sam2(
            self.model_cfg,
            self.checkpoint_path,
            device=self.device,
            apply_postprocessing=False,
        )

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=16,  # 64
            points_per_batch=128,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )
        print(f"SAM2 model {model_type} loaded successfully")

    def _remove_overlapping(self, masks: list[np.ndarray]) -> list[np.ndarray]:
        """
        Process masks to remove overlapping regions from larger masks.

        Args:
            masks: List of binary masks as numpy arrays

        Returns:
            List of processed masks
        """
        # Work on a copy of the masks so that the originals remain unchanged
        processed_masks = [mask.copy() for mask in masks]
        n_masks = len(processed_masks)

        # Compare each pair of masks
        for i in range(n_masks):
            for j in range(i + 1, n_masks):
                mask_i = processed_masks[i]
                mask_j = processed_masks[j]

                # Determine if there is any overlap (logical AND)
                overlap = mask_i & mask_j
                if np.any(overlap):
                    # Compute the "area" (i.e. count of True pixels) for each mask
                    area_i = np.sum(mask_i)
                    area_j = np.sum(mask_j)

                    if area_i > area_j:
                        # Subtract mask_j from mask_i: wherever mask_j is True, set mask_i to False
                        processed_masks[i] = mask_i & (~mask_j)
                    elif area_j > area_i:
                        # Subtract mask_i from mask_j
                        processed_masks[j] = mask_j & (~mask_i)
                    # If equal areas, leave unchanged

        return processed_masks

    def _ensure_rgb_image(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a grayscale image to RGB if needed.

        Args:
            image: Input image as numpy array

        Returns:
            RGB image as numpy array
        """
        # Check if the image is grayscale (2D or 3D with 1 channel)
        if len(image.shape) == 2 or (
            len(image.shape) == 3 and image.shape[2] == 1
        ):
            # Convert to 3-channel RGB
            if len(image.shape) == 2:
                return np.repeat(image[..., np.newaxis], 3, axis=2)
            else:
                return np.repeat(image, 3, axis=2)
        return image

    def _touint8(self, image: np.ndarray) -> np.ndarray:
        """Convert image to uint8 format with proper normalization.

        Parameters
        ----------
        image : np.ndarray
            Input image array. Can be float (0-1 or arbitrary range) or int.

        Returns
        -------
        np.ndarray
            Converted uint8 image with values 0-255.
        """
        if image.dtype != np.uint8:
            if image.min() >= 0 and image.max() <= 255:
                pass
            else:
                if not (0 <= image.min() <= 1 and 0 <= image.max() <= 1):
                    image = image - image.min()
                    image = image / image.max()
                image = image * 255
        return image.astype(np.uint8)

    def generate_sam_masks(
        self, image: np.ndarray, model_size: str = "large"
    ) -> None:
        """
        Generate masks for a single image using SAM2 and store them in self.sam2_predictions.

        Args:
            image: Input image as numpy array
            model_size: Size of the SAM2 model to use (tiny, small, base, large)
        """
        # Ensure model is loaded
        if self.mask_generator is None:
            self.load_model(model_size)

        # Ensure the image is RGB
        image = self._touint8(image)
        rgb_image = self._ensure_rgb_image(image)

        # Generate masks and store them in self.sam2_predictions
        masks = self.mask_generator.generate(rgb_image)
        # get only the masks, ignore the rest (bbox, area, pred_iou, etc.)
        sam2_predictions = [mask["segmentation"] for mask in masks]
        sam2_predictions = self._remove_overlapping(sam2_predictions)
        self.sam2_predictions = torch.tensor(sam2_predictions, device="cpu")

    def save_masks(self, filepath: str) -> None:
        """
        Save the generated SAM2 masks to a torch file.

        Args:
            filepath: Path where to save the masks

        Raises:
            ValueError: If no SAM2 predictions exist to save
        """
        if self.sam2_predictions is None:
            raise ValueError("No SAM2 predictions exist to save")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Save predictions to file
        torch.save(
            {
                "sam2_predictions": self.sam2_predictions,
            },
            filepath,
        )
        print(f"SAM2 masks saved to {filepath}")

    def load_masks(self, filepath: str) -> None:
        """
        Load SAM2 masks from a torch file.

        Args:
            filepath: Path to the saved masks file

        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the loaded file doesn't contain valid SAM2 predictions
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Load predictions from file
        checkpoint = torch.load(filepath, map_location="cpu")

        if "sam2_predictions" not in checkpoint:
            raise ValueError(f"Invalid SAM2 masks file: {filepath}")

        self.sam2_predictions = checkpoint["sam2_predictions"]
        print(f"SAM2 masks loaded from {filepath}")

    def refine_prediction_with_sam_masks(
        self, coarse_prediction, pred_obj_white: bool = False
    ) -> torch.Tensor:
        """
        Refine masks using a coarse_prediction.

        Args:
            coarse_prediction: Prediction tensor
            pred_obj_white: Whether predicted objects are white (True) or black (False)

        Returns:
            Refined prediction tensor
        """

        # check pred is torch tensor
        if not isinstance(coarse_prediction, torch.Tensor):
            coarse_prediction = torch.tensor(
                coarse_prediction, device=self.device
            )

        if self.sam2_predictions is None:
            raise ValueError(
                "No SAM2 predictions available. Call generate_sam_masks first or load masks from a file."
            )

        # Create initial mask
        refined_mask = (
            torch.zeros(coarse_prediction.shape, device=self.device)
            if pred_obj_white
            else torch.ones(coarse_prediction.shape, device=self.device)
        )

        # Process each mask
        for mask in self.sam2_predictions:
            if isinstance(mask, np.ndarray):
                # Convert mask to tensor
                mask_tensor = torch.tensor(mask, device=self.device)
            else:
                mask_tensor = mask.to(device=self.device)

            # Calculate average prediction value within the mask
            mask_sum = mask_tensor.sum()
            if mask_sum > 0:  # Avoid division by zero
                pixel_value = (
                    coarse_prediction * mask_tensor
                ).sum() / mask_sum
                mask_with_value = mask_tensor * pixel_value

                if not pred_obj_white:
                    # For dark objects, invert the mask
                    mask_with_value = mask_with_value + (
                        1 - mask_tensor.to(torch.uint8)
                    )

                # Update the refined mask
                if pred_obj_white:
                    refined_mask = torch.maximum(refined_mask, mask_with_value)
                else:
                    refined_mask = torch.minimum(refined_mask, mask_with_value)

        return refined_mask.cpu().numpy()

    def get_refined_instances_with_sam_prediction(
        self,
        coarse_prediction: torch.Tensor,
        pred_obj_white: bool = False,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Refine masks using a prediction.

        Args:
            prediction: Prediction tensor
            pred_obj_white: Whether predicted objects are white (True) or black (False)
            threshold: Threshold for determining instances

        Returns:
            Refined prediction tensor
        """
        if self.sam2_predictions is None:
            raise ValueError(
                "No SAM2 predictions available. Call generate_sam_masks first or load masks from a file."
            )

        # Create initial mask
        refined_mask = torch.zeros(coarse_prediction.shape, device=self.device)

        if pred_obj_white:
            th_op = lambda x: x > threshold
        else:
            th_op = lambda x: x < threshold
        n_instance_id = 1
        # Process each mask
        for mask in self.sam2_predictions:
            if isinstance(mask, np.ndarray):
                # Convert mask to tensor
                mask_tensor = torch.tensor(mask, device=self.device)
            else:
                mask_tensor = mask.to(device=self.device)

            # Calculate average prediction value within the mask
            mask_sum = mask_tensor.sum()
            if mask_sum > 0:  # Avoid division by zero
                pixel_value = (
                    coarse_prediction * mask_tensor
                ).sum() / mask_sum
                if th_op(pixel_value):
                    pixel_value = n_instance_id
                    n_instance_id += 1

                    mask_with_value = mask_tensor * pixel_value
                    refined_mask = torch.maximum(refined_mask, mask_with_value)

        return refined_mask.cpu().numpy()

    def __del__(self):
        self.sam2_model = None
        self.mask_generator = None
        self._delete_predictions()
        torch.cuda.empty_cache()
