from unittest.mock import patch

import numpy as np
import pytest
import torch

from napari_dinosim.utils import DINOSim_pipeline


class MockDINOv2Model:
    def forward_features(self, x):
        batch_size = x.shape[0]
        patch_tokens = torch.randn(batch_size, 37 * 37, 384)
        return {"x_norm_patchtokens": patch_tokens}


def _make_pipeline(device=None):
    if device is None:
        device = torch.device("cpu")
    return DINOSim_pipeline(
        model=MockDINOv2Model(),
        model_patch_size=14,
        device=device,
        img_preprocessing=lambda x: x,
        feat_dim=384,
        dino_image_size=518,
    )


def test_pipeline_end_to_end():
    """Smoke test: init, compute embeddings, set reference, get distances."""
    pipeline = _make_pipeline()

    assert pipeline.patch_h == 37
    assert not pipeline.exist_reference

    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )
    assert pipeline.emb_precomputed
    assert pipeline.embeddings.shape == (2, 37, 37, 384)

    pipeline.set_reference_vector([(0, 100, 100)])
    assert pipeline.exist_reference

    distances = pipeline.get_ds_distances_sameRef(verbose=False, k=5)
    assert distances.shape == (2, 37, 37)


def test_precompute_cancel_check():
    """Cancelled precompute must not mark embeddings as ready."""
    pipeline = _make_pipeline()
    dataset = np.random.rand(1, 1036, 1036, 3).astype(np.float32)
    checks = {"count": 0}

    def cancel_check():
        checks["count"] += 1
        return checks["count"] > 2

    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
        cancel_check=cancel_check,
    )
    assert not pipeline.emb_precomputed


def test_load_embeddings_rejects_shape_mismatch(tmp_path):
    """Loading embeddings for a different image shape raises ValueError."""
    pipeline = _make_pipeline()
    dataset = np.random.rand(1, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )
    filepath = tmp_path / "embeddings.pt"
    pipeline.save_embeddings(str(filepath))

    wrong_shape = (1, 256, 256, 3)
    with pytest.raises(ValueError, match="Incompatible image shape"):
        pipeline.load_embeddings(
            str(filepath),
            image_shape=wrong_shape,
            expected_crop_shape=(518, 518, 3),
        )


def test_load_embeddings_respects_gpu_memory(tmp_path):
    """Embeddings stay on CPU when GPU memory check fails."""
    pipeline = _make_pipeline()
    dataset = np.random.rand(1, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )
    filepath = tmp_path / "embeddings.pt"
    pipeline.save_embeddings(str(filepath))
    pipeline.delete_precomputed_embeddings()

    original_device = pipeline.device
    pipeline.device = torch.device("cuda")
    try:
        with patch.object(pipeline, "check_gpu_memory", return_value=False):
            pipeline.load_embeddings(
                str(filepath),
                image_shape=(1, 518, 518, 3),
                expected_crop_shape=(518, 518, 3),
            )
    finally:
        pipeline.device = original_device

    assert pipeline.embeddings_on_cpu
    assert pipeline.embeddings.device.type == "cpu"
