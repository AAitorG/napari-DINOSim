import numpy as np
import torch

from napari_dinosim.utils import DINOSim_pipeline


class MockDINOv2Model:
    def forward_features(self, x):
        batch_size = x.shape[0]
        patch_tokens = torch.randn(batch_size, 37 * 37, 384)
        return {"x_norm_patchtokens": patch_tokens}


def test_pipeline_end_to_end():
    """Smoke test: init, compute embeddings, set reference, get distances."""
    device = torch.device("cpu")
    pipeline = DINOSim_pipeline(
        model=MockDINOv2Model(),
        model_patch_size=14,
        device=device,
        img_preprocessing=lambda x: x,
        feat_dim=384,
        dino_image_size=518,
    )

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
