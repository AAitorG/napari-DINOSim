import numpy as np
import pytest
import torch

from napari_dinosim.dinoSim_pipeline import DinoSim_pipeline


# Mock DINOv2 model for testing
class MockDINOv2Model:
    def forward_features(self, x):
        batch_size = x.shape[0]
        patch_tokens = torch.randn(
            batch_size, 518 // 14 * 518 // 14, 384
        )  # Example dimensions
        return {"x_norm_patchtokens": patch_tokens}


@pytest.fixture
def mock_model():
    return MockDINOv2Model()


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def img_preprocessing():
    def preprocess(x):
        # Simple mock preprocessing that maintains tensor shape
        return x

    return preprocess


@pytest.fixture
def pipeline(mock_model, device, img_preprocessing):
    model_patch_size = 14  # Standard DINO patch size
    feat_dim = 384  # Standard DINO feature dimension
    return DinoSim_pipeline(
        model=mock_model,
        model_patch_size=model_patch_size,
        device=device,
        img_preprocessing=img_preprocessing,
        feat_dim=feat_dim,
        dino_image_size=518,
    )


def test_pipeline_initialization(pipeline):
    """Test if pipeline is initialized with correct parameters."""
    assert pipeline.dino_image_size == 518
    assert (
        pipeline.patch_h == pipeline.patch_w == pipeline.embedding_size == 37
    )  # 518/14
    assert pipeline.feat_dim == 384
    assert not pipeline.exist_reference
    assert isinstance(pipeline.embeddings, np.ndarray)
    assert len(pipeline.embeddings) == 0


def test_pre_compute_embeddings(pipeline):
    """Test embedding computation for a small dataset."""
    # Create a small test dataset
    batch_size = 2
    dataset = np.random.rand(batch_size, 518, 518, 3).astype(np.float32)

    # Compute embeddings without overlap to avoid patch multiplication
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(
            0,
            0,
        ),  # Changed from (0.5, 0.5) to avoid patch multiplication
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )

    # Check if embeddings were computed correctly
    assert pipeline.emb_precomputed
    assert isinstance(pipeline.embeddings, torch.Tensor)
    assert (
        pipeline.embeddings.shape[0] == batch_size
    )  # Should match input batch size
    assert (
        pipeline.embeddings.shape[1] == pipeline.embeddings.shape[2] == 37
    )  # 518/14
    assert pipeline.embeddings.shape[3] == pipeline.feat_dim


def test_set_reference_vector(pipeline):
    """Test setting reference vectors from coordinates."""
    # First compute embeddings
    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )

    # Set reference vectors
    list_coords = [(0, 100, 100)]  # Example coordinates
    pipeline.set_reference_vector(list_coords)

    # Check if reference vectors were set correctly
    assert pipeline.exist_reference
    assert isinstance(pipeline.reference_color, torch.Tensor)
    assert pipeline.reference_color.shape == (pipeline.feat_dim,)
    assert isinstance(pipeline.reference_emb, torch.Tensor)


def test_get_ds_distances(pipeline):
    """Test distance computation between dataset and reference embeddings."""
    # Setup test data
    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(
            0,
            0,
        ),  # Changed from (0.5, 0.5) to avoid patch multiplication
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )

    # Set reference vectors
    list_coords = [(0, 100, 100)]
    pipeline.set_reference_vector(list_coords)

    # Compute distances
    distances = pipeline.get_ds_distances_sameRef(verbose=False, k=5)

    # Check distance computation results
    assert isinstance(distances, np.ndarray)
    assert distances.shape[0] == 2  # Should match input batch size
    assert distances.shape[1] == 37  # embedding size
    assert distances.shape[2] == 37  # embedding size


def test_save_load_reference(pipeline, tmp_path):
    """Test saving and loading reference vectors."""
    # Setup test data and compute reference
    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )
    list_coords = [(0, 100, 100)]
    pipeline.set_reference_vector(list_coords)

    # Save reference
    save_path = tmp_path / "reference.pt"
    pipeline.save_reference(save_path)

    # Create new pipeline and load reference
    new_pipeline = DinoSim_pipeline(
        model=pipeline.model,
        model_patch_size=14,
        device=pipeline.device,
        img_preprocessing=pipeline.img_preprocessing,
        feat_dim=pipeline.feat_dim,
        dino_image_size=518,
    )
    new_pipeline.load_reference(save_path)

    # Check if reference was loaded correctly
    assert new_pipeline.exist_reference

    # Ensure both references are tensors before comparison
    if not isinstance(new_pipeline.reference_color, torch.Tensor):
        new_pipeline.reference_color = torch.tensor(
            new_pipeline.reference_color, device=new_pipeline.device
        )
    if not isinstance(pipeline.reference_color, torch.Tensor):
        pipeline.reference_color = torch.tensor(
            pipeline.reference_color, device=pipeline.device
        )

    assert torch.allclose(
        new_pipeline.reference_color, pipeline.reference_color
    )

    # Ensure both references are tensors before comparison
    if not isinstance(new_pipeline.reference_emb, torch.Tensor):
        new_pipeline.reference_emb = torch.tensor(
            new_pipeline.reference_emb, device=new_pipeline.device
        )
    if not isinstance(pipeline.reference_emb, torch.Tensor):
        pipeline.reference_emb = torch.tensor(
            pipeline.reference_emb, device=pipeline.device
        )

    assert torch.allclose(new_pipeline.reference_emb, pipeline.reference_emb)


def test_distance_post_processing(pipeline):
    """Test post-processing of computed distances."""
    # Setup test data
    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0, 0),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )

    # Set reference and compute distances
    list_coords = [(0, 100, 100)]
    pipeline.set_reference_vector(list_coords)
    distances = pipeline.get_ds_distances_sameRef(verbose=False, k=5)

    # Test post-processing
    processed_distances = pipeline.distance_post_processing(
        distances, low_res_filter=None, upsampling_mode="bilinear"
    )

    # Check processed results
    assert isinstance(processed_distances, np.ndarray)
    assert processed_distances.shape == (
        2,
        518,
        518,
        1,
    )  # Should match original image dimensions


def test_cleanup(pipeline):
    """Test proper cleanup of resources."""
    # Setup some data
    dataset = np.random.rand(2, 518, 518, 3).astype(np.float32)
    pipeline.pre_compute_embeddings(
        dataset,
        overlap=(0.5, 0.5),
        padding=(0, 0),
        crop_shape=(518, 518, 3),
        verbose=False,
        batch_size=1,
    )
    list_coords = [(0, 100, 100)]
    pipeline.set_reference_vector(list_coords)

    # Test deletion
    pipeline.delete_precomputed_embeddings()
    assert len(pipeline.embeddings) == 0
    assert not pipeline.emb_precomputed

    pipeline.delete_references()
    assert not pipeline.exist_reference


def test_quantile_normalization(pipeline):
    """Test quantile normalization of distances."""
    # Create test tensor
    test_tensor = torch.rand(100, 100)

    # Apply normalization
    normalized = pipeline.quantile_normalization(
        test_tensor, lower_quantile=0.01, upper_quantile=0.99
    )

    # Check results
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)
    assert normalized.shape == test_tensor.shape
