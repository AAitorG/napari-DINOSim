"""Test suite for napari-dinoSim plugin."""

import pytest

# Configure pytest to show deprecation warnings
pytest.register_assert_rewrite("numpy.testing")

# If you need to register custom markers, use pytest.ini or setup.cfg instead
# Example in setup.cfg:
# [tool:pytest]
# markers =
#     napari_dinosim: mark tests that are specific to the napari-dinoSim plugin
