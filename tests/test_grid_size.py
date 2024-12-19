import pytest

from yasiu_vis.pandas import get_grid_dims


@pytest.mark.parametrize('size', [a for a in range(100)])
def test_grid_size(size):
    rows, cols = get_grid_dims(size)
    assert (rows * cols) >= size
