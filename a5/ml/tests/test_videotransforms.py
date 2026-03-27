"""Unit tests for i3d_msft/videotransforms.py."""

import numpy as np
import pytest

from i3d_msft.videotransforms import CenterCrop, RandomCrop, RandomHorizontalFlip


@pytest.fixture
def sample_video():
    """A dummy video tensor of shape (T, H, W, C)."""
    return np.random.randint(0, 256, (8, 100, 120, 3), dtype=np.uint8)


class TestRandomCrop:
    def test_int_size(self):
        crop = RandomCrop(64)
        assert crop.size == (64, 64)

    def test_tuple_size(self):
        crop = RandomCrop((50, 60))
        assert crop.size == (50, 60)

    def test_output_shape(self, sample_video):
        crop = RandomCrop((50, 60))
        out = crop(sample_video)
        assert out.shape == (8, 50, 60, 3)

    def test_same_size_noop(self, sample_video):
        """When crop size equals input size, output equals input."""
        h, w = sample_video.shape[1], sample_video.shape[2]
        crop = RandomCrop((h, w))
        out = crop(sample_video)
        np.testing.assert_array_equal(out, sample_video)

    def test_get_params_same_size(self, sample_video):
        h, w = sample_video.shape[1], sample_video.shape[2]
        i, j, th, tw = RandomCrop.get_params(sample_video, (h, w))
        assert i == 0
        assert j == 0
        assert th == h
        assert tw == w

    def test_get_params_smaller_crop(self, sample_video):
        i, j, th, tw = RandomCrop.get_params(sample_video, (50, 60))
        assert 0 <= i <= sample_video.shape[1] - 50
        assert 0 <= j <= sample_video.shape[2] - 60
        assert th == 50
        assert tw == 60

    def test_get_params_h_equals_th(self):
        """When only height matches, i should be 0 but j can vary."""
        imgs = np.zeros((4, 50, 100, 3))
        i, j, th, tw = RandomCrop.get_params(imgs, (50, 60))
        assert i == 0
        assert th == 50
        assert tw == 60

    def test_get_params_w_equals_tw(self):
        """When only width matches, j should be 0 but i can vary."""
        imgs = np.zeros((4, 100, 60, 3))
        i, j, th, tw = RandomCrop.get_params(imgs, (50, 60))
        assert j == 0
        assert th == 50
        assert tw == 60

    def test_repr(self):
        crop = RandomCrop(64)
        assert "RandomCrop" in repr(crop)
        assert "(64, 64)" in repr(crop)

    def test_temporal_dimension_preserved(self, sample_video):
        crop = RandomCrop(50)
        out = crop(sample_video)
        assert out.shape[0] == sample_video.shape[0]

    def test_channels_preserved(self, sample_video):
        crop = RandomCrop(50)
        out = crop(sample_video)
        assert out.shape[3] == sample_video.shape[3]


class TestCenterCrop:
    def test_int_size(self):
        crop = CenterCrop(64)
        assert crop.size == (64, 64)

    def test_tuple_size(self):
        crop = CenterCrop((50, 60))
        assert crop.size == (50, 60)

    def test_output_shape(self, sample_video):
        crop = CenterCrop((50, 60))
        out = crop(sample_video)
        assert out.shape == (8, 50, 60, 3)

    def test_center_position(self):
        """Verify the crop is actually centered."""
        video = np.arange(4 * 10 * 10 * 3).reshape(4, 10, 10, 3)
        crop = CenterCrop((4, 6))
        out = crop(video)
        # Center crop: i = round((10-4)/2) = 3, j = round((10-6)/2) = 2
        expected = video[:, 3:7, 2:8, :]
        np.testing.assert_array_equal(out, expected)

    def test_same_size_noop(self, sample_video):
        h, w = sample_video.shape[1], sample_video.shape[2]
        crop = CenterCrop((h, w))
        out = crop(sample_video)
        np.testing.assert_array_equal(out, sample_video)

    def test_repr(self):
        crop = CenterCrop(64)
        assert "CenterCrop" in repr(crop)
        assert "(64, 64)" in repr(crop)


class TestRandomHorizontalFlip:
    def test_always_flip(self):
        flip = RandomHorizontalFlip(p=1.0)
        video = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)
        out = flip(video)
        expected = np.flip(video, axis=2)
        np.testing.assert_array_equal(out, expected)

    def test_never_flip(self):
        flip = RandomHorizontalFlip(p=0.0)
        video = np.arange(2 * 4 * 6 * 3).reshape(2, 4, 6, 3)
        out = flip(video)
        np.testing.assert_array_equal(out, video)

    def test_output_shape(self, sample_video):
        flip = RandomHorizontalFlip(p=0.5)
        out = flip(sample_video)
        assert out.shape == sample_video.shape

    def test_repr(self):
        flip = RandomHorizontalFlip(p=0.3)
        assert "RandomHorizontalFlip" in repr(flip)
        assert "0.3" in repr(flip)

    def test_flip_returns_copy(self):
        """Flipped result should be a separate array (not a view)."""
        flip = RandomHorizontalFlip(p=1.0)
        video = np.ones((2, 4, 6, 3))
        out = flip(video)
        out[0, 0, 0, 0] = 999
        assert video[0, 0, 0, 0] != 999

    def test_default_probability(self):
        flip = RandomHorizontalFlip()
        assert flip.p == 0.5
