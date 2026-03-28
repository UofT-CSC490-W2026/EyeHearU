"""
Coverage-depth tests for ``app.services.preprocessing`` — the most complex module.

WHY THIS MODULE?
----------------
``preprocessing.py`` is the critical accuracy path between a raw phone video and
the I3D model input tensor.  Every frame passes through temporal sampling, spatial
resize, normalization, padding, and center-cropping.  A bug in any stage silently
degrades prediction accuracy (the root cause of the original low-accuracy issue
was a spatial resize bug here).  The module must handle:

- Arbitrary aspect ratios (portrait 9:16, landscape 16:9, square, ultra-wide)
- Extreme resolutions (4K, 8K, tiny webcam)
- Short / long / empty / single-frame videos
- Corrupt or partial decodes (cv2.read returning False)
- Missing dependencies (cv2 not installed)
- Filesystem edge cases (temp file cleanup failures)

Each test below targets a specific edge case, failure mode, or important use case.
Tests are grouped: POSITIVE tests verify correct behavior; NEGATIVE tests verify
graceful error handling.
"""

import builtins
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.services import preprocessing as pre
from app.services.preprocessing import (
    CROP_SIZE,
    MOBILE_MAX_LONG_SIDE,
    RESIZE_SIDE,
    TOTAL_FRAMES,
    _center_crop,
    _ensure_both_sides_at_least,
    _load_rgb_frames,
    _pad_frames,
    _resize_bgr_uint8,
    preprocess_video,
)


def _bgr_frame(h: int, w: int) -> np.ndarray:
    """Helper: synthetic BGR uint8 frame."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


def _mock_cap(total_frames: int, frame_h: int, frame_w: int, *, fail_after: int | None = None):
    """Build a mock cv2.VideoCapture that returns ``total_frames`` of (h, w) frames.

    If ``fail_after`` is set, reads succeed for that many calls then return (False, None).
    """
    import cv2

    cap = MagicMock()

    def _get(prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return total_frames
        return 0

    cap.get.side_effect = _get

    if fail_after is not None:
        img = _bgr_frame(frame_h, frame_w)
        cap.read.side_effect = [(True, img)] * fail_after + [(False, None)] * (total_frames - fail_after + 10)
    else:
        img = _bgr_frame(frame_h, frame_w)
        cap.read.return_value = (True, img)

    return cap


# ============================================================================
# POSITIVE TESTS — verify correct behavior for valid inputs
# ============================================================================


class TestPositive:
    """Tests that verify correct output for valid inputs across edge-case scenarios."""

    def test_portrait_9_16_preserves_spatial_detail(self):
        """
        EDGE CASE: Portrait phone video (1080x1920 → 720x1280 after mobile cap).
        WHY: This was the original bug — the old min-226/max-256 pipeline crushed
        portrait width to ~144px.  The short-side-256 resize must keep the shorter
        dimension (width=720 → 256) and scale height proportionally (~456).
        """
        cap = _mock_cap(30, 1280, 720)
        with patch("cv2.VideoCapture", return_value=cap):
            out = _load_rgb_frames("/fake.mp4", max_frames=4)
        # Short side must be exactly RESIZE_SIDE
        assert min(out.shape[1], out.shape[2]) == RESIZE_SIDE
        # Long side must be proportionally larger (not crushed)
        assert max(out.shape[1], out.shape[2]) > RESIZE_SIDE
        # Aspect ratio roughly preserved: 1280/720 ≈ 1.78
        ratio = max(out.shape[1], out.shape[2]) / min(out.shape[1], out.shape[2])
        assert abs(ratio - 1280 / 720) < 0.05

    def test_4k_video_downscaled_before_resize(self):
        """
        EDGE CASE: 4K video (3840x2160) exceeds MOBILE_MAX_LONG_SIDE.
        WHY: Without the mobile cap, 4K frames waste memory and can amplify codec
        artifacts.  The pipeline must first downscale the long side to 1280, then
        apply short-side-256 resize.
        """
        cap = _mock_cap(10, 2160, 3840)
        with patch("cv2.VideoCapture", return_value=cap):
            out = _load_rgb_frames("/fake.mp4", max_frames=4)
        assert out.shape[0] >= 1
        # After mobile cap + short-side-256, short side == RESIZE_SIDE
        assert min(out.shape[1], out.shape[2]) == RESIZE_SIDE
        # Long side should be capped proportionally (not 3840)
        assert max(out.shape[1], out.shape[2]) <= 512  # well below original

    def test_single_frame_video_padded_to_64(self):
        """
        EDGE CASE: Video has only 1 decodable frame.
        WHY: Very short recordings or corrupt videos may yield just 1 frame.
        The pipeline must pad to TOTAL_FRAMES (64) by repeating the last frame,
        producing a valid tensor for the model.
        """
        frames = np.random.rand(1, RESIZE_SIDE, RESIZE_SIDE, 3).astype(np.float32)
        padded = _pad_frames(frames, TOTAL_FRAMES)
        assert padded.shape == (TOTAL_FRAMES, RESIZE_SIDE, RESIZE_SIDE, 3)
        # All 64 frames should be identical (copies of the single input frame)
        for i in range(1, TOTAL_FRAMES):
            np.testing.assert_array_equal(padded[i], padded[0])

    def test_normalization_range_minus1_to_plus1(self):
        """
        IMPORTANT USE CASE: I3D training uses [-1, 1] normalization, NOT ImageNet.
        WHY: Using the wrong normalization (e.g. ImageNet mean/std) silently produces
        wrong predictions.  This test verifies that output pixel values are in [-1, 1].
        """
        cap = _mock_cap(10, 256, 256)
        with patch("cv2.VideoCapture", return_value=cap):
            out = _load_rgb_frames("/fake.mp4", max_frames=4)
        assert out.min() >= -1.0
        assert out.max() <= 1.0
        # Verify it's not all zeros (normalization actually happened)
        assert out.max() > out.min()

    def test_full_pipeline_output_shape_and_dtype(self):
        """
        IMPORTANT USE CASE: End-to-end pipeline must produce (1, 3, 64, 224, 224) float32.
        WHY: The I3D model expects exactly this shape.  Any deviation causes a runtime
        crash or silent shape mismatch.  This tests the full preprocess_video path.
        """
        frames = np.random.rand(64, 300, 256, 3).astype(np.float32) * 2 - 1
        with patch.object(pre, "_load_rgb_frames", return_value=frames), \
             patch("app.services.preprocessing.os.unlink"):
            tensor = preprocess_video(b"fake-video-bytes")
        assert tensor.shape == (1, 3, TOTAL_FRAMES, CROP_SIZE, CROP_SIZE)
        assert tensor.dtype == torch.float32

    def test_frameskip_adapts_to_high_fps_video(self):
        """
        EDGE CASE: 200-frame video (e.g. 60fps × 3.3s) triggers frameskip=3.
        WHY: Without adaptive frame skipping, 60fps video would oversample the
        beginning and miss the end of the sign.  Frameskip=3 ensures temporal
        coverage across the whole clip, matching training behavior.
        """
        cap = _mock_cap(200, 256, 256)
        with patch("cv2.VideoCapture", return_value=cap):
            out = _load_rgb_frames("/fake.mp4", max_frames=64)
        # With 200 frames and frameskip=3, we sample every 3rd frame
        # from a centered window, yielding up to 64 frames
        assert out.shape[0] > 0
        assert out.shape[0] <= TOTAL_FRAMES

    def test_square_video_no_aspect_distortion(self):
        """
        EDGE CASE: Square video (e.g. Instagram-style 1:1).
        WHY: Ensures the short-side-256 resize works for square aspect ratios where
        both sides are equal.  Both dimensions should become exactly RESIZE_SIDE.
        """
        cap = _mock_cap(10, 500, 500)
        with patch("cv2.VideoCapture", return_value=cap):
            out = _load_rgb_frames("/fake.mp4", max_frames=4)
        assert out.shape[1] == RESIZE_SIDE
        assert out.shape[2] == RESIZE_SIDE

    def test_center_crop_extracts_center_region(self):
        """
        IMPORTANT USE CASE: Center crop must extract the geometrically central region.
        WHY: ASL signs are performed in front of the signer, typically centered in frame.
        An off-center crop would cut off hands/fingers.  This verifies the exact crop
        position by planting a marker at the center.
        """
        h, w = 300, 400
        imgs = np.zeros((1, h, w, 3), dtype=np.float32)
        # Plant a bright pixel at the exact center
        cy, cx = h // 2, w // 2
        imgs[0, cy, cx, :] = 1.0
        cropped = _center_crop(imgs, CROP_SIZE)
        # The marker should be near the center of the cropped output
        expected_y = cy - int(np.round((h - CROP_SIZE) / 2.0))
        expected_x = cx - int(np.round((w - CROP_SIZE) / 2.0))
        assert cropped[0, expected_y, expected_x, 0] == 1.0

    def test_resize_uses_area_interpolation_when_shrinking(self):
        """
        IMPORTANT USE CASE: Downscaling should use INTER_AREA for sharp results.
        WHY: INTER_LINEAR creates aliasing artifacts when downscaling.  INTER_AREA
        averages pixels in the source region, producing cleaner results for the
        typical phone-to-model resolution reduction (1080→256).
        """
        import cv2
        img = _bgr_frame(1000, 1000)
        with patch("cv2.resize", wraps=cv2.resize) as mock_resize:
            _resize_bgr_uint8(img, 256, 256, shrinking=True)
            _, kwargs = mock_resize.call_args
            assert kwargs["interpolation"] == cv2.INTER_AREA

    def test_resize_uses_linear_interpolation_when_enlarging(self):
        """
        EDGE CASE: Small input frames need enlarging (e.g. old webcam at 160x120).
        WHY: INTER_AREA is undefined for upscaling; INTER_LINEAR produces smooth
        results.  This verifies the correct interpolation is selected.
        """
        import cv2
        img = _bgr_frame(100, 100)
        with patch("cv2.resize", wraps=cv2.resize) as mock_resize:
            _resize_bgr_uint8(img, 256, 256, shrinking=False)
            _, kwargs = mock_resize.call_args
            assert kwargs["interpolation"] == cv2.INTER_LINEAR


# ============================================================================
# NEGATIVE TESTS — verify graceful error handling for invalid/edge inputs
# ============================================================================


class TestNegative:
    """Tests that verify errors are raised or handled gracefully for bad inputs."""

    def test_zero_frame_video_raises_value_error(self):
        """
        FAILURE MODE: Video file exists but has 0 decodable frames (corrupt container).
        WHY: The model cannot infer from zero frames.  preprocess_video must raise
        a clear ValueError rather than passing an empty tensor to the model (which
        would produce garbage predictions silently).
        """
        cap = _mock_cap(0, 256, 256)
        with patch("cv2.VideoCapture", return_value=cap):
            with pytest.raises(ValueError, match="no decodable"):
                preprocess_video(b"corrupt-video")

    def test_all_reads_fail_raises_value_error(self):
        """
        FAILURE MODE: cv2 reports frames but every read() returns False (truncated file).
        WHY: This simulates a video with a valid header but corrupt/missing frame data.
        The pipeline should raise ValueError, not return a zero tensor.
        """
        cap = _mock_cap(30, 256, 256, fail_after=0)
        with patch("cv2.VideoCapture", return_value=cap):
            with pytest.raises(ValueError, match="no decodable"):
                preprocess_video(b"truncated-video")

    def test_center_crop_rejects_undersized_frames(self):
        """
        FAILURE MODE: Frames smaller than CROP_SIZE (shouldn't happen after resize,
        but defensive check).
        WHY: If upstream resize fails or is bypassed, center_crop must raise a clear
        error rather than producing a silently wrong crop with negative indices.
        """
        tiny = np.zeros((4, 100, 100, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="smaller than crop"):
            _center_crop(tiny, CROP_SIZE)

    def test_missing_opencv_raises_runtime_error(self, monkeypatch):
        """
        FAILURE MODE: opencv-python-headless not installed.
        WHY: In minimal Docker images or broken environments, cv2 may be missing.
        The error message must tell the user exactly what to install, not show a
        cryptic ImportError.
        """
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        with pytest.raises(RuntimeError, match="opencv"):
            _load_rgb_frames("/nonexistent.mp4")

    def test_temp_file_cleanup_failure_does_not_crash(self):
        """
        FAILURE MODE: os.unlink fails (e.g. permission error, file already deleted).
        WHY: preprocess_video writes to a temp file and must clean up.  If unlink
        fails, it should NOT propagate the OSError — the prediction result is still
        valid.  This verifies the finally-block suppresses the exception.
        """
        frames = np.random.rand(64, 256, 256, 3).astype(np.float32) * 2 - 1
        with patch.object(pre, "_load_rgb_frames", return_value=frames), \
             patch("app.services.preprocessing.os.unlink", side_effect=OSError("permission denied")):
            # Should NOT raise — OSError from unlink is suppressed
            tensor = preprocess_video(b"valid-video")
        assert tensor.shape == (1, 3, 64, 224, 224)

    def test_decode_error_propagates_through_pipeline(self):
        """
        FAILURE MODE: _load_rgb_frames raises an unexpected error (e.g. codec crash).
        WHY: Unexpected errors should propagate to the caller (the predict endpoint)
        rather than being silently swallowed.  The endpoint returns a 500 with the
        error message.
        """
        with patch.object(pre, "_load_rgb_frames", side_effect=RuntimeError("codec crash")):
            with pytest.raises(RuntimeError, match="codec crash"):
                preprocess_video(b"bad-codec-video")
