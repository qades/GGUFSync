"""Unit tests for download detection."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from link_models.core.watcher import DownloadDetector


class TestDownloadDetector:
    """Tests for DownloadDetector class."""
    
    def test_is_partial_with_part_extension(self) -> None:
        detector = DownloadDetector()
        assert detector.is_partial(Path("model.gguf.part")) is True
    
    def test_is_partial_with_tmp_extension(self) -> None:
        detector = DownloadDetector()
        assert detector.is_partial(Path("model.gguf.tmp")) is True
    
    def test_is_partial_complete_file(self) -> None:
        detector = DownloadDetector()
        assert detector.is_partial(Path("model.gguf")) is False
    
    def test_get_real_name_from_partial(self) -> None:
        detector = DownloadDetector()
        assert detector.get_real_name(Path("model.gguf.part")) == "model.gguf"
    
    def test_get_real_name_from_complete(self) -> None:
        detector = DownloadDetector()
        assert detector.get_real_name(Path("model.gguf")) == "model.gguf"
    
    def test_add_pending_tracks_download(self) -> None:
        detector = DownloadDetector()
        path = Path("model.gguf.part")
        
        pending = detector.add_pending(path)
        
        assert pending.path == path
        assert pending.real_name == "model.gguf"
        assert detector.pending_count == 1
    
    def test_remove_pending_removes_tracking(self, temp_dir: Path) -> None:
        detector = DownloadDetector()
        path = temp_dir / "model.gguf.part"
        path.write_text("content")
        
        detector.add_pending(path)
        assert detector.pending_count == 1
        
        removed = detector.remove_pending(path)
        assert removed is not None
        assert detector.pending_count == 0
    
    def test_check_complete_for_nonexistent_file(self, temp_dir: Path) -> None:
        detector = DownloadDetector()
        path = temp_dir / "nonexistent.gguf"
        
        is_complete, final_path = detector.check_complete(path)
        
        assert is_complete is False
        assert final_path is None
    
    def test_check_complete_file_still_growing(self, temp_dir: Path) -> None:
        detector = DownloadDetector(
            check_interval=0.01,
            stable_count=3,
        )
        path = temp_dir / "model.gguf"
        path.write_text("initial content")
        
        # Add to pending
        detector.add_pending(path)
        
        # First check - size changed from 0
        is_complete, _ = detector.check_complete(path)
        assert is_complete is False
        
        # Modify file (simulating download)
        path.write_text("more content here")
        
        # Check again - size changed
        is_complete, _ = detector.check_complete(path)
        assert is_complete is False
    
    def test_check_complete_stable_file(self, temp_dir: Path) -> None:
        detector = DownloadDetector(
            check_interval=0.001,
            stable_count=2,
        )
        path = temp_dir / "model.gguf"
        path.write_text("stable content")
        
        # Add to pending
        detector.add_pending(path)
        
        # First check - records size
        is_complete, _ = detector.check_complete(path)
        assert is_complete is False
        
        # Second check - size same, stable_count = 1
        is_complete, _ = detector.check_complete(path)
        assert is_complete is False
        
        # Third check - size same, stable_count = 2, meets requirement
        is_complete, final_path = detector.check_complete(path)
        assert is_complete is True
        assert final_path == path
    
    def test_check_complete_partial_file_renamed(self, temp_dir: Path) -> None:
        detector = DownloadDetector()
        partial_path = temp_dir / "model.gguf.part"
        real_path = temp_dir / "model.gguf"
        
        partial_path.write_text("content")
        
        # Add to pending
        detector.add_pending(partial_path)
        
        # Simulate download completion by renaming
        partial_path.rename(real_path)
        
        # Check should detect the renamed file
        is_complete, final_path = detector.check_complete(partial_path)
        
        assert is_complete is True
        assert final_path == real_path
    
    def test_check_all_pending_returns_completed(self, temp_dir: Path) -> None:
        detector = DownloadDetector(
            check_interval=0.001,
            stable_count=1,
        )
        
        path1 = temp_dir / "model1.gguf"
        path2 = temp_dir / "model2.gguf"
        path1.write_text("content1")
        path2.write_text("content2")
        
        detector.add_pending(path1)
        detector.add_pending(path2)
        
        # First check establishes baseline
        completed = detector.check_all_pending()
        assert len(completed) == 0
        
        # Second check confirms stability, both should complete
        completed = detector.check_all_pending()
        
        assert len(completed) == 2
        assert all(final_path is not None for _, final_path in completed)
    
    def test_timeout_removes_stale_download(self, temp_dir: Path) -> None:
        detector = DownloadDetector(
            check_interval=0.001,
            stable_count=10,  # High so it won't complete naturally
            max_wait=0.01,  # Very short timeout
        )
        
        path = temp_dir / "model.gguf"
        path.write_text("content")
        
        detector.add_pending(path)
        
        # Wait for timeout
        time.sleep(0.02)
        
        # Check should timeout
        is_complete, _ = detector.check_complete(path)
        assert is_complete is True  # Times out as complete
        assert detector.pending_count == 0
