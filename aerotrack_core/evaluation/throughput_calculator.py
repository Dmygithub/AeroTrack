import time
import logging

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch as _torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class ThroughputCalculator:
    """Track inference throughput (MPS/FPS) and peak GPU/CPU usage."""

    def __init__(self):
        self.total_infer_time = 0.0
        self.total_masks_generated = 0
        # Total processed frames across all (video, category) runs. Enables
        # a frame-rate metric (FPS) that complements MPS for sparse-tracking
        # baselines where masks_per_frame is small.
        self.total_frames_processed = 0
        self._start_time = None
        self.failed_runs = []
        self.peak_gpu_mb = 0.0
        self.peak_cpu_mb = 0.0

    def start_timer(self):
        """Record inference start time."""
        self._start_time = time.perf_counter()

    def stop_timer(self, masks_count: int, frames_count: int = 0):
        """Record inference end, accumulate counts, and sample resource peaks.

        Args:
            masks_count: Number of valid masks produced in this run.
            frames_count: Frames processed; 0 disables FPS aggregation.
        """
        if self._start_time is None:
            logging.warning("ThroughputCalculator.stop_timer called before start_timer.")
            return

        elapsed = time.perf_counter() - self._start_time
        self.total_infer_time += elapsed
        self.total_masks_generated += masks_count
        if frames_count and frames_count > 0:
            self.total_frames_processed += int(frames_count)
        self._start_time = None

        if _HAS_TORCH and _torch.cuda.is_available():
            gpu_mb = _torch.cuda.max_memory_allocated() / (1024 * 1024)
            if gpu_mb > self.peak_gpu_mb:
                self.peak_gpu_mb = gpu_mb

        if _HAS_PSUTIL:
            try:
                proc = _psutil.Process()
                cpu_mb = proc.memory_info().rss / (1024 * 1024)
                if cpu_mb > self.peak_cpu_mb:
                    self.peak_cpu_mb = cpu_mb
            except Exception:
                pass

    def get_mps(self) -> float:
        """Return masks per second."""
        if self.total_infer_time == 0:
            return 0.0
        return self.total_masks_generated / self.total_infer_time

    def get_fps(self) -> float:
        """Return frames per second."""
        if self.total_infer_time == 0 or self.total_frames_processed == 0:
            return 0.0
        return self.total_frames_processed / self.total_infer_time

    def get_system_metrics(self) -> dict:
        """Return aggregated throughput and resource metrics."""
        fps = self.get_fps()
        ms_per_frame = (1000.0 / fps) if fps > 0 else 0.0
        return {
            "mps":           round(self.get_mps(), 2),
            "fps":           round(fps, 2),
            "ms_per_frame":  round(ms_per_frame, 2),
            "peak_gpu_mb":   round(self.peak_gpu_mb, 1),
            "peak_cpu_mb":   round(self.peak_cpu_mb, 1),
            "total_time_s":  round(self.total_infer_time, 2),
            "total_masks":   self.total_masks_generated,
            "total_frames":  self.total_frames_processed,
            "failed_runs":   len(self.failed_runs),
            "failures":      self.failed_runs,
        }

    def report(self, prefix=""):
        """Return a legacy text report string."""
        mps = self.get_mps()
        report_str = (
            f"[{prefix}] MPS: {mps:.2f} | "
            f"Peak GPU: {self.peak_gpu_mb:.0f} MB | "
            f"Peak CPU: {self.peak_cpu_mb:.0f} MB | "
            f"Time: {self.total_infer_time:.2f}s | "
            f"Masks: {self.total_masks_generated}"
        )
        logging.info(report_str)
        return report_str

    def reset(self):
        """Reset all accumulated counters."""
        self.total_infer_time = 0.0
        self.total_masks_generated = 0
        self.total_frames_processed = 0
        self._start_time = None
        self.failed_runs = []
        self.peak_gpu_mb = 0.0
        self.peak_cpu_mb = 0.0
