"""Evaluation package public API.

Keep imports lightweight so CLI help/config checks do not require optional metric
dependencies such as pycocotools. Heavy metric utilities are loaded lazily.
"""

from importlib import import_module

from .dataset_registry import DATASET_NAMES, get_dataset_config

_METRIC_EXPORTS = {
    "run_ytvis_map_evaluation",
    "run_hota_evaluation",
    "compute_odr",
    "compute_mask_j_score",
    "compute_jf_score",
    "compute_per_category_stats",
    "compute_frame_segm_ap",
    "compute_spatiotemporal_f1",
}

_YTVIS_EXPORTS = {
    "cache_to_ytvis_predictions",
}

_TRACKER_EXPORTS = {
    "build_tracker",
    "load_config",
}

_INFERENCE_EXPORTS = {
    "run_inference",
}

_AUX_EXPORTS = {
    "ThroughputCalculator": ".throughput_calculator",
    "ODRCalculator": ".odr_metric",
    "PGIoUCalculator": ".pgiou_metric",
}


def __getattr__(name):
    if name in _TRACKER_EXPORTS:
        tracker_builder = import_module(".tracker_builder", __name__)
        value = getattr(tracker_builder, name)
        globals()[name] = value
        return value
    if name in _INFERENCE_EXPORTS:
        inference_engine = import_module(".inference_engine", __name__)
        value = getattr(inference_engine, name)
        globals()[name] = value
        return value
    if name in _METRIC_EXPORTS:
        metrics = import_module(".metrics", __name__)
        value = getattr(metrics, name)
        globals()[name] = value
        return value
    if name in _YTVIS_EXPORTS:
        ytvis_utils = import_module(".ytvis_utils", __name__)
        value = getattr(ytvis_utils, name)
        globals()[name] = value
        return value
    if name in _AUX_EXPORTS:
        module = import_module(_AUX_EXPORTS[name], __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "get_dataset_config",
    "DATASET_NAMES",
    "build_tracker",
    "load_config",
    "run_inference",
    "ThroughputCalculator",
    "ODRCalculator",
    "PGIoUCalculator",
    *_METRIC_EXPORTS,
    *_YTVIS_EXPORTS,
]
