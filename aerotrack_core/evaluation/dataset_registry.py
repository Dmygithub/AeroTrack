"""Dataset path registry for OVTS benchmarks and merged AeroVIS."""

import os

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_REGISTRY = {
    "seadronessee": {
        "ovts_dir":  os.path.join(_PROJECT_ROOT, "data", "SeaDronesSee-OVTS"),
        "gt_json":   "seadronessee_ovts.json",
        "hota_name": "SeaDronesSee-OVTS",
    },
    "uavdt": {
        "ovts_dir":  os.path.join(_PROJECT_ROOT, "data", "UAVDT-OVTS"),
        "gt_json":   "uavdt_ovts.json",
        "hota_name": "UAVDT-OVTS",
    },
    "visdrone": {
        "ovts_dir":  os.path.join(_PROJECT_ROOT, "data", "VisDrone-OVTS"),
        "gt_json":   "visdrone_ovts.json",
        "hota_name": "VisDrone-OVTS",
    },
    "aero_ovts": {
        "ovts_dir":  os.path.join(_PROJECT_ROOT, "data", "AeroVIS"),
        "gt_json":   "aero_vis.json",
        "hota_name": "AeroVIS",
    },
}

DATASET_NAMES = list(_REGISTRY.keys())


def get_dataset_config(name: str) -> dict:
    """Return absolute paths and metadata for one registered dataset.

    Returns:
        dict with keys: name, ovts_dir, gt_json, seq_root, hota_name.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset: '{name}'. Available options: {DATASET_NAMES}"
        )
    entry = _REGISTRY[name]
    ovts_dir = entry["ovts_dir"]
    return {
        "name":      name,
        "ovts_dir":  ovts_dir,
        "gt_json":   os.path.join(ovts_dir, entry["gt_json"]),
        "seq_root":  os.path.join(ovts_dir, "sequences"),
        "hota_name": entry["hota_name"],
    }
