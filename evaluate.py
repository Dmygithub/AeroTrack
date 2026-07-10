"""Unified OVTS evaluation entry point for AeroTrack."""
# nohup python evaluate.py --baseline 4 --text car --output outputs/bl4/all/car > log/bl4/all/car.log 2>&1 &
# nohup python evaluate.py --baseline 4 --output outputs/bl4/all > log/bl4/all/all.log 2>&1 &

import os
import sys
import time

os.environ["LOG_LEVEL"] = "ERROR"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Line-buffer stdout/stderr so nohup log files flush promptly.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import argparse
import json
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from aerotrack_core.evaluation import DATASET_NAMES, get_dataset_config


_RESULT_SIGNATURE_KEYS = (
    "dataset",
    "baseline",
    "config",
    "gt_json",
    "seq_root",
    "text",
    "categories",
    "pred_json",
    "max_videos",
    "relocation_interval",
    "max_objects",
    "max_trk_keep_alive",
    "max_missing_segments",
    "max_dets",
    "nms_iou_thr",
    "match_iou_thr",
    "box_threshold",
    "text_threshold",
    "score_thr",
    "mask_nms_iou_thr",
    "max_mask_area_ratio",
    "new_det_thresh",
    "use_track_lifecycle",
    "lost_track_ttl_segments",
    "lifecycle_match_score_thr",
    "lifecycle_center_gate",
    "lifecycle_area_ratio_min",
    "lifecycle_area_ratio_max",
    "gdino_infer_scale",
    "yolo_infer_scale",
    "sam3_ckpt",
    "sam2_ckpt",
    "yolo_ckpt",
    "gdino_ckpt",
    "gdino_bert_path",
)

_CATEGORY_TUNING_SIGNATURE_KEYS = (
    "relocation_interval",
    "max_objects",
    "max_trk_keep_alive",
    "max_missing_segments",
    "max_dets",
    "nms_iou_thr",
    "match_iou_thr",
    "box_threshold",
    "text_threshold",
    "score_thr",
    "mask_nms_iou_thr",
    "max_mask_area_ratio",
    "new_det_thresh",
    "use_track_lifecycle",
    "lost_track_ttl_segments",
    "lifecycle_match_score_thr",
    "lifecycle_center_gate",
    "lifecycle_area_ratio_min",
    "lifecycle_area_ratio_max",
    "gdino_infer_scale",
    "yolo_infer_scale",
)


def _jsonable_signature_value(value):
    """Convert a config value into a JSON-serializable signature field."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_signature_value(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _jsonable_signature_value(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    return str(value)


def _category_result_signature(args, cname, cid, cat_videos, gt_data_eval):
    """Build a cache signature for one category evaluation run."""
    params = {
        key: _jsonable_signature_value(getattr(args, key))
        for key in _RESULT_SIGNATURE_KEYS
        if hasattr(args, key)
    }
    params["sam3_tuning"] = _jsonable_signature_value(getattr(args, "sam3_tuning", None))

    from aerotrack_core.evaluation import load_config

    cat_cfg_path = os.path.join(_ROOT, "configs", "categories", f"{str(cname).lower()}.yaml")
    if not os.path.exists(cat_cfg_path):
        cat_cfg_path = os.path.join(_ROOT, "configs", "categories", "auto.yaml")
    cat_cfg = load_config(cat_cfg_path) if os.path.exists(cat_cfg_path) else {}
    cli_overrides = getattr(args, "_cli_tuning_overrides", {}) or {}
    category_params = {}
    for key in _CATEGORY_TUNING_SIGNATURE_KEYS:
        if cli_overrides.get(key) is not None:
            value = cli_overrides[key]
        elif key in cat_cfg:
            value = cat_cfg[key]
        elif hasattr(args, key):
            value = getattr(args, key)
        else:
            continue
        category_params[key] = _jsonable_signature_value(value)

    video_ids = sorted(int(v["id"]) for v in cat_videos)
    return {
        "schema": "aerotrack_eval_result_v3_metrics_ap_hota_fap",
        "category": str(cname),
        "category_id": int(cid),
        "category_config": os.path.relpath(cat_cfg_path, _ROOT) if os.path.exists(cat_cfg_path) else None,
        "category_params": category_params,
        "video_ids": video_ids,
        "num_annotations": len(gt_data_eval.get("annotations", [])),
        "params": params,
    }


_BASELINE_CONFIGS = {
    1: "configs/baselines/baseline1_sam3_only.yaml",
    2: "configs/baselines/baseline2_sam3_reloc.yaml",
    3: "configs/baselines/baseline3_yoloworld_sam3.yaml",
    4: "configs/baselines/baseline4_gdino_sam3.yaml",
    5: "configs/baselines/baseline5_yoloworld_sam2.yaml",
    6: "configs/baselines/baseline6_gdino_sam2.yaml",
    7: "configs/baselines/baseline7_sam3_incremental.yaml",
    8: "configs/baselines/baseline8_gdino_incremental.yaml",
    9: "configs/baselines/baseline9_yoloworld_incremental.yaml",
}

def parse_args():
    """Parse CLI args and merge baseline, category, and override configs."""
    p = argparse.ArgumentParser(
        description="AeroTrack unified OVTS evaluation entry point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--dataset", dest="dataset", type=str, default="aero_ovts", choices=DATASET_NAMES,
        help=f"Target dataset (default: aero_ovts): {DATASET_NAMES}",
    )
    p.add_argument(
        "--baseline", type=int, required=True, choices=[1,2,3,4,5,6,7,8,9],
        help="Baseline id (1-9 supported; v1 main experiments use BL2-BL6)",
    )

    p.add_argument("--config",   type=str, default=None, help="YAML config path (auto-derived from --baseline if omitted)")
    p.add_argument("--gt_json",  type=str, default=None, help="Override default GT JSON path")
    p.add_argument("--seq_root", type=str, default=None, help="Override default sequence root")
    p.add_argument("--output",   type=str, default=None, help="Output root (default: outputs/{dataset}/bl{baseline})")
    p.add_argument(
        "--json", dest="pred_json", type=str, default=None,
        help="Existing predictions JSON; skip inference and evaluate metrics only",
    )

    p.add_argument("--categories", type=str, default=None, help="Evaluate only these category ids, comma-separated (e.g. 1,3)")
    p.add_argument("--text",       type=str, default=None,
                   help="Filter by category name(s), comma-separated (e.g. person or car,truck); "
                        "resolves to category_id and keeps videos with GT for that category")
    p.add_argument("--max", dest="max_videos", type=int, default=None, help="Limit number of videos (debug)")

    p.add_argument("--sam3_ckpt",  type=str, default=None)
    p.add_argument("--yolo_ckpt",  type=str, default=None)
    p.add_argument("--gdino_ckpt", type=str, default=None)
    p.add_argument("--verbose",    action="store_true", default=False)
    p.add_argument("--nooutput",   dest="no_vis", action="store_true", default=False,
                   help="Skip visualization output (saves disk and time)")
    p.add_argument("--jpg", action="store_true", default=False,
                   help="Write visualization as JPG frame folders instead of MP4")
    p.add_argument("--memory",     type=float, default=None,
                   help="Reserve GPU memory in GB")
    p.add_argument("--relocation_interval", type=int, default=None,
                   help="Key-frame / hard-restart interval; overrides baseline YAML")

    p.add_argument("--max_objects",          type=int,   default=None)
    p.add_argument("--max_trk_keep_alive",   type=int,   default=None)
    p.add_argument("--max_missing_segments", type=int,   default=None)
    p.add_argument("--max_dets",             type=int,   default=None)
    p.add_argument("--nms_iou_thr",          type=float, default=None)
    p.add_argument("--match_iou_thr",        type=float, default=None)
    p.add_argument("--box_threshold",        type=float, default=None)
    p.add_argument("--text_threshold",       type=float, default=None)
    p.add_argument("--score_thr",            type=float, default=None)
    p.add_argument("--mask_nms_iou_thr",     type=float, default=None)
    p.add_argument("--max_mask_area_ratio",  type=float, default=None)
    p.add_argument("--new_det_thresh",       type=float, default=None)
    from aerotrack_core.evaluation.tracker_builder import parse_bool
    p.add_argument("--use_track_lifecycle", type=parse_bool, default=None)
    p.add_argument("--lost_track_ttl_segments", type=int, default=None)
    p.add_argument("--lifecycle_match_score_thr", type=float, default=None)
    p.add_argument("--lifecycle_center_gate", type=float, default=None)
    p.add_argument("--lifecycle_area_ratio_min", type=float, default=None)
    p.add_argument("--lifecycle_area_ratio_max", type=float, default=None)

    p.add_argument("--gdino_infer_scale",    type=float, default=None,
                   help="GDINO infer scale; 1.0 = 800/1333 (default). Used by BL4/BL6")
    p.add_argument("--yolo_infer_scale",     type=float, default=None,
                   help="YOLO-World infer scale; 1.0 = 640 (default). Used by BL3/BL5")

    args = p.parse_args()
    from aerotrack_core.evaluation import load_config

    # Preserve the raw CLI --text before YAML flattening overwrites it.
    _cli_text = args.text

    _CLI_TUNING = [
        "relocation_interval", "max_objects", "max_trk_keep_alive",
        "max_missing_segments", "max_dets", "nms_iou_thr", "match_iou_thr",
        "box_threshold", "text_threshold",
        "score_thr", "mask_nms_iou_thr", "max_mask_area_ratio", "new_det_thresh",
        "use_track_lifecycle", "lost_track_ttl_segments", "lifecycle_match_score_thr",
        "lifecycle_center_gate", "lifecycle_area_ratio_min", "lifecycle_area_ratio_max",
        "gdino_infer_scale", "yolo_infer_scale",
    ]
    _cli_tuning_overrides = {k: getattr(args, k, None) for k in _CLI_TUNING}
    args._cli_tuning_overrides = _cli_tuning_overrides

    if args.config is None:
        args.config = _BASELINE_CONFIGS[args.baseline]
        print(f"  [Auto] config = {args.config}")

    cfg = load_config(args.config)
    for k, v in cfg.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    for nested_key in ["detection", "sam3_tuning", "adaptive_trigger"]:
        if nested_key in cfg and getattr(args, nested_key, None) is None:
            setattr(args, nested_key, cfg[nested_key])

    if _cli_text:
        _cat_names = [n.strip().lower() for n in _cli_text.split(",") if n.strip()]
        _cat_file  = f"{_cat_names[0]}.yaml" if len(_cat_names) == 1 else "auto.yaml"
    else:
        _cat_file = "auto.yaml"
    _cat_cfg_path = os.path.join(_ROOT, "configs", "categories", _cat_file)
    if os.path.exists(_cat_cfg_path):
        _cat_cfg = load_config(_cat_cfg_path)
        for k, v in _cat_cfg.items():
            setattr(args, k, v)
        print(f"  [Auto] category config = configs/categories/{_cat_file}")

    for k, v in _cli_tuning_overrides.items():
        if v is not None:
            setattr(args, k, v)
            print(f"  [CLI override] {k} = {v}")

    if args.baseline in (2, 3, 4, 5, 6) and _cli_tuning_overrides.get("max_missing_segments") is not None:
        print("  [WARN] max_missing_segments applies to BL7-BL9 incremental only; ignored for BL2-BL6")
        args.max_missing_segments = None
        args._cli_tuning_overrides["max_missing_segments"] = None

    # Restore CLI --text so YAML tracking.text is not treated as a category filter.
    args.text = _cli_text

    return args



def _fmt(val, fmt=".4f"):
    if val is None or val == "N/A": return "N/A"
    if isinstance(val, (int, float)): return format(float(val), fmt)
    return str(val)

def _fmt_pct(val):
    if val is None or val == "N/A": return "N/A"
    return f"{float(val)*100:.1f}%"


def _metric_value(data, *keys):
    """Return the first non-None metric value from nested dict keys."""
    for key in keys:
        if isinstance(data, dict) and data.get(key) is not None:
            return data.get(key)
    return None


def _metric_fallback(value, fallback):
    return fallback if value is None else value


def _build_summary_table(all_results: dict) -> list:
    """Build the per-category summary table in the requested metric order."""
    width = 132
    lines = [
        "  [ AeroTrack Evaluation | Video AP + HOTA + Frame AP ]",
        "  " + "=" * width,
        "  {:<14} | {:<13} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<8} | {:<10} | {:<10}  {}".format(
            "Category", "GT/Pred", "HOTA", "AP", "AP25", "AP50", "AP75",
            "DetA", "AssA", "Mask J", "F-AP@.25", "F-AP@.50"
        ),
        "  " + "-" * width,
    ]

    metric_keys = ["hota", "ap", "ap25", "ap50", "ap75", "deta", "assa", "mask_j", "fap25", "fap50"]
    weighted = {k: {"sum": 0.0, "weight": 0} for k in metric_keys}
    total_gt = total_pred = 0

    for cname, r in all_results.items():
        video_ap_per = {c["category_id"]: c for c in r.get("video_ap", {}).get("per_category", [])}
        frame_ap_per = {c["category_id"]: c for c in r.get("frame_ap", {}).get("per_category", [])}
        j_per = {c["category_id"]: c for c in r.get("seg_metrics", {}).get("per_category", [])}
        hota_per = {c["category_id"]: c for c in (r.get("hota") or {}).get("per_category", [])}

        for row in r.get("per_category", []):
            if row["gt_tracks"] == 0 and row["pred_tracks"] == 0:
                continue
            cid = row["category_id"]
            gt_count = row["gt_tracks"]
            total_gt += gt_count
            total_pred += row["pred_tracks"]

            hota_entry = hota_per.get(cid) or {}
            vap_entry = video_ap_per.get(cid) or {}
            fap_entry = frame_ap_per.get(cid) or {}

            hota = _metric_fallback(_metric_value(hota_entry, "HOTA"), _metric_value(r.get("hota") or {}, "HOTA"))
            deta = _metric_fallback(_metric_value(hota_entry, "DetA"), _metric_value(r.get("hota") or {}, "DetA"))
            assa = _metric_fallback(_metric_value(hota_entry, "AssA"), _metric_value(r.get("hota") or {}, "AssA"))
            mask_j = _metric_fallback(_metric_value(j_per.get(cid) or {}, "J"), _metric_value(r.get("seg_metrics") or {}, "J"))

            video_ap = _metric_fallback(_metric_value(vap_entry, "AP"), _metric_value(r.get("video_ap") or {}, "AP"))
            video_ap25 = _metric_fallback(_metric_value(vap_entry, "AP25"), _metric_value(r.get("video_ap") or {}, "AP25"))
            video_ap50 = _metric_fallback(_metric_value(vap_entry, "AP50"), _metric_value(r.get("video_ap") or {}, "AP50"))
            video_ap75 = _metric_fallback(_metric_value(vap_entry, "AP75"), _metric_value(r.get("video_ap") or {}, "AP75"))
            fap25 = _metric_fallback(_metric_value(fap_entry, "F-AP@0.25", "F_AP25"), _metric_value(r.get("frame_ap") or {}, "F-AP@0.25", "F_AP25"))
            fap50 = _metric_fallback(_metric_value(fap_entry, "F-AP@0.50", "F_AP50"), _metric_value(r.get("frame_ap") or {}, "F-AP@0.50", "F_AP50"))

            values = {
                "hota": hota, "ap": video_ap, "ap25": video_ap25, "ap50": video_ap50, "ap75": video_ap75,
                "deta": deta, "assa": assa, "mask_j": mask_j, "fap25": fap25, "fap50": fap50,
            }
            for key, val in values.items():
                if val is not None:
                    weighted[key]["sum"] += float(val) * gt_count
                    weighted[key]["weight"] += gt_count

            gt_pred = f"{gt_count}/{row['pred_tracks']}"
            lines.append(
                f"  {cname:<14} | {gt_pred:<13} "
                f"| {_fmt(hota):<8} | {_fmt(video_ap):<8} | {_fmt(video_ap25):<8} "
                f"| {_fmt(video_ap50):<8} | {_fmt(video_ap75):<8} | {_fmt(deta):<8} "
                f"| {_fmt(assa):<8} | {_fmt(mask_j):<10} | {_fmt(fap25):<10}  {_fmt(fap50)}"
            )

    def _wavg(key):
        item = weighted[key]
        return item["sum"] / item["weight"] if item["weight"] > 0 else None

    lines.append("  " + "-" * width)
    overall_gt_pred = f"{total_gt}/{total_pred}"
    lines.append(
        f"  {'OVERALL(weighted)':<14} | {overall_gt_pred:<13} "
        f"| {_fmt(_wavg('hota')):<8} | {_fmt(_wavg('ap')):<8} | {_fmt(_wavg('ap25')):<8} "
        f"| {_fmt(_wavg('ap50')):<8} | {_fmt(_wavg('ap75')):<8} | {_fmt(_wavg('deta')):<8} "
        f"| {_fmt(_wavg('assa')):<8} | {_fmt(_wavg('mask_j')):<10} | {_fmt(_wavg('fap25')):<10}  {_fmt(_wavg('fap50'))}"
    )
    lines.append("  " + "=" * width)
    lines += [
        "  AP/AP25/AP50/AP75 : video-level track AP metrics.",
        "  HOTA/DetA/AssA    : tracking quality and detection/association decomposition.",
        "  Mask J            : mask IoU on matched tracks.",
        "  F-AP@.25/.50      : frame-level AP; does not evaluate ID association.",
        "  OVERALL           : weighted by per-category GT track count.",
        "",
    ]
    return lines


def _update_tracker_params(tracker, cat_name: str, cli_overrides: dict = None) -> None:
    """Apply per-category YAML tuning to a reused tracker instance."""
    if tracker is None:
        return
    cat_cfg_path = os.path.join(_ROOT, "configs", "categories", f"{cat_name.lower()}.yaml")
    if not os.path.exists(cat_cfg_path):
        cat_cfg_path = os.path.join(_ROOT, "configs", "categories", "auto.yaml")
    if not os.path.exists(cat_cfg_path):
        return
    from aerotrack_core.evaluation import load_config

    cat_cfg = load_config(cat_cfg_path)
    _UPDATABLE = [
        "max_objects", "new_det_thresh", "max_trk_keep_alive",
        "relocation_interval", "score_thr", "max_dets", "match_iou_thr",
        "box_threshold", "text_threshold", "nms_iou_thr", "max_missing_segments",
        "mask_nms_iou_thr", "max_mask_area_ratio",
        "use_track_lifecycle", "lost_track_ttl_segments", "lifecycle_match_score_thr",
        "lifecycle_center_gate", "lifecycle_area_ratio_min", "lifecycle_area_ratio_max",
        "yolo_infer_scale", "gdino_infer_scale",
    ]
    updated = []
    for attr in _UPDATABLE:
        if cli_overrides and cli_overrides.get(attr) is not None:
            val = cli_overrides[attr]
        elif attr in cat_cfg:
            val = cat_cfg[attr]
        else:
            continue
        if attr == "yolo_infer_scale":
            if hasattr(tracker, "set_infer_scale") and (
                hasattr(tracker, "yolo_model") or hasattr(tracker, "yolo_checkpoint")
            ):
                tracker.set_infer_scale(val)
                updated.append(f"{attr}={val}")
            continue
        if attr == "gdino_infer_scale":
            if hasattr(tracker, "set_infer_scale") and (
                hasattr(tracker, "gdino_model") or hasattr(tracker, "gdino_checkpoint")
            ):
                tracker.set_infer_scale(val)
                updated.append(f"{attr}={val}")
            continue
        if hasattr(tracker, attr):
            setattr(tracker, attr, val)
            updated.append(f"{attr}={val}")

    # Some SAM3 parameters are copied into the loaded model during
    # load_models().  In loop-mode we reuse one tracker across categories, so
    # changing only tracker.<attr> is not enough for these model-internal knobs.
    model = getattr(getattr(tracker, "video_predictor", None), "model", None)
    if model is not None:
        if hasattr(tracker, "max_objects") and hasattr(model, "max_num_objects"):
            model.max_num_objects = tracker.max_objects
        if hasattr(tracker, "new_det_thresh") and hasattr(model, "new_det_thresh"):
            model.new_det_thresh = tracker.new_det_thresh
        if hasattr(tracker, "max_trk_keep_alive") and hasattr(model, "max_trk_keep_alive"):
            model.max_trk_keep_alive = tracker.max_trk_keep_alive

    src = os.path.basename(cat_cfg_path)
    if updated:
        print(f"  [CatCfg] {cat_name} <- {src}: {', '.join(updated)}")
    else:
        print(f"  [CatCfg] {cat_name} <- {src}: no updatable params (tracker attributes mismatch)")


def _reserve_gpu_memory(gb: float):
    """Reserve GPU memory in the PyTorch caching allocator."""
    import torch
    dev = torch.device("cuda")
    bytes_to_alloc = int(gb * 1024**3)
    _hold = torch.empty(bytes_to_alloc // 4, dtype=torch.float32, device=dev)
    actual_gb = torch.cuda.memory_allocated(dev) / 1024**3
    del _hold
    print(f"  [Memory] Reserved {actual_gb:.1f} GB GPU memory (PyTorch cache pool)")


def _aggregate_sys_metrics(all_results: dict) -> dict:
    """Aggregate per-category sys_metrics into a global view.

    Times sum, throughputs are recomputed from the summed totals (so they
    weight by actual workload), and peak GPU/CPU take the max across
    categories (each is already a peak inside its own category run).
    """
    total_time = 0.0
    total_masks = 0
    total_frames = 0
    peak_gpu = 0.0
    peak_cpu = 0.0
    for r in all_results.values():
        sm = r.get("sys_metrics") or {}
        total_time += float(sm.get("total_time_s") or 0.0)
        total_masks += int(sm.get("total_masks") or 0)
        total_frames += int(sm.get("total_frames") or 0)
        peak_gpu = max(peak_gpu, float(sm.get("peak_gpu_mb") or 0.0))
        peak_cpu = max(peak_cpu, float(sm.get("peak_cpu_mb") or 0.0))
    mps = total_masks / total_time if total_time > 0 else 0.0
    fps = total_frames / total_time if total_time > 0 else 0.0
    ms_per_frame = (1000.0 / fps) if fps > 0 else 0.0
    return {
        "total_time_s":  round(total_time, 2),
        "total_frames":  total_frames,
        "total_masks":   total_masks,
        "fps":           round(fps, 2),
        "mps":           round(mps, 2),
        "ms_per_frame":  round(ms_per_frame, 2),
        "peak_gpu_mb":   round(peak_gpu, 0),
        "peak_cpu_mb":   round(peak_cpu, 0),
    }


def main():
    """Run category-wise inference and metric evaluation."""
    _t_main_start = time.perf_counter()
    args = parse_args()
    from aerotrack_core.evaluation import build_tracker, run_inference
    from aerotrack_core.evaluation.metrics import (
        run_ytvis_map_evaluation,
        run_hota_evaluation,
        compute_odr,
        compute_mask_j_score,
        compute_per_category_stats,
        compute_frame_segm_ap,
    )

    if args.memory is not None:
        os.environ["AEROTRACK_MEMORY_RESERVE"] = "1"
        _reserve_gpu_memory(args.memory)

    ds_cfg       = get_dataset_config(args.dataset)
    gt_json_path = args.gt_json  or ds_cfg["gt_json"]
    seq_root     = args.seq_root or ds_cfg["seq_root"]
    hota_name    = ds_cfg["hota_name"]

    if args.output:
        out_root = os.path.join(_ROOT, args.output)
    else:
        out_root = os.path.join(_ROOT, "outputs", args.dataset, f"bl{args.baseline}")

    print(f"\n{'=' * 70}")
    print(f"  AeroTrack OVTS Evaluation")
    print(f"  Dataset  : {args.dataset.upper()}")
    print(f"  Baseline : {args.baseline}")
    print(f"{'=' * 70}")
    print(f"  GT JSON  : {gt_json_path}")
    print(f"  Seq Root : {seq_root}")
    print(f"  Output root : {out_root}")

    with open(gt_json_path, encoding="utf-8") as f:
        gt_data = json.load(f)

    videos_all = gt_data["videos"]
    categories = gt_data["categories"]
    cat_map    = {c["id"]: c["name"] for c in categories}

    n_ann     = len(gt_data["annotations"])
    n_iscrowd = sum(1 for a in gt_data["annotations"] if a.get("iscrowd", 0) == 1)
    print(f"  Videos      : {len(videos_all)}")
    print(f"  Annotations : {n_ann} (iscrowd=0: {n_ann - n_iscrowd}, iscrowd=1: {n_iscrowd})")

    active_cat_ids = {
        a["category_id"] for a in gt_data["annotations"]
        if a.get("iscrowd", 0) == 0
    }

    if getattr(args, "text", None):
        name_to_id = {c["name"].lower(): c["id"] for c in categories}
        req_names  = [n.strip().lower() for n in args.text.split(",") if n.strip()]
        text_ids   = set()
        for name in req_names:
            if name in name_to_id:
                text_ids.add(name_to_id[name])
            else:
                print(f"  [WARN] --text: category '{name}' not in dataset, skipped")
        if not text_ids:
            raise ValueError(f"--text '{args.text}': no category names exist in dataset")
        eval_cat_ids = sorted(text_ids & active_cat_ids)
        print(f"  --text   : {req_names} -> category_ids = {eval_cat_ids}")
        loop_mode = True
    elif args.categories:
        eval_cat_ids = sorted({int(c) for c in args.categories.split(",")} & active_cat_ids)
        loop_mode    = True
    else:
        eval_cat_ids = sorted(active_cat_ids)
        loop_mode    = True
        print(f"  Mode       : per-category loop with resume support")

    eval_cat_names = {cid: cat_map[cid] for cid in eval_cat_ids if cid in cat_map}
    eval_cat_ids   = sorted(eval_cat_names.keys())
    print(f"  Categories : {len(categories)} (JSON) / {len(active_cat_ids)} (with GT)")
    print(f"  Eval set   : {eval_cat_names}")

    if not args.pred_json:
        tracker = build_tracker(args)
    else:
        tracker = None

    all_results = {}

    for cid in eval_cat_ids:
        cname = eval_cat_names[cid]

        cat_out_dir = os.path.join(out_root, cname)

        result_json_path = os.path.join(cat_out_dir, "eval_results.json")

        os.makedirs(cat_out_dir, exist_ok=True)
        vis_dir = os.path.join(cat_out_dir, "vis")

        print("\n" + "-" * 70)
        _loop_label = " | loop mode" if loop_mode else ""
        print(f"  Category: {cname} (id={cid})" + _loop_label)
        print(f"  Output  : {cat_out_dir}")

        eval_cat_id_set  = {cid}
        relevant_vid_ids = {
            a["video_id"] for a in gt_data["annotations"]
            if a.get("category_id") == cid and a.get("iscrowd", 0) == 0
        }
        cat_videos = [v for v in videos_all if v["id"] in relevant_vid_ids]

        if args.max_videos and args.max_videos < len(cat_videos):
            cat_videos = cat_videos[:args.max_videos]
            print(f"  [DEBUG] Video limit = {len(cat_videos)}")

        eval_vid_ids = {v["id"] for v in cat_videos}
        gt_data_eval = dict(gt_data)
        gt_data_eval["videos"]      = [v for v in gt_data["videos"] if v["id"] in eval_vid_ids]
        gt_data_eval["annotations"] = [
            a for a in gt_data["annotations"]
            if a.get("video_id") in eval_vid_ids and a.get("category_id") in eval_cat_id_set
        ]
        gt_data_eval["categories"]  = [c for c in gt_data["categories"] if c["id"] == cid]

        result_signature = _category_result_signature(args, cname, cid, cat_videos, gt_data_eval)

        # Reuse cached results only when the run signature matches.
        if loop_mode and os.path.exists(result_json_path):
            with open(result_json_path, encoding="utf-8") as f:
                cached_result = json.load(f)
            if cached_result.get("signature") == result_signature:
                print(f"\n  [SKIP] {cname}: cached result with matching signature ({result_json_path})")
                all_results[cname] = cached_result
                continue
            print(f"\n  [RE-RUN] {cname}: cached result signature mismatch, re-evaluating")

        n_non_crowd = sum(1 for a in gt_data_eval["annotations"] if a.get("iscrowd", 0) == 0)
        n_crowd     = len(gt_data_eval["annotations"]) - n_non_crowd
        print(f"  Eval scope : {len(cat_videos)} videos, {n_non_crowd} GT tracks (iscrowd=1: {n_crowd})")

        if args.pred_json:
            with open(args.pred_json, encoding="utf-8") as f:
                _all = json.load(f)
            all_pred = [
                p for p in _all
                if p.get("category_id") == cid and p.get("video_id") in eval_vid_ids
            ]
            print(
                f"  Loaded predictions: {len(all_pred)} tracks"
                f" (from {args.pred_json}, category={cid}, videos={len(eval_vid_ids)})"
            )
            throughput_calc = None
        else:
            _update_tracker_params(tracker, cname, cli_overrides=getattr(args, "_cli_tuning_overrides", None))
            print(f"\n  Stage 1/2: inference")
            all_pred, throughput_calc = run_inference(
                tracker, gt_data_eval, cat_videos, [cid], {cid: cname}, seq_root,
                vis_root=None if args.no_vis else vis_dir,
                vis_as_jpg=args.jpg,
                out_dir=cat_out_dir,
            )
            pred_output = os.path.join(cat_out_dir, "predictions.json")
            with open(pred_output, "w", encoding="utf-8") as f:
                json.dump(all_pred, f)
            print(f"  Predictions: {len(all_pred)} tracks")

        print(f"\n  Stage 2/2: metrics")
        video_ap_results = run_ytvis_map_evaluation(gt_data_eval, all_pred, iou_type="segm")
        frame_ap_results = compute_frame_segm_ap(gt_data_eval, all_pred, uav_adaptive=True)
        seg_results      = compute_mask_j_score(gt_data_eval, all_pred, uav_adaptive=True)
        odr_calc         = compute_odr(gt_data_eval, all_pred)
        hota_results     = run_hota_evaluation(gt_data_eval, all_pred, hota_name, iou_type="segm")
        cat_stats_list   = compute_per_category_stats(gt_data_eval, all_pred)
        sys_metrics      = throughput_calc.get_system_metrics() if throughput_calc else {}

        result = {
            "dataset":         args.dataset,
            "baseline":        args.baseline,
            "signature":       result_signature,
            "category":        cname,
            "category_id":     cid,
            "num_videos":      len(cat_videos),
            "num_predictions": len(all_pred),
            "sys_metrics":     sys_metrics,
            "odr":             odr_calc.get_odr(),
            "video_ap":        video_ap_results,
            "frame_ap":        frame_ap_results,
            "seg_metrics":     seg_results,
            "hota":            hota_results,
            "per_category":    cat_stats_list,
        }
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)

        txt_path = os.path.join(cat_out_dir, "eval_results.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            hota_v = (hota_results or {}).get("HOTA")
            deta_v = (hota_results or {}).get("DetA")
            assa_v = (hota_results or {}).get("AssA")
            mask_j_v = seg_results.get("J")
            f.write(f"AeroTrack OVTS Evaluation Results - {cname}\n")
            f.write(f"Dataset  : {args.dataset.upper()}\nBaseline : {args.baseline}\n")
            f.write("=" * 60 + "\n")
            f.write(f"HOTA       = {_fmt(hota_v)}\n")
            f.write(f"AP         = {_fmt((video_ap_results or {}).get('AP'))}\n")
            f.write(f"AP25       = {_fmt((video_ap_results or {}).get('AP25'))}\n")
            f.write(f"AP50       = {_fmt((video_ap_results or {}).get('AP50'))}\n")
            f.write(f"AP75       = {_fmt((video_ap_results or {}).get('AP75'))}\n")
            f.write(f"DetA       = {_fmt(deta_v)}\n")
            f.write(f"AssA       = {_fmt(assa_v)}\n")
            f.write(f"Mask J     = {_fmt(mask_j_v)}\n")
            f.write(f"F-AP@0.25  = {_fmt(_metric_value(frame_ap_results, 'F-AP@0.25', 'F_AP25'))}\n")
            f.write(f"F-AP@0.50  = {_fmt(_metric_value(frame_ap_results, 'F-AP@0.50', 'F_AP50'))}\n")
            if throughput_calc:
                sm = sys_metrics
                f.write(f"MPS        = {sm.get('mps', 0):.2f}\n")
                f.write(f"Peak GPU   = {sm.get('peak_gpu_mb', 0):.0f} MB\n")
                f.write(f"Total Time = {sm.get('total_time_s', 0):.1f} s\n")

        all_results[cname] = result

        mps_str = f"  MPS={sys_metrics.get('mps', 0):.2f}" if sys_metrics else ""
        gpu_str = f"  PeakGPU={sys_metrics.get('peak_gpu_mb', 0):.0f}MB" if sys_metrics else ""
        print(
            f"\n  [{cname}] Done:"
            f"HOTA={_fmt((hota_results or {}).get('HOTA'))}  "
            f"AP={_fmt((video_ap_results or {}).get('AP'))}  "
            f"AP50={_fmt((video_ap_results or {}).get('AP50'))}  "
            f"F-AP@0.25={_fmt(_metric_value(frame_ap_results, 'F-AP@0.25', 'F_AP25'))}"
            f"{mps_str}{gpu_str}"
        )

    summary_lines = _build_summary_table(all_results)
    summary_str   = "\n".join(summary_lines)

    print(f"\n{'=' * 66}")
    print(f"  AeroTrack Evaluation Summary | {args.dataset.upper()} | BL{args.baseline}")
    print(f"{'=' * 66}")
    print(summary_str)

    perf = _aggregate_sys_metrics(all_results)
    perf_lines = [
        "  " + "-" * 74,
        f"  Throughput     : {perf['fps']:.2f} FPS  |  {perf['mps']:.2f} MPS  "
        f"|  {perf['ms_per_frame']:.1f} ms/frame",
        f"  Resource       : Peak GPU {perf['peak_gpu_mb']:.0f} MB  "
        f"|  Peak CPU {perf['peak_cpu_mb']:.0f} MB",
        f"  Inference      : {perf['total_time_s']:.1f} s over "
        f"{perf['total_frames']} frames / {perf['total_masks']} masks",
    ]
    perf_str = "\n".join(perf_lines)
    print(perf_str)

    os.makedirs(out_root, exist_ok=True)
    summary_txt = os.path.join(out_root, "eval_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"AeroTrack Evaluation Summary | {args.dataset.upper()} | BL{args.baseline}\n")
        f.write(summary_str)
        f.write("\n" + perf_str + "\n")

    _wall_total = time.perf_counter() - _t_main_start
    _h, _r = divmod(int(_wall_total), 3600)
    _m, _s = divmod(_r, 60)
    _wall_str = f"{_h}h{_m:02d}m{_s:02d}s" if _h else f"{_m}m{_s:02d}s"

    print(f"\n{'=' * 70}")
    print(f"  Evaluation complete | {args.dataset.upper()} | Baseline {args.baseline}")
    print(f"  Summary file      : {summary_txt}")
    for cname, r in all_results.items():
        _sub = os.path.join(out_root, cname)
        print(f"  {cname:<14} : {_sub}/")
    if not args.no_vis and args.jpg:
        print(f"  JPG output        : {out_root}")
    print(f"  Total wall time : {_wall_str}  ({_wall_total:.1f} s)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
