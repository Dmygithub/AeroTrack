"""Single-video demo entry point for AeroTrack."""

import os
# Set before importing torch to reduce CUDA allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import glob
import sys
import gc
import cv2
import time
import importlib.util
import numpy as np

os.environ["LOG_LEVEL"] = "ERROR"
import logging
logging.getLogger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# Import tracker_builder directly to avoid heavy evaluation dependencies.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_TRACKER_BUILDER = os.path.join(_ROOT, "aerotrack_core", "evaluation", "tracker_builder.py")
_spec = importlib.util.spec_from_file_location("aerotrack_tracker_builder", _TRACKER_BUILDER)
_tracker_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tracker_builder)
build_tracker = _tracker_builder.build_tracker
parse_bool = _tracker_builder.parse_bool
load_config = _tracker_builder.load_config

DEFAULT_VIDEO             = "data/videos/VisDrone2019-VID-val/sequences/uav0000137_00458_v"
DEFAULT_TEXT              = "car"
DEFAULT_PROMPT_FRAME      = 0
DEFAULT_OUTPUT            = "outputs/videos"
DEFAULT_RELOCATION_INTERVAL = 0
DEFAULT_VERBOSE             = False
DEFAULT_SHOW_PROGRESS       = True

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

_CLI_TUNING = [
    "relocation_interval", "max_objects", "max_trk_keep_alive",
    "max_missing_segments", "max_dets", "nms_iou_thr", "match_iou_thr",
    "box_threshold", "text_threshold",
    "score_thr", "mask_nms_iou_thr", "max_mask_area_ratio", "new_det_thresh",
    "use_track_lifecycle", "lost_track_ttl_segments", "lifecycle_match_score_thr",
    "lifecycle_center_gate", "lifecycle_area_ratio_min", "lifecycle_area_ratio_max",
    "gdino_infer_scale", "yolo_infer_scale",
]


def parse_args():
    """Parse CLI args and merge baseline, category, and override configs."""
    parser = argparse.ArgumentParser(description="AeroTrack unified demo")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (e.g. configs/baselines/baseline4_gdino_sam3.yaml)")
    parser.add_argument("--baseline", type=int, choices=[1,2,3,4,5,6,7,8,9], default=None,
                        help="Choose baseline. V1 paper scope: 2-6; 1 and 7-9 are engineering/v2.")
    parser.add_argument("--video", type=str, default=None, help="Input video or frame folder")
    parser.add_argument("--text", type=str, default=None, help="Text prompt")
    parser.add_argument("--prompt_frame", type=int, default=None, help="Frame index for prompt")
    parser.add_argument("--output", type=str, default=None, help="Output directory or file")
    parser.add_argument("--jpg", action="store_true", default=False,
                        help="Save visualization as a folder of JPG frames instead of MP4")
    parser.add_argument("--sam3_ckpt", type=str, default=None)
    parser.add_argument("--sam2_ckpt", type=str, default=None)
    parser.add_argument("--yolo_ckpt", type=str, default=None)
    parser.add_argument("--gdino_ckpt", type=str, default=None)
    parser.add_argument("--verbose", action="store_true", default=None,
                        help="Enable verbose logs")
    parser.add_argument("--no_progress", action="store_true", default=None,
                        help="Disable progress bar output")
    parser.add_argument("--relocation_interval", type=int, default=None)

    parser.add_argument("--max_objects",          type=int,   default=None)
    parser.add_argument("--max_trk_keep_alive",   type=int,   default=None)
    parser.add_argument("--max_missing_segments", type=int,   default=None)
    parser.add_argument("--max_dets",             type=int,   default=None)
    parser.add_argument("--nms_iou_thr",          type=float, default=None)
    parser.add_argument("--match_iou_thr",        type=float, default=None)
    parser.add_argument("--box_threshold",        type=float, default=None)
    parser.add_argument("--text_threshold",       type=float, default=None)
    parser.add_argument("--score_thr",            type=float, default=None)
    parser.add_argument("--mask_nms_iou_thr",     type=float, default=None)
    parser.add_argument("--max_mask_area_ratio",  type=float, default=None)
    parser.add_argument("--new_det_thresh",       type=float, default=None)
    parser.add_argument("--use_track_lifecycle", type=parse_bool, default=None)
    parser.add_argument("--lost_track_ttl_segments", type=int, default=None)
    parser.add_argument("--lifecycle_match_score_thr", type=float, default=None)
    parser.add_argument("--lifecycle_center_gate", type=float, default=None)
    parser.add_argument("--lifecycle_area_ratio_min", type=float, default=None)
    parser.add_argument("--lifecycle_area_ratio_max", type=float, default=None)
    parser.add_argument("--gdino_infer_scale",    type=float, default=None,
                        help="GDINO infer scale; 1.0 = 800/1333 (default). Used by BL4/BL6")
    parser.add_argument("--yolo_infer_scale",     type=float, default=None,
                        help="YOLO-World infer scale; 1.0 = 640 (default). Used by BL3/BL5")
    args = parser.parse_args()

    cli_text = args.text
    cli_tuning_overrides = {k: getattr(args, k, None) for k in _CLI_TUNING}

    cfg = {
        "baseline": 3,
        "video": DEFAULT_VIDEO, "text": DEFAULT_TEXT,
        "prompt_frame": DEFAULT_PROMPT_FRAME, "output": DEFAULT_OUTPUT,
        "relocation_interval": DEFAULT_RELOCATION_INTERVAL,
        "sam3_ckpt": "checkpoints/sam3.pt",
        "sam2_ckpt": "checkpoints/sam2.1_hiera_large.pt",
        "yolo_ckpt": "checkpoints/l_stage1-7d280586.pth",
        "gdino_ckpt": "checkpoints/groundingdino_swint_ogc.pth",
        "verbose": DEFAULT_VERBOSE,
        "show_progress": DEFAULT_SHOW_PROGRESS,
    }

    config_path = args.config
    if config_path is None and args.baseline is not None:
        config_path = _BASELINE_CONFIGS.get(args.baseline)
        print(f"  [Auto] config = {config_path}")
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg.update(load_config(config_path))

    # Category YAML follows the raw CLI --text value, same as evaluate.py.
    if cli_text:
        cat_names = [n.strip().lower() for n in cli_text.split(",") if n.strip()]
        cat_file = f"{cat_names[0]}.yaml" if len(cat_names) == 1 else "auto.yaml"
    else:
        cat_file = "auto.yaml"
    cat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "categories")
    cat_path = os.path.join(cat_dir, cat_file)
    if not os.path.exists(cat_path):
        cat_file = "auto.yaml"
        cat_path = os.path.join(cat_dir, cat_file)
    if os.path.exists(cat_path):
        cfg.update(load_config(cat_path))
        print(f"  [Auto] category config = configs/categories/{cat_file}")

    for key in ("baseline", "video", "text", "prompt_frame", "output",
                "sam3_ckpt", "sam2_ckpt", "yolo_ckpt", "gdino_ckpt", "verbose"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    for key, val in cli_tuning_overrides.items():
        if val is not None:
            cfg[key] = val
            print(f"  [CLI override] {key} = {val}")
    if cfg.get("baseline") in (2, 3, 4, 5, 6) and cli_tuning_overrides.get("max_missing_segments") is not None:
        print("  [WARN] max_missing_segments applies to BL7-BL9 incremental only; ignored for BL2-BL6")
        cfg["max_missing_segments"] = None
    if args.no_progress is not None and args.no_progress:
        cfg["show_progress"] = False

    for key, val in cfg.items():
        setattr(args, key, val)

    return args

def overlay_masks(frame, masks, track_ids, alpha=0.5):
    """Overlay colored masks, boxes, and track IDs on one frame."""
    overlay = frame.copy()
    colors = {}
    for tid in track_ids:
        tid_int = int(tid)
        if tid_int not in colors:
            rng = np.random.default_rng(1000 + tid_int)
            colors[tid_int] = rng.integers(0, 255, size=3).tolist()

    for mask, tid in zip(masks, track_ids):
        tid_int = int(tid)
        color = colors[tid_int]
        if mask.ndim == 3:
            mask = np.squeeze(mask, axis=0)
        mask_bool = mask.astype(bool)
        if mask_bool.shape != frame.shape[:2]:
            mask_bool = cv2.resize(
                mask_bool.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            
        blended = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        overlay[mask_bool] = np.clip(blended, 0, 255).astype(np.uint8)

        y_coords, x_coords = np.where(mask_bool)
        if len(y_coords) > 0:
            y1, y2 = y_coords.min(), y_coords.max()
            x1, x2 = x_coords.min(), x_coords.max()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{tid_int}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
    return overlay.astype(np.uint8)

def resolve_output_path(output_arg, input_video, text, baseline, as_jpg=False):
    """Resolve the final visualization output path."""
    safe_text = text.replace(" ", "_").replace(",", "_")
    base_name = os.path.splitext(os.path.basename(input_video.rstrip(os.sep)))[0]
    stem = f"{base_name}_{safe_text}_BL{baseline}"
    file_name = f"{stem}.mp4"
    if output_arg is None:
        output_dir = "outputs/videos"
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, stem if as_jpg else file_name)
    output_arg = os.path.expanduser(output_arg)
    _, ext = os.path.splitext(output_arg)
    if as_jpg:
        if output_arg.endswith(os.sep) or ext == "":
            os.makedirs(output_arg, exist_ok=True)
            return os.path.join(output_arg, stem)
        os.makedirs(os.path.dirname(output_arg) or ".", exist_ok=True)
        return os.path.splitext(output_arg)[0]
    if output_arg.endswith(os.sep) or ext == "":
        os.makedirs(output_arg, exist_ok=True)
        return os.path.join(output_arg, file_name)
    parent = os.path.dirname(output_arg)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return output_arg

def _iter_render_frames(video_path, cache_dir, total_frames):
    is_frame_dir = os.path.isdir(video_path)
    if is_frame_dir:
        frame_files = []
        for ext in ("jpg", "jpeg", "png"):
            frame_files.extend(glob.glob(os.path.join(video_path, f"*.{ext}")))
        frame_files = sorted(set(frame_files))
        first_frame = cv2.imread(frame_files[0])
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not ret or first_frame is None:
            raise RuntimeError(f"Failed to read first frame from: {video_path}")

    yield ("meta", first_frame, cap, frame_files if is_frame_dir else None)

    for frame_idx in range(total_frames):
        if is_frame_dir:
            frame = cv2.imread(frame_files[frame_idx])
        else:
            ret, frame = cap.read()
            if not ret:
                break

        cache_file = os.path.join(cache_dir, f"{frame_idx:06d}.npz")
        if os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    masks = data["masks"]
                    track_ids = data["track_ids"]
                if masks.size > 0 and track_ids.size > 0:
                    frame = overlay_masks(frame, masks, track_ids)
            except Exception:
                pass
            os.remove(cache_file)

        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        yield (frame_idx, frame, cap, frame_files if is_frame_dir else None)

    if cap is not None:
        cap.release()

def render_visualization(video_path, cache_dir, total_frames, fps, output_path, as_jpg=False, verbose=False):
    """Render cached tracking results to MP4 or JPG frames."""
    stream = _iter_render_frames(video_path, cache_dir, total_frames)
    _meta, first_frame, _cap, _frame_files = next(stream)
    height, width = first_frame.shape[:2]
    writer = None
    if as_jpg:
        os.makedirs(output_path, exist_ok=True)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if verbose:
        target = output_path if as_jpg else output_path
        print(f"Rendering outputs to {target}...")

    for frame_idx, frame, _cap, _frame_files in stream:
        if as_jpg:
            cv2.imwrite(os.path.join(output_path, f"{frame_idx:06d}.jpg"), frame)
        else:
            writer.write(frame)

    if writer is not None:
        writer.release()
    if verbose:
        print("Rendering complete!")

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

def render_video(video_path, cache_dir, total_frames, fps, output_path, verbose=False):
    """Render cached tracking results to MP4."""
    render_visualization(
        video_path, cache_dir, total_frames, fps, output_path,
        as_jpg=False, verbose=verbose
    )

def main():
    """Run tracking on one video and render the visualization output."""
    args = parse_args()
    if args.verbose:
        print(f"\n=== Initializing Baseline {args.baseline} ===")
    tracker = build_tracker(args)

    result = tracker.run_tracking(
        video_path=args.video, 
        text_prompt=args.text, 
        prompt_frame_idx=args.prompt_frame
    )
    
    if result is None:
        print("Pipeline aborted due to missing target.")
        sys.exit(1)
        
    cache_dir, total_frames, fps, _ = result

    output_path = resolve_output_path(
        args.output, args.video, args.text, args.baseline, as_jpg=args.jpg
    )
    render_visualization(
        args.video, cache_dir, total_frames, fps, output_path,
        as_jpg=args.jpg, verbose=args.verbose
    )
    
if __name__ == "__main__":
    main()
