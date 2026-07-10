"""Main inference loop over videos and categories with optional visualization."""

import gc
import glob
import json
import os
import shutil
import time

import cv2
import numpy as np
import torch

from .throughput_calculator import ThroughputCalculator
from .ytvis_utils import cache_to_ytvis_predictions


def _get_demo_render():
    """Lazily import demo_video.render_visualization."""
    import importlib.util, sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _demo = os.path.join(_root, "demo_video.py")
    if "demo_video" not in sys.modules:
        spec = importlib.util.spec_from_file_location("demo_video", _demo)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules["demo_video"] = mod
    mod = sys.modules["demo_video"]
    return mod.render_visualization


def _render_visualization(
    seq_dir: str,
    cache_dir: str,
    total_frames: int,
    out_path: str,
    fps: float = 30.0,
    as_jpg: bool = False,
) -> None:
    """Render cached tracking outputs via demo_video."""
    os.makedirs(os.path.dirname(out_path) or out_path, exist_ok=True)
    render_visualization = _get_demo_render()
    render_visualization(
        seq_dir, cache_dir, total_frames, fps, out_path,
        as_jpg=as_jpg, verbose=False
    )


_CACHE_SIGNATURE_ATTRS = (
    "relocation_interval",
    "max_objects",
    "new_det_thresh",
    "max_trk_keep_alive",
    "max_dets",
    "nms_iou_thr",
    "match_iou_thr",
    "box_threshold",
    "text_threshold",
    "score_thr",
    "mask_nms_iou_thr",
    "max_mask_area_ratio",
    "infer_scale",
    "yolo_config",
    "gdino_config",
    "_sam3_tuning_cfg",
    "use_track_lifecycle",
    "lost_track_ttl_segments",
    "lifecycle_match_score_thr",
    "lifecycle_center_gate",
    "lifecycle_area_ratio_min",
    "lifecycle_area_ratio_max",
    "sam3_checkpoint",
    "sam2_checkpoint",
    "yolo_checkpoint",
    "gdino_checkpoint",
    "gdino_bert_path",
)


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    return str(value)


def _prediction_cache_signature(
    tracker, video_id, category_id, category_name, seq_dir, num_frames, H, W
):
    params = {}
    for attr in _CACHE_SIGNATURE_ATTRS:
        if hasattr(tracker, attr):
            params[attr] = _jsonable(getattr(tracker, attr))
    return {
        "schema": "aerotrack_tmp_preds_v2",
        "tracker": f"{tracker.__class__.__module__}.{tracker.__class__.__name__}",
        "video_id": int(video_id),
        "category_id": int(category_id),
        "category_name": str(category_name),
        "seq_dir": os.path.abspath(seq_dir),
        "num_frames": int(num_frames),
        "height": int(H),
        "width": int(W),
        "params": params,
    }


def run_inference(
    tracker,
    gt_data: dict,
    videos: list,
    eval_cat_ids: list,
    eval_cat_names: dict,
    seq_root: str,
    vis_root=None,
    vis_as_jpg: bool = False,
    out_dir=None,
) -> tuple:
    """Run tracker inference for each (video, category) pair.

    Returns:
        Tuple of (all_predictions, throughput_calc).
    """
    throughput_calc = ThroughputCalculator()
    all_predictions = []

    tmp_preds_dir = None
    if out_dir is not None:
        tmp_preds_dir = os.path.join(out_dir, "tmp_preds")
        os.makedirs(tmp_preds_dir, exist_ok=True)

    # Index video_id -> category ids with iscrowd=0 GT annotations.
    vid_cat_index: dict = {}
    for ann in gt_data.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        vid_id_ = ann.get("video_id")
        cat_id_ = ann.get("category_id")
        if vid_id_ is not None and cat_id_ is not None:
            vid_cat_index.setdefault(vid_id_, set()).add(cat_id_)

    total_runs = sum(
        1
        for v in videos
        for c in eval_cat_ids
        if c in vid_cat_index.get(v["id"], set())
    )
    run_idx = 0

    for vid_info in videos:
        vid_id     = vid_info["id"]
        vid_name   = vid_info["name"]
        num_frames = vid_info["length"]
        H, W       = vid_info["height"], vid_info["width"]
        vid_seq_root = vid_info.get("seq_root") or seq_root
        seq_dir      = os.path.join(vid_seq_root, vid_name)

        if not os.path.exists(seq_dir):
            print(f"  [SKIP] Sequence directory not found: {seq_dir}")
            continue

        vid_has_cats = vid_cat_index.get(vid_id, set())

        for cat_id in eval_cat_ids:
            cat_name = eval_cat_names[cat_id]

            if cat_id not in vid_has_cats:
                print(f"\n  [SKIP] Video={vid_name}  Category={cat_name} (id={cat_id})  — no GT annotations")
                continue

            run_idx += 1
            print(
                f"\n  [{run_idx}/{total_runs}] "
                f"Video={vid_name}  Category={cat_name} (id={cat_id})"
            )

            tmp_json_path = None
            if tmp_preds_dir:
                tmp_json_path = os.path.join(tmp_preds_dir, f"{vid_name}_{cat_id}.json")
                cache_signature = _prediction_cache_signature(
                    tracker, vid_id, cat_id, cat_name, seq_dir, num_frames, H, W
                )
                if os.path.exists(tmp_json_path):
                    try:
                        with open(tmp_json_path, "r", encoding="utf-8") as f:
                            cached_payload = json.load(f)
                        if isinstance(cached_payload, dict) and cached_payload.get("schema") == "aerotrack_tmp_preds_v2":
                            cached_signature = cached_payload.get("signature")
                            preds = cached_payload.get("predictions", [])
                        else:
                            cached_signature = None
                            preds = cached_payload
                        # Reject empty or all-None segmentation caches.
                        signature_ok = cached_signature == cache_signature
                        is_valid = (
                            signature_ok
                            and
                            isinstance(preds, list)
                            and len(preds) > 0
                            and any(
                                any(s is not None for s in p.get("segmentations", []))
                                for p in preds
                            )
                        )
                        if is_valid:
                            all_predictions.extend(preds)
                            print(f"    => [RESUME] Loaded cached predictions: {tmp_json_path}; skipping inference.")
                            continue
                        else:
                            reason = "signature mismatch" if not signature_ok else "empty or all-None masks"
                            print(f"    => [RESUME-INVALID] Invalid cache ({reason}); deleting and re-running: {tmp_json_path}")
                            os.remove(tmp_json_path)
                    except Exception as e:
                        print(f"    => [RESUME-ERROR] Failed to load cache; re-running evaluation: {e}")
                        try:
                            os.remove(tmp_json_path)
                        except Exception:
                            pass

            t0 = time.time()
            cache_dir = None
            try:
                throughput_calc.start_timer()

                cache_dir, total_frames, fps, (w, h) = tracker.run_tracking(
                    video_path=seq_dir,
                    text_prompt=cat_name,
                    prompt_frame_idx=0,
                )

                preds = cache_to_ytvis_predictions(
                    cache_dir, vid_id, cat_id, num_frames, H, W
                )

                if tmp_json_path:
                    # Atomic write via .tmp + rename for resume safety.
                    tmp_json_tmp = tmp_json_path + ".tmp"
                    cache_payload = {
                        "schema": "aerotrack_tmp_preds_v2",
                        "signature": cache_signature,
                        "predictions": preds,
                    }
                    with open(tmp_json_tmp, "w", encoding="utf-8") as f:
                        json.dump(cache_payload, f)
                    os.replace(tmp_json_tmp, tmp_json_path)

                masks_count = sum(
                    sum(1 for s in p["segmentations"] if s is not None)
                    for p in preds
                )
                throughput_calc.stop_timer(masks_count, frames_count=total_frames)
                all_predictions.extend(preds)

                elapsed = time.time() - t0
                print(
                    f"    => {len(preds)} tracks, "
                    f"{total_frames} frames, {elapsed:.1f}s"
                )

                if vis_root is not None:
                    safe_cat = cat_name.replace(" ", "_")
                    vid_vis_dir = os.path.join(vis_root, vid_name)
                    out_path = (
                        os.path.join(vid_vis_dir, safe_cat)
                        if vis_as_jpg else
                        os.path.join(vid_vis_dir, f"{safe_cat}.mp4")
                    )
                    _render_visualization(
                        seq_dir, cache_dir, total_frames, out_path,
                        fps=fps, as_jpg=vis_as_jpg
                    )
                    if vis_as_jpg:
                        print(f"    => Visualization frames saved: {out_path}")
                    else:
                        print(f"    => Visualization video saved: {out_path}")
                else:
                    shutil.rmtree(cache_dir, ignore_errors=True)

            except Exception as e:
                throughput_calc.stop_timer(0)
                throughput_calc.failed_runs.append({
                    "video": vid_name,
                    "video_id": vid_id,
                    "category": cat_name,
                    "category_id": cat_id,
                    "error": str(e),
                })
                import traceback
                print(f"    [ERROR] Tracking failed: {e}")
                traceback.print_exc()

                if cache_dir and os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)

                if tmp_json_path and os.path.exists(tmp_json_path):
                    try:
                        os.remove(tmp_json_path)
                        print(f"    [CLEANUP] Removed invalid cache file: {tmp_json_path}")
                    except Exception:
                        pass
                if tmp_json_path and os.path.exists(tmp_json_path + ".tmp"):
                    try:
                        os.remove(tmp_json_path + ".tmp")
                    except Exception:
                        pass

                try:
                    vp = getattr(tracker, "video_predictor", None)
                    # SAM3 path: close all open sessions via handle_request
                    all_states = getattr(vp, "_ALL_INFERENCE_STATES", None)
                    if all_states:
                        for _sid in list(all_states.keys()):
                            try:
                                vp.handle_request(
                                    request=dict(type="close_session", session_id=_sid)
                                )
                            except Exception:
                                pass
                        all_states.clear()
                    # SAM2 path: reset_state clears GPU tensors held in inference_state
                    _sam2_istate = getattr(tracker, "_sam2_inference_state", None)
                    if _sam2_istate is not None and vp is not None:
                        try:
                            vp.reset_state(_sam2_istate)
                        except Exception:
                            pass
                        # Also wipe image-feature cache (largest VRAM source in SAM2)
                        _sam2_istate.get("cached_features", {}).clear()
                        _sam2_istate.get("output_dict_per_obj", {}).clear()
                        _sam2_istate.get("temp_output_dict_per_obj", {}).clear()
                except Exception:
                    pass
                if torch.cuda.is_available() and not os.environ.get("AEROTRACK_MEMORY_RESERVE"):
                    torch.cuda.empty_cache()

            try:
                all_states = getattr(
                    getattr(tracker, "video_predictor", None),
                    "_ALL_INFERENCE_STATES", {}
                )
                for _state_entry in all_states.values():
                    _istate = _state_entry.get("state", {})
                    fc = _istate.get("feature_cache")
                    if fc:
                        fc.clear()
                    for _ts in _istate.get("tracker_inference_states", []):
                        _od = _ts.get("output_dict", {})
                        _od.get("non_cond_frame_outputs", {}).clear()
                        for _pobj in _ts.get("output_dict_per_obj", {}).values():
                            _pobj.get("non_cond_frame_outputs", {}).clear()
            except Exception:
                pass

            gc.collect()
            if torch.cuda.is_available() and not os.environ.get("AEROTRACK_MEMORY_RESERVE"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

    if throughput_calc.failed_runs:
        print("\n  [WARN] Some inference runs failed; metrics use successful runs only:")
        for item in throughput_calc.failed_runs:
            print(
                f"    - Video={item['video']}  Category={item['category']} "
                f"(id={item['category_id']}): {item['error']}"
            )

    return all_predictions, throughput_calc
