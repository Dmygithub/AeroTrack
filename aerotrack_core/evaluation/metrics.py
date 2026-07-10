"""Metric computation for YTVIS AP, HOTA, Frame-AP, mask IoU, ODR, and per-category stats."""

import copy
import numpy as np
import pycocotools.mask as mask_util

from .odr_metric import ODRCalculator


def _ensure_rle_bytes(seg):
    """Normalize RLE counts to bytes for pycocotools."""
    if not isinstance(seg, dict):
        return seg
    counts = seg.get("counts")
    if isinstance(counts, bytes):
        return seg
    if isinstance(counts, str):
        return {"counts": counts.encode("ascii"), "size": seg["size"]}
    if isinstance(counts, list):
        size = seg.get("size", [])
        if len(size) >= 2:
            compressed = mask_util.frPyObjects(seg, size[0], size[1])
            return compressed
    return seg


def _fix_segmentations_rle(ann_list: list) -> None:
    """In-place fix RLE counts fields in annotation segmentations."""
    for ann in ann_list:
        segs = ann.get("segmentations", [])
        ann["segmentations"] = [
            _ensure_rle_bytes(s) if s is not None else None
            for s in segs
        ]


def _mean_precision_at(precision: np.ndarray, thr_idx: int, cat_idx=None):
    """Return COCO/YTVIS mean precision at one IoU-threshold index."""
    try:
        vals = precision[thr_idx, :, :, 0, -1] if cat_idx is None else precision[thr_idx, :, cat_idx, 0, -1]
        vals = vals[vals > -1]
        return float(np.mean(vals)) if len(vals) else None
    except Exception:
        return None


def _mean_precision_all_thresholds(precision: np.ndarray, cat_idx=None):
    """Return COCO/YTVIS mean precision over all IoU thresholds."""
    try:
        vals = precision[:, :, :, 0, -1] if cat_idx is None else precision[:, :, cat_idx, 0, -1]
        vals = vals[vals > -1]
        return float(np.mean(vals)) if len(vals) else None
    except Exception:
        return None


def _precision_at_iou(precision: np.ndarray, iou_thrs, target: float, cat_idx=None):
    """Return AP at a named IoU threshold if the evaluator contains it."""
    matches = np.where(np.isclose(np.asarray(iou_thrs, dtype=float), float(target)))[0]
    if len(matches) == 0:
        return None
    return _mean_precision_at(precision, int(matches[0]), cat_idx)


def run_ytvis_map_evaluation(gt_json_data: dict, predictions: list, iou_type: str = "segm") -> dict:
    """Compute YTVIS/COCO-style segmentation mAP."""
    import sys
    from unittest.mock import MagicMock

    if 'torch.nn.attention' not in sys.modules:
        sys.modules['torch.nn.attention'] = MagicMock()

    from aerotrack_core.models.sam3.eval.ytvis_coco_wrapper import YTVIS
    from aerotrack_core.models.sam3.eval.ytvis_eval import YTVISeval

    gt_data_fixed = copy.deepcopy(gt_json_data)
    _fix_segmentations_rle(gt_data_fixed.get("annotations", []))

    gt = YTVIS(ignore_gt_cats=False)
    gt.dataset = gt_data_fixed
    gt.createIndex()

    if not predictions:
        print("  [WARN] Empty predictions; skipping mAP computation")
        return {
            "AP": 0.0,
            "AP25": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "per_category": [
                {"category_id": c["id"], "AP": 0.0, "AP25": 0.0, "AP50": 0.0, "AP75": 0.0}
                for c in gt_json_data.get("categories", [])
            ],
        }

    predictions_for_eval = []
    for ann in predictions:
        ann2 = dict(ann)
        if "video_id" in ann2 and "image_id" not in ann2:
            ann2["image_id"] = int(ann2["video_id"])
        ann2["segmentations"] = [
            _ensure_rle_bytes(s) if s is not None else None
            for s in ann2.get("segmentations", [])
        ]
        predictions_for_eval.append(ann2)

    dt = gt.loadRes(predictions_for_eval)
    ytvis_eval = YTVISeval(gt, dt, iouType=iou_type)
    ytvis_eval.evaluate()
    ytvis_eval.accumulate()
    ytvis_eval.summarize()

    metric_names = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    ]
    results = {}
    for i, name in enumerate(metric_names):
        if i < len(ytvis_eval.stats):
            val = float(ytvis_eval.stats[i])
            results[name] = val if val >= 0 else None

    if hasattr(ytvis_eval, "eval") and "precision" in ytvis_eval.eval:
        p = ytvis_eval.eval["precision"]
        results["AP25"] = None
        results["AP50"] = _precision_at_iou(p, ytvis_eval.params.iouThrs, 0.50)
        results["AP75"] = _precision_at_iou(p, ytvis_eval.params.iouThrs, 0.75)
    results["iou_thrs"] = list(ytvis_eval.params.iouThrs)

    # Standard YTVIS/COCO AP uses IoU=.50:.95 and therefore has no AP25.
    # Run a separate one-threshold pass so AP remains standard while AP25 is available.
    ytvis_eval25 = YTVISeval(gt, dt, iouType=iou_type)
    ytvis_eval25.params.iouThrs = np.array([0.25])
    ytvis_eval25.evaluate()
    ytvis_eval25.accumulate()
    if hasattr(ytvis_eval25, "eval") and "precision" in ytvis_eval25.eval:
        p25 = ytvis_eval25.eval["precision"]
        results["AP25"] = _precision_at_iou(p25, ytvis_eval25.params.iouThrs, 0.25)

    # Extract per-category video AP / AP25 / AP50 / AP75
    per_category = []
    if hasattr(ytvis_eval, 'eval') and 'precision' in ytvis_eval.eval:
        p = ytvis_eval.eval['precision']
        for i, cat_id in enumerate(ytvis_eval.params.catIds):
            ap25 = None
            if hasattr(ytvis_eval25, "eval") and "precision" in ytvis_eval25.eval:
                ap25 = _precision_at_iou(ytvis_eval25.eval["precision"], ytvis_eval25.params.iouThrs, 0.25, i)
            per_category.append({
                "category_id": cat_id,
                "AP": _mean_precision_all_thresholds(p, i),
                "AP25": ap25,
                "AP50": _precision_at_iou(p, ytvis_eval.params.iouThrs, 0.50, i),
                "AP75": _precision_at_iou(p, ytvis_eval.params.iouThrs, 0.75, i),
            })
    results["per_category"] = per_category

    return results



def run_hota_evaluation(
    gt_json_data: dict,
    predictions: list,
    hota_dataset_name: str,
    iou_type: str = "segm",
) -> dict:
    """Compute HOTA and related tracking metrics via TrackEval."""
    try:
        import sys
        from unittest.mock import MagicMock

        if 'torch.nn.attention' not in sys.modules:
            sys.modules['torch.nn.attention'] = MagicMock()

        from aerotrack_core.models.sam3.eval.hota_eval_toolkit.trackeval import Evaluator
        from aerotrack_core.models.sam3.eval.hota_eval_toolkit.trackeval.datasets.youtube_vis import YouTubeVIS
        from aerotrack_core.models.sam3.eval.hota_eval_toolkit.trackeval.metrics.hota import HOTA
    except ImportError as e:
        print(f"  [WARN] HOTA eval toolkit unavailable: {e}")
        return {}
    has_identity = False

    if not predictions:
        print("  [WARN] Empty predictions; skipping HOTA computation")
        return {}

    gt_data = copy.deepcopy(gt_json_data)

    video_hw = {v["id"]: (v["height"], v["width"]) for v in gt_data["videos"]}
    for ann in gt_data["annotations"]:
        vid_id = ann.get("video_id")
        if vid_id in video_hw:
            ann["height"], ann["width"] = video_hw[vid_id]

    dataset_config = {
        "GT_JSON_OBJECT":      gt_data,
        "TRACKER_JSON_OBJECT": predictions,
        "IOU_TYPE":            iou_type,
        "DATASET_NAME":        hota_dataset_name,
        "PRINT_CONFIG":        False,
    }
    eval_config = {
        "USE_PARALLEL":          False,
        "PRINT_RESULTS":         True,
        "PRINT_ONLY_COMBINED":   True,
        "PRINT_CONFIG":          False,
        "TIME_PROGRESS":         False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY":        False,
        "OUTPUT_DETAILED":       False,
        "OUTPUT_EMPTY_CLASSES":  True,
        "PLOT_CURVES":           False,
    }

    try:
        dataset   = YouTubeVIS(dataset_config)
        evaluator = Evaluator(eval_config)
        metric_list = [HOTA()]
        if has_identity:
            metric_list.append(IDENTITY())

        output_res, _ = evaluator.evaluate([dataset], metric_list, show_progressbar=False)

        results = {}
        if hota_dataset_name in output_res:
            tracker_res = list(output_res[hota_dataset_name].values())
            if tracker_res and tracker_res[0] is not None:
                combined = tracker_res[0].get("COMBINED_SEQ", {})

                hota_det_av = combined.get("cls_comb_det_av", {})
                hota_data = hota_det_av.get("HOTA", {})
                for field in ["HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA"]:
                    val = hota_data.get(field)
                    if val is None:
                        val = hota_det_av.get(field)
                    if val is not None:
                        results[field] = float(np.mean(val)) if isinstance(val, np.ndarray) else float(val)

                hota_cls_av = combined.get("cls_comb_cls_av", {})
                hota_data_cls = hota_cls_av.get("HOTA", {})
                for field in ["HOTA", "DetA", "AssA"]:
                    val = hota_data_cls.get(field)
                    if val is None:
                        val = hota_cls_av.get(field)
                    if val is not None:
                        results[f"{field}_cls_av"] = float(np.mean(val)) if isinstance(val, np.ndarray) else float(val)

                if has_identity:
                    id_data = hota_det_av.get("Identity", {})
                    for field in ["IDF1", "IDR", "IDP"]:
                        val = id_data.get(field)
                        if val is None:
                            val = hota_det_av.get(field)
                        if val is not None:
                            results[field] = float(np.mean(val)) if isinstance(val, np.ndarray) else float(val)

                cats_name_to_id = {c["name"]: c["id"] for c in gt_json_data["categories"]}
                per_category = []
                for cat_name, cat_res in tracker_res[0].items():
                    if cat_name in cats_name_to_id:
                        cat_id = cats_name_to_id[cat_name]
                        hota_data = cat_res.get("COMBINED_SEQ", cat_res)
                        hota_subdict = hota_data.get("HOTA", {})

                        def _extract(key):
                            """Extract scalar from HOTA sub-dict; handles ndarray (alpha-averaged)."""
                            v = hota_subdict.get(key) if isinstance(hota_subdict, dict) else None
                            if v is None:
                                v = hota_data.get(key)
                            if v is None:
                                return None
                            return float(np.mean(v)) if isinstance(v, np.ndarray) else float(v)

                        entry = {"category_id": cat_id}
                        for field in ("HOTA", "DetA", "AssA"):
                            entry[field] = _extract(field)
                        per_category.append(entry)
                results["per_category"] = per_category

        return results

    except Exception as e:
        import traceback
        print(f"  [WARN] HOTA evaluation failed: {e}")
        traceback.print_exc()

    return {}


def _uav_iou_threshold(gt_ann: dict) -> float:
    """Return UAV scale-adaptive IoU threshold for one GT track."""
    bboxes = [b for b in gt_ann.get("bboxes", []) if b and b != [0, 0, 0, 0]]
    if not bboxes:
        return 0.5
    areas = [b[2] * b[3] for b in bboxes]
    median_area = float(np.median(areas))
    if median_area < 32 ** 2:
        return 0.25
    if median_area < 96 ** 2:
        return 0.35
    return 0.5


def compute_mask_j_score(
    gt_json_data: dict,
    predictions: list,
    iou_threshold: float = 0.5,
    uav_adaptive: bool = True,
) -> dict:
    """Compute mask IoU (J), segmentation precision/recall/F1 with track matching."""
    from scipy.optimize import linear_sum_assignment

    def _ensure_bytes(seg):
        """Ensure RLE counts are bytes for pycocotools."""
        if isinstance(seg.get("counts"), str):
            return {"counts": seg["counts"].encode(), "size": seg["size"]}
        return seg

    def _track_mean_iou(gt_segs: list, pred_segs: list) -> float:
        """Return mean mask IoU on frames where GT has a valid segmentation."""
        n = min(len(gt_segs), len(pred_segs))
        frame_ious = []
        for f in range(n):
            gs = gt_segs[f]
            if not gs:
                continue
            ps = pred_segs[f]
            if not ps:
                frame_ious.append(0.0)
                continue
            iou_val = float(mask_util.iou([_ensure_bytes(ps)], [_ensure_bytes(gs)], [0])[0][0])
            frame_ious.append(iou_val)
        for f in range(n, len(gt_segs)):
            if gt_segs[f]:
                frame_ious.append(0.0)
        return float(np.mean(frame_ious)) if frame_ious else 0.0

    gt_groups: dict = {}
    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        key = (ann["video_id"], ann["category_id"])
        gt_groups.setdefault(key, []).append(ann)

    pred_groups: dict = {}
    for pred in predictions:
        key = (pred["video_id"], pred["category_id"])
        pred_groups.setdefault(key, []).append(pred)

    all_keys      = set(gt_groups.keys()) | set(pred_groups.keys())
    matched_ious  = []
    cat_j         = {}   # {cat_id: [iou, iou...]}
    total_gt      = 0
    total_pred    = 0
    tp_at_thr     = 0

    for key in all_keys:
        gts   = gt_groups.get(key, [])
        preds = pred_groups.get(key, [])
        total_gt   += len(gts)
        total_pred += len(preds)

        cat_id = key[1]
        if cat_id not in cat_j:
            cat_j[cat_id] = []

        if not gts or not preds:
            continue

        n_gt, n_pred = len(gts), len(preds)
        iou_matrix = np.zeros((n_gt, n_pred), dtype=np.float32)
        for i, gt_ann in enumerate(gts):
            for j, pred in enumerate(preds):
                iou_matrix[i, j] = _track_mean_iou(
                    gt_ann.get("segmentations", []),
                    pred.get("segmentations", []),
                )

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        for r, c in zip(row_ind, col_ind):
            iou_val = float(iou_matrix[r, c])
            matched_ious.append(iou_val)
            cat_j[cat_id].append(iou_val)
            thr = _uav_iou_threshold(gts[r]) if uav_adaptive else iou_threshold
            if iou_val >= thr:
                tp_at_thr += 1

    J             = float(np.mean(matched_ious)) if matched_ious else None
    seg_recall    = tp_at_thr / max(1, total_gt)
    seg_precision = tp_at_thr / max(1, total_pred)
    seg_F1 = (
        2 * seg_precision * seg_recall / (seg_precision + seg_recall)
        if (seg_precision + seg_recall) > 0 else 0.0
    )
    
    per_category = []
    for cat_id, ious in cat_j.items():
        per_category.append({
            "category_id": cat_id,
            "J": float(np.mean(ious)) if ious else None
        })

    return {
        "J":                 round(J, 4) if J is not None else None,
        "seg_recall":        round(seg_recall, 4),
        "seg_precision":     round(seg_precision, 4),
        "seg_F1":            round(seg_F1, 4),
        "total_gt_tracks":   total_gt,
        "total_pred_tracks": total_pred,
        "matched_tracks":    len(matched_ious),
        "per_category":      per_category,
    }


def compute_odr(gt_json_data: dict, predictions: list) -> ODRCalculator:
    """Build an ODRCalculator from prediction bboxes and GT annotations."""
    odr_calc = ODRCalculator()

    gt_frames = {}
    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        vid = ann["video_id"]
        cid = ann["category_id"]
        for f_idx, gb in enumerate(ann.get("bboxes", [])):
            if gb and gb != [0, 0, 0, 0]:
                box = [gb[0], gb[1], gb[0] + gb[2], gb[1] + gb[3]]
                key = (vid, f_idx, cid)
                gt_frames.setdefault(key, []).append(box)

    pred_frames = {}
    for pred in predictions:
        vid = pred["video_id"]
        cid = pred["category_id"]
        for f_idx, pb in enumerate(pred.get("bboxes", [])):
            if pb and pb != [0, 0, 0, 0]:
                box = [pb[0], pb[1], pb[0] + pb[2], pb[1] + pb[3]]
                key = (vid, f_idx, cid)
                pred_frames.setdefault(key, []).append(box)

    all_keys = set(gt_frames.keys()) | set(pred_frames.keys())
    for key in all_keys:
        vid, f_idx, cid = key
        gt_boxes = gt_frames.get(key, [])
        pred_boxes = pred_frames.get(key, [])
        odr_calc.add_frame_results(
            pred_boxes, gt_boxes,
            [cid] * len(pred_boxes), [cid] * len(gt_boxes)
        )

    return odr_calc


def compute_per_category_stats(gt_json_data: dict, predictions: list) -> list:
    """Return GT and predicted track counts per category."""
    cats = {c["id"]: c["name"] for c in gt_json_data["categories"]}

    gt_counts: dict   = {}
    pred_counts: dict = {}

    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 0:
            cid = ann["category_id"]
            gt_counts[cid] = gt_counts.get(cid, 0) + 1

    for pred in predictions:
        cid = pred["category_id"]
        pred_counts[cid] = pred_counts.get(cid, 0) + 1

    return [
        {
            "category_id": cid,
            "name":        cats[cid],
            "gt_tracks":   gt_counts.get(cid, 0),
            "pred_tracks": pred_counts.get(cid, 0),
        }
        for cid in sorted(cats.keys())
    ]


def compute_frame_segm_ap(gt_json_data: dict, predictions: list, uav_adaptive: bool = True) -> dict:
    """Compute frame-level COCO segmentation AP (track-ID agnostic)."""
    import io
    import contextlib
    import copy

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("  [WARN] pycocotools unavailable; skipping Frame-AP computation")
        return {}

    video_info = {v["id"]: v for v in gt_json_data["videos"]}
    image_id_map: dict = {}   # (vid_id, frame_idx) -> image_id
    coco_images = []
    coco_ann_list = []
    img_id = 0
    ann_id = 0

    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        vid_id  = ann["video_id"]
        cat_id  = ann["category_id"]
        segs    = ann.get("segmentations", [])
        bboxes  = ann.get("bboxes", [])   # XYWH
        v_info  = video_info.get(vid_id, {})

        for f_idx, (seg, bbox) in enumerate(zip(segs, bboxes)):
            if not seg or not bbox:
                continue
            key = (vid_id, f_idx)
            if key not in image_id_map:
                img_id += 1
                image_id_map[key] = img_id
                coco_images.append({
                    "id":     img_id,
                    "height": v_info.get("height", 0),
                    "width":  v_info.get("width", 0),
                })
            ann_id += 1
            coco_ann_list.append({
                "id":           ann_id,
                "image_id":     image_id_map[key],
                "category_id":  cat_id,
                "segmentation": seg if isinstance(seg, dict) else seg,
                "bbox":         bbox,         # XYWH
                "area":         max(bbox[2] * bbox[3], 1.0),
                "iscrowd":      0,
            })

    if not coco_images:
        print("  [WARN] Frame-AP: no valid GT frames found; skipping computation")
        return {}

    coco_gt_dict = {
        "images":     coco_images,
        "annotations": coco_ann_list,
        "categories": copy.deepcopy(gt_json_data["categories"]),
    }

    coco_dt_list = []
    for pred in predictions:
        vid_id  = pred["video_id"]
        cat_id  = pred["category_id"]
        segs    = pred.get("segmentations", [])
        bboxes  = pred.get("bboxes", [])    # XYWH
        score   = float(pred.get("score", 1.0))

        for f_idx, (seg, bbox) in enumerate(zip(segs, bboxes)):
            if not seg or not bbox:
                continue
            key = (vid_id, f_idx)
            if key not in image_id_map:
                continue
            coco_dt_list.append({
                "image_id":     image_id_map[key],
                "category_id":  cat_id,
                "segmentation": seg if isinstance(seg, dict) else seg,
                "bbox":         bbox,
                "score":        score,
                "area":         max(bbox[2] * bbox[3], 1.0),
            })

    if not coco_dt_list:
        print("  [WARN] Frame-AP: no valid frame-level predictions; returning zeros")
        frame_iou_thrs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50] if uav_adaptive else None
        return {
            "AP": 0.0,
            "F-AP@0.25": 0.0,
            "F-AP@0.50": 0.0,
            "F_AP25": 0.0,
            "F_AP50": 0.0,
            "iou_thrs": frame_iou_thrs,
            "per_category": [
                {
                    "category_id": c["id"],
                    "AP": 0.0,
                    "F-AP@0.25": 0.0,
                    "F-AP@0.50": 0.0,
                    "F_AP25": 0.0,
                    "F_AP50": 0.0,
                }
                for c in gt_json_data.get("categories", [])
            ],
        }

    coco = COCO()
    coco.dataset = coco_gt_dict
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco.createIndex()

    coco_dt = coco.loadRes(coco_dt_list)
    coco_eval = COCOeval(coco, coco_dt, iouType="segm")

    if uav_adaptive:
        coco_eval.params.iouThrs = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50])

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    metric_names = [
        "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
        "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
    ]
    results = {}
    for i, name in enumerate(metric_names):
        if i < len(coco_eval.stats):
            val = float(coco_eval.stats[i])
            results[name] = val if val >= 0 else None

    if hasattr(coco_eval, "eval") and "precision" in coco_eval.eval:
        p = coco_eval.eval["precision"]
        results["F-AP@0.25"] = _precision_at_iou(p, coco_eval.params.iouThrs, 0.25)
        results["F-AP@0.50"] = _precision_at_iou(p, coco_eval.params.iouThrs, 0.50)
        results["F_AP25"] = results["F-AP@0.25"]
        results["F_AP50"] = results["F-AP@0.50"]

    results["iou_thrs"] = list(coco_eval.params.iouThrs)

    # Extract per-category Frame-AP values.
    per_category = []
    if hasattr(coco_eval, 'eval') and 'precision' in coco_eval.eval:
        p = coco_eval.eval['precision']
        for i, cat_id in enumerate(coco_eval.params.catIds):
            try:
                p_cat = p[:, :, i, 0, -1]
                p_cat = p_cat[p_cat > -1]
                ap = np.mean(p_cat) if len(p_cat) else None
                
                fap25 = _precision_at_iou(p, coco_eval.params.iouThrs, 0.25, i)
                fap50 = _precision_at_iou(p, coco_eval.params.iouThrs, 0.50, i)
                per_category.append({
                    "category_id": cat_id,
                    "AP": float(ap) if ap is not None else None,
                    "F-AP@0.25": fap25,
                    "F-AP@0.50": fap50,
                    "F_AP25": fap25,
                    "F_AP50": fap50,
                })
            except Exception:
                per_category.append({
                    "category_id": cat_id,
                    "AP": None,
                    "F-AP@0.25": None,
                    "F-AP@0.50": None,
                    "F_AP25": None,
                    "F_AP50": None,
                })
    results["per_category"] = per_category

    return results

def compute_spatiotemporal_f1(gt_json_data: dict, predictions: list) -> dict:
    """Compute spatio-temporal pixel-level F1 without track IDs."""

    def _ensure_bytes(seg: dict) -> dict:
        if isinstance(seg.get("counts"), str):
            return {"counts": seg["counts"].encode(), "size": seg["size"]}
        return seg

    def _decode_rle(seg):
        try:
            return mask_util.decode(_ensure_bytes(seg))   # H×W uint8
        except Exception:
            return None

    gt_frame_segs: dict = {}
    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        vid = ann["video_id"]
        cat = ann["category_id"]
        for f_idx, seg in enumerate(ann.get("segmentations", [])):
            if seg:
                gt_frame_segs.setdefault((vid, f_idx, cat), []).append(seg)

    pred_frame_segs: dict = {}
    for pred in predictions:
        vid = pred["video_id"]
        cat = pred["category_id"]
        for f_idx, seg in enumerate(pred.get("segmentations", [])):
            if seg:
                pred_frame_segs.setdefault((vid, f_idx, cat), []).append(seg)

    cat_accum: dict = {}
    all_keys = set(gt_frame_segs.keys()) | set(pred_frame_segs.keys())

    for key in all_keys:
        vid, f_idx, cat = key
        if cat not in cat_accum:
            cat_accum[cat] = {"inter": 0, "pred": 0, "gt": 0}
        acc = cat_accum[cat]

        gt_mask = None
        for seg in gt_frame_segs.get(key, []):
            m = _decode_rle(seg)
            if m is not None:
                gt_mask = m if gt_mask is None else np.bitwise_or(gt_mask, m)

        pred_mask = None
        for seg in pred_frame_segs.get(key, []):
            m = _decode_rle(seg)
            if m is not None:
                pred_mask = m if pred_mask is None else np.bitwise_or(pred_mask, m)

        gt_px   = int(gt_mask.sum())   if gt_mask   is not None else 0
        pred_px = int(pred_mask.sum()) if pred_mask is not None else 0

        if gt_mask is not None and pred_mask is not None:
            h = min(gt_mask.shape[0], pred_mask.shape[0])
            w = min(gt_mask.shape[1], pred_mask.shape[1])
            inter_px = int(np.bitwise_and(gt_mask[:h, :w], pred_mask[:h, :w]).sum())
        else:
            inter_px = 0

        acc["inter"] += inter_px
        acc["pred"]  += pred_px
        acc["gt"]    += gt_px

    cat_map = {c["id"]: c["name"] for c in gt_json_data["categories"]}

    per_category = []
    total_inter = total_pred = total_gt = 0

    for cat_id in sorted(cat_accum.keys()):
        acc = cat_accum[cat_id]
        inter, pred_px, gt_px = acc["inter"], acc["pred"], acc["gt"]
        total_inter += inter
        total_pred  += pred_px
        total_gt    += gt_px

        st_p = inter / pred_px if pred_px > 0 else 0.0
        st_r = inter / gt_px   if gt_px   > 0 else 0.0
        st_f = (2 * st_p * st_r / (st_p + st_r)) if (st_p + st_r) > 0 else 0.0

        per_category.append({
            "category_id":  cat_id,
            "name":         cat_map.get(cat_id, str(cat_id)),
            "ST_precision": round(st_p, 4),
            "ST_recall":    round(st_r, 4),
            "ST_F1":        round(st_f, 4),
            "inter_px":     inter,
            "pred_px":      pred_px,
            "gt_px":        gt_px,
        })

    global_p = total_inter / total_pred if total_pred > 0 else 0.0
    global_r = total_inter / total_gt   if total_gt   > 0 else 0.0
    global_f = (2 * global_p * global_r / (global_p + global_r)) if (global_p + global_r) > 0 else 0.0

    return {
        "ST_precision":   round(global_p, 4),
        "ST_recall":      round(global_r, 4),
        "ST_F1":          round(global_f, 4),
        "total_inter_px": total_inter,
        "total_pred_px":  total_pred,
        "total_gt_px":    total_gt,
        "per_category":   per_category,
    }


def _mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Convert a binary mask to its boundary mask via morphological dilation."""
    h, w = mask.shape
    diag = np.sqrt(h ** 2 + w ** 2)
    radius = max(1, int(round(dilation_ratio * diag)))

    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(mask.astype(np.uint8), kernel)
    boundary = dilated - mask.astype(np.uint8)
    return boundary.astype(bool)


def _boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """Compute boundary F-measure for one frame."""
    pred_b = _mask_to_boundary(pred_mask, dilation_ratio)
    gt_b   = _mask_to_boundary(gt_mask,   dilation_ratio)

    pred_b_sum = pred_b.sum()
    gt_b_sum   = gt_b.sum()
    if pred_b_sum == 0 and gt_b_sum == 0:
        return 1.0
    if pred_b_sum == 0 or gt_b_sum == 0:
        return 0.0

    precision = np.logical_and(pred_b, gt_b).sum() / pred_b_sum
    recall    = np.logical_and(pred_b, gt_b).sum() / gt_b_sum
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_jf_score(
    gt_json_data: dict,
    predictions: list,
    iou_threshold: float = 0.5,
    dilation_ratio: float = 0.02,
) -> dict:
    """Compute DAVIS-style J&F (mask IoU + boundary F) with Hungarian matching."""
    import cv2
    from scipy.optimize import linear_sum_assignment

    def _ensure_bytes(seg):
        if isinstance(seg.get("counts"), str):
            return {"counts": seg["counts"].encode(), "size": seg["size"]}
        return seg

    def _decode(seg):
        try:
            return mask_util.decode(_ensure_bytes(seg)).astype(bool)
        except Exception:
            return None

    def _track_jf(gt_segs, pred_segs):
        n = min(len(gt_segs), len(pred_segs))
        j_vals, f_vals = [], []
        for f in range(n):
            gs, ps = gt_segs[f], pred_segs[f]
            if not gs or not ps:
                continue
            gm = _decode(gs)
            pm = _decode(ps)
            if gm is None or pm is None:
                continue
            # resize pred to gt shape if needed
            if gm.shape != pm.shape:
                pm = cv2.resize(pm.astype(np.uint8), (gm.shape[1], gm.shape[0]),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
            inter = np.logical_and(gm, pm).sum()
            union = np.logical_or(gm, pm).sum()
            j_vals.append(float(inter / union) if union > 0 else 1.0)
            f_vals.append(_boundary_f1(pm, gm, dilation_ratio))
        j = float(np.mean(j_vals)) if j_vals else 0.0
        f = float(np.mean(f_vals)) if f_vals else 0.0
        return j, f

    gt_groups: dict = {}
    for ann in gt_json_data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        key = (ann["video_id"], ann["category_id"])
        gt_groups.setdefault(key, []).append(ann)

    pred_groups: dict = {}
    for pred in predictions:
        key = (pred["video_id"], pred["category_id"])
        pred_groups.setdefault(key, []).append(pred)

    all_keys = set(gt_groups.keys()) | set(pred_groups.keys())
    cat_j: dict = {}
    cat_f: dict = {}

    for key in all_keys:
        gts   = gt_groups.get(key, [])
        preds = pred_groups.get(key, [])
        cat_id = key[1]
        cat_j.setdefault(cat_id, [])
        cat_f.setdefault(cat_id, [])

        if not gts or not preds:
            continue

        n_gt, n_pred = len(gts), len(preds)
        j_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
        f_mat = np.zeros((n_gt, n_pred), dtype=np.float32)
        for i, gt_ann in enumerate(gts):
            for j, pred in enumerate(preds):
                jv, fv = _track_jf(gt_ann.get("segmentations", []),
                                   pred.get("segmentations", []))
                j_mat[i, j] = jv
                f_mat[i, j] = fv

        # Hungarian on J matrix (primary quality signal)
        row_ind, col_ind = linear_sum_assignment(-j_mat)
        for r, c in zip(row_ind, col_ind):
            cat_j[cat_id].append(float(j_mat[r, c]))
            cat_f[cat_id].append(float(f_mat[r, c]))

    all_j = [v for vals in cat_j.values() for v in vals]
    all_f = [v for vals in cat_f.values() for v in vals]
    J = float(np.mean(all_j)) if all_j else 0.0
    F = float(np.mean(all_f)) if all_f else 0.0

    per_category = []
    for cat_id in sorted(set(cat_j.keys()) | set(cat_f.keys())):
        jv = float(np.mean(cat_j[cat_id])) if cat_j.get(cat_id) else 0.0
        fv = float(np.mean(cat_f[cat_id])) if cat_f.get(cat_id) else 0.0
        per_category.append({
            "category_id": cat_id,
            "J":       round(jv, 4),
            "F":       round(fv, 4),
            "JF_mean": round((jv + fv) / 2, 4),
        })

    return {
        "J":           round(J, 4),
        "F":           round(F, 4),
        "JF_mean":     round((J + F) / 2, 4),
        "per_category": per_category,
    }
