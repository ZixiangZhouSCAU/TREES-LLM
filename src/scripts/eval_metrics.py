"""
评估指标模块
"""

import numpy as np
from typing import Dict, List, Optional


def compute_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """计算IoU"""
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / (union + 1e-8))


def compute_miou(preds: List[np.ndarray], targets: List[np.ndarray]) -> float:
    """计算平均IoU"""
    ious = [compute_iou(p, t) for p, t in zip(preds, targets)]
    return float(np.mean(ious))


def compute_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """计算RMSE"""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """计算R²"""
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


def compute_f1(pred: np.ndarray, target: np.ndarray) -> float:
    """计算F1-score"""
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def evaluate_tree_params(
    predictions: Dict[str, Dict[str, float]],
    ground_truth: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    评估树木参数提取精度

    Args:
        predictions: {tree_id: {"height": float, "dbh": float, ...}}
        ground_truth: {tree_id: {"height": float, "dbh": float, ...}}

    Returns:
        {"height": {"rmse": float, "r2": float}, "dbh": {...}, ...}
    """
    results = {}
    common_ids = set(predictions.keys()) & set(ground_truth.keys())

    if not common_ids:
        print("Warning: No common tree IDs between predictions and ground truth")
        return results

    params = ["height", "dbh", "crown_width", "stem_volume"]

    for param in params:
        pred_vals = []
        gt_vals = []

        for tid in common_ids:
            if param in predictions[tid] and param in ground_truth[tid]:
                pred_vals.append(predictions[tid][param])
                gt_vals.append(ground_truth[tid][param])

        if len(pred_vals) > 0:
            pred_arr = np.array(pred_vals)
            gt_arr = np.array(gt_vals)

            results[param] = {
                "rmse": compute_rmse(pred_arr, gt_arr),
                "r2": compute_r2(pred_arr, gt_arr),
                "mae": float(np.mean(np.abs(pred_arr - gt_arr))),
                "bias": float(np.mean(pred_arr - gt_arr)),
                "n_samples": len(pred_vals),
            }

    return results
