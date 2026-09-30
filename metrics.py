import torch
import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

    return dice.item()


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)

    return iou.item()


def precision_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positive = (preds * targets).sum()
    predicted_positive = preds.sum()

    precision = (true_positive + smooth) / (predicted_positive + smooth)

    return precision.item()


def recall_score(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    true_positive = (preds * targets).sum()
    actual_positive = targets.sum()

    recall = (true_positive + smooth) / (actual_positive + smooth)

    return recall.item()


def hausdorff_distance_95(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if not pred_mask.any() or not gt_mask.any():
        return np.nan

    dt_gt = distance_transform_edt(~gt_mask)
    dt_pred = distance_transform_edt(~pred_mask)

    distances_pred_to_gt = dt_gt[pred_mask]
    distances_gt_to_pred = dt_pred[gt_mask]

    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])

    hd95 = np.percentile(all_distances, 95)

    return hd95
