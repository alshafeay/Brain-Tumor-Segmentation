import torch
import numpy as np
from tqdm import tqdm

from metrics import dice_score, iou_score, precision_score, recall_score, hausdorff_distance_95


def evaluate_model(model, loader, device, threshold=0.5):
    model.eval()

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_hd95 = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            batch_size = images.shape[0]
            for i in range(batch_size):
                logit = outputs[i:i+1]
                target = masks[i:i+1]

                all_dice.append(dice_score(logit, target, threshold))
                all_iou.append(iou_score(logit, target, threshold))
                all_precision.append(precision_score(logit, target, threshold))
                all_recall.append(recall_score(logit, target, threshold))

                prob = torch.sigmoid(logit)
                pred_mask = (prob > threshold).cpu().numpy().squeeze()
                gt_mask = target.cpu().numpy().squeeze()

                hd95 = hausdorff_distance_95(pred_mask, gt_mask)
                if not np.isnan(hd95):
                    all_hd95.append(hd95)

    results = {
        "Dice": np.mean(all_dice),
        "IoU": np.mean(all_iou),
        "Precision": np.mean(all_precision),
        "Recall": np.mean(all_recall),
        "HD95": np.mean(all_hd95) if len(all_hd95) > 0 else float("nan"),
    }

    return results
