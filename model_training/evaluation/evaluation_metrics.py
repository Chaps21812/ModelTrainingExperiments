from torch.types import Tensor
import numpy as np
import torch

def _find_centroid(tensor: Tensor) -> Tensor:
    """
    Find the centroid of the bounding box given X1, Y1, X2, Y2.Tensor must be in shape [N,4].

    Args:
        tensor (tensor): tensor bounding box [X1,Y1,X2,Y2], shape [N,4]

    Returns:
        centroids (tensor): Torch tensor of centroids
    """
    x = (tensor[:,2]+tensor[:,0])/2
    y = (tensor[:,3]+tensor[:,1])/2
    centroids = torch.stack((x,y), dim=1)
    return centroids

def _calculate_true_positives(centroids:Tensor, targets:Tensor) -> tuple[Tensor, Tensor]:
    """
    Find the centroid of the bounding box given X1, Y1, X2, Y2.Tensor must be in shape [N,4].

    Args:
        tensor (tensor): tensor bounding box [X1,Y1,X2,Y2], shape [N,4]

    Returns:
        centroids (tensor): List of files present in the train test split
    """
    predicted_x = centroids[:,0].unsqueeze(1)
    predicted_y = centroids[:,1].unsqueeze(1)

    target_x1 = targets[:, 0].unsqueeze(0)
    target_y1 = targets[:, 1].unsqueeze(0)
    target_x2 = targets[:, 2].unsqueeze(0)
    target_y2 = targets[:, 3].unsqueeze(0)

    inside_x = (predicted_x >= target_x1) & (predicted_x <= target_x2)
    inside_y = (predicted_y >= target_y1) & (predicted_y <= target_y2)
    truth_table = (inside_x & inside_y)
    target_matched = truth_table.any(dim=0)
    prediction_matched = truth_table.any(dim=1)

    true_positives = target_matched.sum()
    false_negatives = (~prediction_matched).sum()
    false_positives = (~target_matched).sum()

    return true_positives.item(), false_positives.item(), false_negatives.item()

def _calculate_IOU(predictions: Tensor, targets: Tensor):
    x1_prediction, y1_prediction, x2_prediction, y2_prediction = predictions[:, 0], predictions[:, 1], predictions[:, 2], predictions[:, 3]
    x1_target, y1_target, x2_target, y2_target = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]


    intersect_x1 = torch.maximum(x1_prediction[:,None], x1_target)
    intersect_y1 = torch.maximum(y1_prediction[:,None], y1_target)
    intersect_x2 = torch.minimum(x2_prediction[:,None], x2_target)
    intersect_y2 = torch.minimum(y2_prediction[:,None], y2_target)

    intersect_width = (intersect_x2-intersect_x1).clamp(min=0)
    intersect_height = (intersect_y2-intersect_y1).clamp(min=0)
    intersect_area = intersect_width*intersect_height

    prediction_area = (x2_prediction-x1_prediction)*(y2_prediction-y1_prediction)
    target_area = (x2_target-x1_target)*(y2_target-y1_target)

    union_area = prediction_area[:,None] + target_area - intersect_area

    iou= intersect_area/(union_area + 1e-8)

    return iou

def _calculate_nearest_box_loss(prediction_centroids:Tensor, target_centroids:Tensor) -> float:

    if target_centroids.numel() > 0 and prediction_centroids.numel() == 0:
        return 256
    if target_centroids.numel() == 0 and prediction_centroids.numel() > 0:
        return 256
    if target_centroids.numel() == 0 and prediction_centroids.numel() == 0:
        return 0


    differences = prediction_centroids[:,None,:] - target_centroids[None, :,:]
    distances = torch.norm(differences, dim=2)
    nearest_indicies = torch.argmin(distances, dim=1)

    nearest_distances = distances[torch.arange(len(prediction_centroids)), nearest_indicies]
    return torch.sum(nearest_distances).item()


def calculate_ap_at_threshold(iou_matrix, iou_threshold):
    """
    Calculates precision and recall at a given IoU threshold, then returns AP.

    Args:
        iou_matrix: Tensor of shape (N_pred, N_gt)
        iou_threshold: float, e.g. 0.5 or 0.75

    Returns:
        precision, recall, ap (single-point precision at this threshold)
    """
    N_pred, N_gt = iou_matrix.shape
    matched_gt = torch.zeros(N_gt, dtype=torch.bool)
    matched_pred = torch.zeros(N_pred, dtype=torch.bool)

    TP = 0

    # Greedy match: go over predictions in descending max IoU order
    for pred_idx in range(N_pred):
        # Find best GT for this prediction
        ious = iou_matrix[pred_idx]
        max_iou, gt_idx = ious.max(0)

        if max_iou >= iou_threshold and not matched_gt[gt_idx]:
            TP += 1
            matched_gt[gt_idx] = True
            matched_pred[pred_idx] = True

    FP = (~matched_pred).sum().item()
    FN = (~matched_gt).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    ap = precision  # AP at fixed threshold is just precision for matched GTs

    return precision, recall, ap

def compute_ap_from_iou(iou_matrix, pred_scores, iou_threshold=0.5):
    """
    Compute AP@0.5 using a precomputed IoU matrix and prediction scores.

    Args:
        iou_matrix (Tensor[N, M]): IoU between N predicted boxes and M GT boxes.
        pred_scores (Tensor[N]): Confidence scores for each prediction.
        iou_threshold (float): IoU threshold for a match (e.g., 0.5 or 0.75).

    Returns:
        float: AP@0.5
    """
    N, M = iou_matrix.shape
    if N == 0:
        return 1.0 if M == 0 else 0.0

    # Sort predictions by descending score
    scores, sorted_indices = pred_scores.sort(descending=True)
    iou_matrix = iou_matrix[sorted_indices]

    matched_gt = torch.zeros(M, dtype=torch.bool)
    tps = []
    fps = []

    for i in range(N):
        if M == 0:
            tps.append(0.0)
            fps.append(1.0)
            continue

        ious = iou_matrix[i]
        max_iou, gt_idx = ious.max(0)

        if max_iou >= iou_threshold and not matched_gt[gt_idx]:
            tps.append(1.0)
            fps.append(0.0)
            matched_gt[gt_idx] = True
        else:
            tps.append(0.0)
            fps.append(1.0)

    tps = torch.tensor(tps).cumsum(0)
    fps = torch.tensor(fps).cumsum(0)
    recalls = tps / (M + 1e-8)
    precisions = tps / (tps + fps + 1e-8)

    ap = torch.trapz(precisions, recalls).item()
    return ap

def compute_ar_from_iou(iou_matrix, pred_scores, iou_threshold=0.5):
    """
    Compute AR@0.5 using greedy matching sorted by prediction scores.

    Args:
        iou_matrix (Tensor[N_pred, N_gt]): IoU between all predictions and GTs.
        pred_scores (Tensor[N_pred]): Confidence scores for each prediction.
        iou_threshold (float): IoU threshold to count as a match.

    Returns:
        float: AR@0.5
    """
    N_pred, N_gt = iou_matrix.shape
    if N_gt == 0:
        return 1.0 if N_pred == 0 else 0.0

    matched_gt = torch.zeros(N_gt, dtype=torch.bool)
    TP = 0

    # Sort predictions by descending score
    sorted_indices = pred_scores.sort(descending=True).indices
    iou_matrix = iou_matrix[sorted_indices]

    for i in range(N_pred):
        ious = iou_matrix[i]  # IoUs to all GTs for current prediction

        # Mask out already matched GTs
        ious[matched_gt] = -1.0

        max_iou, gt_idx = ious.max(0)
        if max_iou >= iou_threshold:
            matched_gt[gt_idx] = True
            TP += 1

    recall = TP / (N_gt + 1e-8)
    return recall

def calculate_bbox_metrics(preds: list,targets: list) -> dict:
    """
    Calculate bounding box metrics such as average precision, average recall and IOU score

    Args:
        predictions (list): List of dictionaries containing predictions with bounding boxes, confidences and labels
        targets (list): List of dictionaries containingwith bounding boxes, image, and labels.

    Returns:
        results (dict): Dictionary containing the anchor point precision, recall and F1 score
    """

    all_iou_rows = []
    all_scores = []
    total_gt_count = 0
    gt_offsets = []

    for prediction, target in zip(preds, targets):
        iou_scores = _calculate_IOU(prediction["boxes"], target["boxes"])
        scores = prediction["scores"]  # shape (N_i,)

        # Adjust ground truth indices to global count
        all_iou_rows.append(iou_scores)  # keep original shape, to be padded later
        all_scores.append(scores)
        gt_offsets.append(total_gt_count)
        total_gt_count += iou_scores.shape[1]

    # Pad all IoU matrices into a (N_total_preds, M_total_gts) global IoU matrix
    N_total = sum(x.shape[0] for x in all_iou_rows)
    M_total = total_gt_count

    global_iou = torch.zeros((N_total, M_total))
    global_scores = []

    row_offset = 0
    for i, (iou, scores, gt_off) in enumerate(zip(all_iou_rows, all_scores, gt_offsets)):
        n_preds, n_gts = iou.shape
        global_iou[row_offset:row_offset+n_preds, gt_off:gt_off+n_gts] = iou
        global_scores.append(scores)
        row_offset += n_preds

    global_scores = torch.cat(global_scores, dim=0)

    # Now compute global AP
    ap50 = compute_ap_from_iou(global_iou, global_scores, iou_threshold=0.5)
    ap75 = compute_ap_from_iou(global_iou, global_scores, iou_threshold=0.75)
    ar50 = compute_ar_from_iou(global_iou, global_scores, iou_threshold=0.50)
    ar75 = compute_ar_from_iou(global_iou, global_scores, iou_threshold=0.75)

    return {"AP50": ap50, "AP75": ap75, "AR50":ar50, "AR75":ar75 }

def centroid_accuracy(preds:list, targets:list) -> dict:
    """
    Calculates the true Positive, false positives, F1 scores for dictionaries of predictions and Outputs.

    Args:
        predictions (list): List of dictionaries containing predictions with bounding boxes, confidences and labels
        targets (list): List of dictionaries containingwith bounding boxes, image, and labels.

    Returns:
        results (dict): Dictionary containing the anchor point precision, recall and F1 score
    """
    TP = 0
    FP = 0
    FN = 0
    for prediction, target in zip(preds, targets):
        centroids = _find_centroid(prediction["boxes"])
        tp, fp, fn = _calculate_true_positives(centroids, target["boxes"])
        TP += tp
        FP += fp
        FN += fn
    recall = TP/(TP+FN+1e-8)
    precision = TP/(TP+FP+1e-8)
    f1 = precision*recall/(precision+recall+1e-8)
    return {"Anchor_F1": f1, "Anchor_Precision": precision, "Anchor_Recall ": recall}

def calculate_centroid_difference(preds:list, targets:list) -> dict:
    num_boxes = []
    total_distance = []
    avg_distance = []
    target_box_surplus = []

    for prediction, target in zip(preds, targets):
        predicted_centroids = _find_centroid(prediction["boxes"])
        target_centroids = _find_centroid(target["boxes"])

        centroid_distances = _calculate_nearest_box_loss(predicted_centroids, target_centroids)
        num_pred_boxes = len(predicted_centroids)
        num_target_boxes = len(target_centroids)

        num_boxes.append(num_pred_boxes)
        total_distance.append(centroid_distances)
        target_box_surplus.append(num_pred_boxes-num_target_boxes)
        avg_distance.append(centroid_distances/max(1, num_pred_boxes))    
    
    return {"Median_predicted_boxes": np.mean(num_boxes), "Median_total_distance": np.median(total_distance), "Median_centroid_distance": np.median(target_box_surplus), "Median_box_surplus": np.median(avg_distance),  "Mean_predicted_boxes": np.mean(num_boxes), "Mean_total_distance": np.mean(total_distance), "Mean_centroid_distance": np.mean(target_box_surplus), "Mean_box_surplus": np.mean(avg_distance)}

def calculate_centroid_difference_with_confidence(preds:list, targets:list, confidence_threshold=.90) -> dict:
    num_boxes = []
    total_distance = []
    avg_distance = []
    target_box_surplus = []

    for prediction, target in zip(preds, targets):
        predicted_centroids = _find_centroid(prediction["boxes"])
        target_centroids = _find_centroid(target["boxes"])

        boxes_above_threshold = torch.nonzero(prediction["scores"] > confidence_threshold).squeeze()

        centroid_distances = _calculate_nearest_box_loss(predicted_centroids[boxes_above_threshold,:], target_centroids)
        num_pred_boxes = len(predicted_centroids)
        num_target_boxes = len(target_centroids)

        num_boxes.append(num_pred_boxes)
        total_distance.append(centroid_distances)
        target_box_surplus.append(num_pred_boxes-num_target_boxes)
        avg_distance.append(centroid_distances/max(1, num_pred_boxes))    
    
    return {"Median_predicted_boxes_90c": np.mean(num_boxes), "Median_total_distance_90c": np.median(total_distance), "Median_centroid_distance_90c": np.median(target_box_surplus), "Median_box_surplus_90c": np.median(avg_distance),  "Mean_predicted_boxes_90c": np.mean(num_boxes), "Mean_total_distance_90c": np.mean(total_distance), "Mean_centroid_distance_90c": np.mean(target_box_surplus), "Mean_box_surplus_90c": np.mean(avg_distance)}

