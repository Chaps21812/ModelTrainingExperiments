from torch.types import Tensor
import numpy as np
import torch

def _find_centroid(tensor: Tensor) -> Tensor:
    """
    Find the centroid of the bounding box given XC, YC, W, H.Tensor must be in shape [5,N].

    Args:
        tensor (tensor): tensor bounding box [XC, YC, W, H], shape [5,N]

    Returns:
        centroids (tensor): Torch tensor of centroids
    """
    x = tensor[0,:]
    y = tensor[1,:]
    if x.ndim == 1:
        x.unsqueeze(0)
    if y.ndim == 1:
        y.unsqueeze(0)
    if x.ndim == 3:
        x.squeeze()
    if y.ndim == 3:
        y.squeeze()
    centroids = torch.stack((x,y), dim=0).transpose(1,0)
    return centroids

def _calculate_true_positives(centroids:Tensor, targets:Tensor) -> tuple[Tensor, Tensor]:
    """
    Find the centroid of the bounding box given XC, YC, W, H.Tensor must be in shape [N,4].

    Args:
        tensor (tensor): tensor bounding box [XC, YC, W, H], shape [N,4]

    Returns:
        centroids (tensor): List of files present in the train test split
    """
    predicted_x = centroids[:,0].unsqueeze(1)
    predicted_y = centroids[:,1].unsqueeze(1)

    target_x_centroid = targets[0,:].unsqueeze(0)
    target_y_centroid = targets[1,:].unsqueeze(0)
    target_width = targets[2,:].unsqueeze(0)
    target_height = targets[3,:].unsqueeze(0)

    inside_x = (predicted_x >= target_x_centroid - target_width/2) & (predicted_x <= target_x_centroid + target_width/2)
    inside_y = (predicted_y >= target_y_centroid - target_height/2) & (predicted_y <= target_y_centroid + target_height/2)
    truth_table = (inside_x & inside_y)
    target_matched = truth_table.any(dim=0)
    prediction_matched = truth_table.any(dim=1)

    true_positives = target_matched.sum()
    false_negatives = (~prediction_matched).sum()
    false_positives = (~target_matched).sum()

    return true_positives.item(), false_positives.item(), false_negatives.item()

def _calculate_IOU(predictions: Tensor, targets: Tensor):
    xc_prediction, yc_prediction, width_prediction, height_prediction = predictions[0, :], predictions[1, :], predictions[2, :], predictions[3, :]
    xc_target, yc_target, width_target, height_target = targets[0, :], targets[1, :], targets[2, :], targets[3, :]

    x1_prediction = xc_prediction-width_prediction/2
    y1_prediction = yc_prediction-height_prediction/2
    x2_prediction = xc_prediction+width_prediction/2
    y2_prediction = yc_prediction+height_prediction/2

    x1_target = xc_target-width_target/2
    y1_target = yc_target-height_target/2
    x2_target = xc_target+width_target/2
    y2_target = yc_target+height_target/2

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

def _centroid_metric(pred, tgt, H=None, W=None, unmatched_lambda=1.0, eps=1e-6):
    """
    pred, tgt: [N, 2] tensors (x, y) in pixels or normalized coordinates.
    If in pixels, pass H, W to set the unmatched cost. If normalized, leave H/W None.
    Returns a normalized loss where 0 is perfect, 1 is ~worst-case (given our normalization).
    """
    n_p, n_g = len(pred), len(tgt)

    # Handle trivial cases
    if n_p == 0 and n_g == 0:
        return 0.0

    # Max possible distance
    if H is not None and W is not None:
        c_max = (H**2 + W**2) ** 0.5
    else:
        # assume coords in [0,1]
        c_max = 2.0 ** 0.5

    if n_p == 0 or n_g == 0:
        # All boxes are unmatched; penalize each one
        unmatched = max(n_p, n_g)
        loss = unmatched_lambda * unmatched * c_max
        return loss / (unmatched * c_max + eps)  # → 1.0

    # Pairwise distances
    d = torch.cdist(pred, tgt)  # [n_p, n_g]

    # Chamfer-like symmetric sum of nearest distances
    pred_to_tgt = d.min(dim=1).values  # [n_p]
    tgt_to_pred = d.min(dim=0).values  # [n_g]

    chamfer = pred_to_tgt.sum() + tgt_to_pred.sum()

    # Count mismatch cost (how many we "couldn't" match 1-1)
    unmatched = abs(n_p - n_g)
    count_pen = unmatched_lambda * unmatched * c_max

    # Normalize to something that lives roughly in [0, 1]
    # denom = (n_p + n_g) * c_max + eps
    total_count = n_p + n_g

    avg_distance = (chamfer + count_pen) / (total_count + eps)
    return avg_distance.item()

def calculate_ap_at_threshold(iou_matrix, iou_threshold, image_height=None):
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

def compute_ap_from_iou(iou_matrix, pred_scores, iou_threshold=0.5, image_height=None):
    """
    Compute AP@0.5 using a precomputed IoU matrix and prediction scores.

    Args:
        iou_matrix (Tensor[N, M]): IoU between N predicted boxes and M GT boxes.
        pred_scores (Tensor[N]): Confidence scores for each prediction.
        iou_threshold (float): IoU threshold for a match (e.g., 0.5 or 0.75).

    Returns:
        float: AP@0.5
    """
    if iou_matrix.ndim==3:
        print(iou_matrix.shape)
    N, M = iou_matrix.shape
    if N == 0:
        return 1.0 if M == 0 else 0.0

    # Sort predictions by descending score
    scores, sorted_indices = pred_scores.sort(descending=True)
    sorted_indices = sorted_indices.squeeze() if sorted_indices.ndim > 1 else sorted_indices
    iou_matrix = iou_matrix[sorted_indices]

    matched_gt = torch.zeros(M, dtype=torch.bool)
    tps = []
    fps = []

    for i in range(N):
        if M == 0:
            tps.append(0.0)
            fps.append(1.0)
            continue

        if iou_matrix.ndim==3:
            print(iou_matrix.shape)
        ious = iou_matrix[i]
        max_iou, gt_idx = ious.max(0)
        if iou_matrix.ndim==3:
            print(iou_matrix.shape)

        # max_iou = max_iou[0].item() if max_iou.numel() > 1 else max_iou
        # matched_target =  matched_gt[gt_idx][0].item() if matched_gt[gt_idx].numel() > 1 else matched_gt[gt_idx]
        matched_target =  matched_gt[gt_idx]

        if max_iou >= iou_threshold and not matched_target:
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

def compute_ar_from_iou(iou_matrix, pred_scores, iou_threshold=0.5, image_height=None):
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
    matched_gt = matched_gt.squeeze() if matched_gt.ndim > 1 else matched_gt
    TP = 0

    # Sort predictions by descending score
    sorted_indices = pred_scores.sort(descending=True).indices
    # sorted_indices = sorted_indices.squeeze() if sorted_indices.ndim() > 1 else sorted_indices
    iou_matrix = iou_matrix[sorted_indices]

    for i in range(N_pred):
        ious = iou_matrix[i]  # IoUs to all GTs for current prediction
        ious = ious.squeeze() if ious.ndim > 1 else ious

        # Mask out already matched GTs
        ious[matched_gt] = -1.0

        max_iou, gt_idx = ious.max(0)
        if max_iou >= iou_threshold:
            matched_gt[gt_idx] = True
            TP += 1

    recall = TP / (N_gt + 1e-8)
    return recall

def calculate_bbox_metrics(preds: list,targets: list, image_height=None) -> dict:
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
        mask = prediction[4] != 0
        processed_predictions = prediction[:,mask]
        mask = target[4] != 0
        processed_targets = target[:,mask]


        iou_scores = _calculate_IOU(processed_predictions[:4,:], processed_targets[:4,:])
        scores = processed_predictions[4,:]  # shape (N_i,)

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

def centroid_accuracy(preds:list, targets:list, image_height=None) -> dict:
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
        mask = prediction[4] != 0
        processed_predictions = prediction[:,mask]
        mask = target[4] != 0
        processed_targets = target[:,mask]

        centroids = _find_centroid(processed_predictions[:4,:])
        tp, fp, fn = _calculate_true_positives(centroids, processed_targets[:4,:])
        TP += tp
        FP += fp
        FN += fn
    recall = TP/(TP+FN+1e-8)
    precision = TP/(TP+FP+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    return {"Anchor_F1": f1, "Anchor_Precision": precision, "Anchor_Recall ": recall}

def calculate_centroid_difference(preds:torch.Tensor, targets:torch.Tensor, image_height=None) -> dict:
    num_boxes = []
    total_distance = []
    avg_distance = []
    target_box_surplus = []

    for prediction, target in zip(preds, targets):
        mask = prediction[4] != 0
        processed_predictions = prediction[:,mask]
        mask = target[4] != 0
        processed_targets = target[:,mask]

        predicted_centroids = _find_centroid(processed_predictions[:4,:])
        target_centroids = _find_centroid(processed_targets[:4,:])

        if image_height is None:
            centroid_distances = _calculate_nearest_box_loss(predicted_centroids, target_centroids)
        else:
            centroid_distances = _centroid_metric(predicted_centroids, target_centroids, H=image_height[1], W=image_height[0])
        num_pred_boxes = len(predicted_centroids)
        num_target_boxes = len(target_centroids)

        num_boxes.append(num_pred_boxes)
        total_distance.append(centroid_distances)
        target_box_surplus.append(num_pred_boxes-num_target_boxes)
        avg_distance.append(centroid_distances/max(1, num_pred_boxes))    
    
    return {"Median_predicted_boxes": np.mean(num_boxes), "Median_centroid_distance": np.median(avg_distance),  "Mean_predicted_boxes": np.mean(num_boxes), "Mean_centroid_distance": np.mean(avg_distance)}
