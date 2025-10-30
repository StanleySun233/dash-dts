from typing import List, Tuple


def compute_pk(reference: List[int], hypothesis: List[int], k: int = None) -> float:
    n = len(reference)
    if k is None:
        k = max(1, n // 2)

    pk_sum = 0.0

    for i in range(n - k + 1):
        ref_window = reference[i:i + k]
        hyp_window = hypothesis[i:i + k]

        # Calculate segment count difference in window
        ref_count = sum(ref_window)
        hyp_count = sum(hyp_window)

        pk_sum += abs(ref_count - hyp_count) / k

    return pk_sum / (n - k + 1)


def compute_wd(reference: List[int], hypothesis: List[int]) -> float:
    n = len(reference)
    wd_sum = 0.0

    for k in range(1, n + 1):
        pk_value = compute_pk(reference, hypothesis, k)
        wd_sum += pk_value

    return wd_sum / n


def compute_segmentation_f1(reference: List[int], hypothesis: List[int],
                            window_tolerance: int = 1) -> Tuple[float, float, float]:
    n = len(reference)

    # Get true segment positions
    true_segments = [i for i, val in enumerate(reference) if val == 1]
    # Get predicted segment positions
    pred_segments = [i for i, val in enumerate(hypothesis) if val == 1]

    # Calculate matched segments
    matched_true = set()
    matched_pred = set()

    for i, true_pos in enumerate(true_segments):
        for j, pred_pos in enumerate(pred_segments):
            if abs(true_pos - pred_pos) <= window_tolerance:
                matched_true.add(i)
                matched_pred.add(j)

    # Calculate metrics
    tp = len(matched_true)
    fp = len(pred_segments) - len(matched_pred)
    fn = len(true_segments) - len(matched_true)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_segmentation(reference: List[int], hypothesis: List[int]) -> dict:
    # Calculate PK
    pk = compute_pk(reference, hypothesis)

    # Calculate WD
    wd = compute_wd(reference, hypothesis)

    # Calculate F1
    precision, recall, f1 = compute_segmentation_f1(reference, hypothesis)

    return {
        'PK': pk,
        'WD': wd,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }