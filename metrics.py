from sklearn.metrics import f1_score, precision_score, recall_score

def compute_pk(reference, hypothesis, k: int = None) -> float:
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


def compute_wd(reference, hypothesis) -> float:
    n = len(reference)
    wd_sum = 0.0

    for k in range(1, n + 1):
        pk_value = compute_pk(reference, hypothesis, k)
        wd_sum += pk_value

    return wd_sum / n


def compute_segmentation_f1(reference, hypothesis,
                            window_tolerance: int = 1) -> Tuple[float, float, float]:
    # Ensure both sequences have the same length
    min_length = min(len(reference), len(hypothesis))
    ref_seq = reference[:min_length]
    hyp_seq = hypothesis[:min_length]
    precision = precision_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)
    recall = recall_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)
    f1 = f1_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)
    
    return precision, recall, f1


def evaluate_segmentation(reference, hypothesis) -> dict:
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
