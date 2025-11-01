import segeval
from sklearn.metrics import f1_score, precision_score, recall_score


def convert_to_segment_lengths(boundaries, total_len):
    seg_lengths = []
    cnt = 1
    for i in range(len(boundaries)):
        if boundaries[i] == 1:
            seg_lengths.append(cnt)
            cnt = 1
        else:
            cnt += 1
    seg_lengths.append(cnt)
    return seg_lengths


def evaluate_wd_pk_f1(pred_boundaries, true_boundaries):
    pred_seg = convert_to_segment_lengths(pred_boundaries, len(pred_boundaries))
    true_seg = convert_to_segment_lengths(true_boundaries, len(true_boundaries))
    wd = segeval.window_diff(pred_seg, true_seg)
    pk = segeval.pk(pred_seg, true_seg)
    pred_labels = [1 if b == 1 else 0 for b in pred_boundaries]
    true_labels = [1 if b == 1 else 0 for b in true_boundaries]
    f1 = f1_score(true_labels, pred_labels)
    return wd, pk, f1


def evaluate_segmentation(reference, hypothesis):
    # Calculate WD, PK, F1 using segeval
    wd, pk, f1 = evaluate_wd_pk_f1(hypothesis, reference)
    
    # Calculate Precision and Recall for compatibility
    min_length = min(len(reference), len(hypothesis))
    ref_seq = reference[:min_length]
    hyp_seq = hypothesis[:min_length]
    precision = precision_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)
    recall = recall_score(ref_seq, hyp_seq, pos_label=1, zero_division=0)

    return {
        'PK': pk,
        'WD': wd,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }
