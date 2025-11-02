import json


def parse_llm_response(response_text):
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        response_text = response_text[start:end].strip()

    parsed_response = json.loads(response_text)
    if "result" in parsed_response:
        return parsed_response["result"]
    else:
        return parsed_response


def convert_numpy_types(obj):
    import numpy as np
    from decimal import Decimal

    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle Decimal types (from segeval library)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        # Recursively process dictionary, but handle numeric string values
        result = {}
        for key, value in obj.items():
            # For metrics keys (PK, WD, Precision, Recall, F1), ensure numeric values are floats
            if key in ['PK', 'WD', 'Precision', 'Recall', 'F1'] and isinstance(value, str):
                try:
                    result[key] = float(value)
                except (ValueError, TypeError):
                    result[key] = convert_numpy_types(value)
            else:
                result[key] = convert_numpy_types(value)
        return result
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def convert_segments_to_boundary(segments, total_length):
    boundary = [0] * total_length

    current_pos = 0
    for i, segment_length in enumerate(segments):
        if segment_length > 0:
            current_pos += segment_length
            if i < len(segments) - 1 and current_pos < total_length:
                boundary[current_pos - 1] = 1

    return boundary


def convert_predictions_to_boundary(predictions, total_length=None):
    boundary = []

    for i, pred in enumerate(predictions):
        # 第一个位置永远不应该是分割点（对话开始前没有边界）
        if i == 0:
            boundary.append(0)
            continue

        if isinstance(pred, dict) and pred.get('success', False):
            parsed = pred.get('parsed_response', {})
            if parsed and parsed.get('result') == 'SEGMENT':
                boundary.append(1)
            else:
                boundary.append(0)
        else:
            boundary.append(0)

    return boundary

import yaml
import os


def load_config(path="config.yaml"):
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Expand environment variables (supports $VAR and ${VAR} formats)
        content = os.path.expandvars(content)
        config = yaml.load(content, Loader=yaml.FullLoader)
    return config


def resolve_dataset_path(dataset_name_or_path: str) -> str:
    if not dataset_name_or_path:
        dataset_name_or_path = 'vfh'
    if dataset_name_or_path.lower().endswith('.json'):
        return dataset_name_or_path
    mapping = {
        'vfh': './dataset/vhf.json',
        'dialseg_711': './dataset/dialseg_711.json',
        'doc2dial': './dataset/doc2dial.json',
    }
    key = dataset_name_or_path.lower()
    return mapping.get(key, mapping['vfh'])