import csv
import json
import os

import numpy as np
from tqdm import tqdm

from dialogue_dataset import DialogueDataset
from metrics import evaluate_segmentation
from model.DSAgent import DSAgent
from model.DTSAgent import DTSAgent
from model.HSAgent import HSAgent
from model.LLMReassessmentAgent import create_reassessment_agent
from model.PNAgent import PNAgent
from utils import load_config, resolve_dataset_path

# Ensure HF_ENDPOINT environment variable is set (use mirror if not set)
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def save_prediction_results_to_json(test_samples, prediction_results, handshake_results, few_shot_examples_list,
                                    similarity_examples_list):
    results = []

    for dialogue_idx, dialogue in enumerate(test_samples):
        dialogue_results = []

        real_boundaries = convert_segments_to_boundary(dialogue.segments, len(dialogue.utterances))

        for utterance_idx, utterance in enumerate(dialogue.utterances):
            input_content = dialogue.load_index(utterance_idx, 3)

            handshake_tag = "O"
            if handshake_results and dialogue_idx < len(handshake_results):
                handshake_dialogue = handshake_results[dialogue_idx]
                if utterance_idx < len(handshake_dialogue):
                    handshake_result = handshake_dialogue[utterance_idx]
                    if isinstance(handshake_result, dict) and handshake_result.get('success', False):
                        parsed = handshake_result.get('parsed_response', {})
                        if parsed and 'result' in parsed:
                            handshake_tag = parsed['result']

            pos_neg_sample = None
            if few_shot_examples_list and dialogue_idx < len(few_shot_examples_list):
                dialogue_few_shot = few_shot_examples_list[dialogue_idx]
                if utterance_idx < len(dialogue_few_shot):
                    pos_neg_sample = dialogue_few_shot[utterance_idx]

            sim_sample = None
            if similarity_examples_list and dialogue_idx < len(similarity_examples_list):
                sim_sample = similarity_examples_list[dialogue_idx]

            result_item = {
                "input_content": input_content,
                "handshake_tag": handshake_tag,
                "pos_neg_sample": pos_neg_sample,
                "sim_sample": sim_sample
            }

            dialogue_results.append(result_item)

        results.append({
            "dialogue_id": dialogue.dial_id,
            "utterances": dialogue_results
        })

    output_file = "prediction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Prediction results saved to: {output_file}")


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


def evaluate_single_dialogue(utterance, prediction_results):
    total_length = len(utterance.utterances)
    reference = convert_segments_to_boundary(utterance.segments, total_length)
    hypothesis = convert_predictions_to_boundary(prediction_results, total_length)
    min_length = min(len(reference), len(hypothesis))
    reference = reference[:min_length]
    hypothesis = hypothesis[:min_length]
    metrics = evaluate_segmentation(reference, hypothesis)

    return metrics


def run_evaluation(num_samples=None, dataset_name_or_path='vfh', hs=True, re=True, ds=True, pn=True):
    config = load_config("config.yaml")

    if num_samples is None:
        num_samples = config.get("num_samples", 4)

    print(f"=== Dialogue Topic Segmentation Evaluation for {dataset_name_or_path} ===")

    dataset = DialogueDataset(resolve_dataset_path(dataset_name_or_path))

    if num_samples == -1:
        test_samples = dataset
    else:
        test_samples = dataset[:num_samples]

    effective_turns = len(test_samples)
    if num_samples == -1:
        print(f"Number of evaluation samples: {effective_turns} (full dataset)")
    else:
        print(f"Number of evaluation samples: {effective_turns}")
    print("=" * 50)

    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]
    window_size = config.get("window_size", 3)
    num_threads = config.get("num_threads", 8)

    # Ablation study parameters: use the passed parameter values directly (all True by default)
    enable_similarity_examples = ds
    enable_few_shot_examples = pn
    enable_handshake_results = hs
    enable_reassessment = re

    print("=== Step 1: Get Segment Embeddings ===")
    ds_agent = DSAgent(dataset)
    print("Attempting to load existing segment embeddings...")
    if not ds_agent.load_segment_embeddings():
        print("No existing embeddings found, generating segment embeddings...")
        ds_agent.generate_segment_embeddings()
    else:
        print("Successfully loaded existing segment embeddings")

    similarity_examples_list = None
    if enable_similarity_examples:
        print("Precomputing similarity examples...")
        similarity_examples_list = []
        for dialogue in tqdm(test_samples, desc="Generating similarity examples", unit="dialogue"):
            try:
                segments = dialogue.get_segments()
                segment_info = dialogue.get_segment_info()
                best_overall_result = None
                best_overall_score = -1

                for seg_idx, segment in enumerate(segments):
                    if (seg_idx >= len(dialogue.segment_embeddings) or
                            dialogue.segment_embeddings[seg_idx] is None):
                        continue

                    context_segment_ids = [seg_idx]
                    result = ds_agent.find_most_similar_for_context(dialogue, context_segment_ids)

                    if result is not None and result['similarity_score'] > best_overall_score:
                        best_overall_result = result
                        best_overall_score = result['similarity_score']

                if best_overall_result is not None:
                    similar_utterance = best_overall_result['most_similar_utterance']
                    similar_segment = similar_utterance.get_segments()[best_overall_result['most_similar_segment_id']]
                    examples = [{
                        'similarity_score': best_overall_result['similarity_score'],
                        'similar_segment': similar_segment,
                        'dial_id': similar_utterance.dial_id,
                        'context_segment_ids': best_overall_result['context_segment_ids']
                    }]
                    similarity_examples_list.append(str(examples))
                else:
                    similarity_examples_list.append("No similarity examples available")
            except Exception as e:
                print(f"Warning: Error precomputing similarity examples: {e}")
                similarity_examples_list.append("No similarity examples available")
        print("Similarity example precomputation completed")
    else:
        print("Skipping similarity example generation (ablation study)")

    print("\n=== Step 2: Handshake Detection ===")
    handshake_results = None
    if enable_handshake_results:
        print("Starting Handshake detection...")
        hs_agent = HSAgent(test_samples, api_key, base_url, model, window_size=window_size)
        handshake_results = hs_agent.generate_handshake(max_turns=effective_turns, num_threads=num_threads)
        print("Handshake detection completed")
    else:
        print("Skipping Handshake detection (ablation study)")

    print("\n=== Step 3: Generate Positive/Negative Samples ===")
    few_shot_examples_list = None
    if enable_few_shot_examples:
        print("Starting positive/negative sample generation...")
        pn_agent = PNAgent(test_samples, api_key, base_url, model, window_size=7)
        few_shot_examples_list = []

        for dialogue_idx, dialogue in enumerate(
                tqdm(test_samples, desc="Generating few-shot examples", unit="dialogue")):
            dialogue_few_shot = []

            try:
                pn_results = pn_agent._process_single_dialogue(dialogue, dialogue_idx, num_threads)
            except Exception as e:
                print(f"Warning: Error generating few-shot for dialogue {dialogue_idx}: {e}")
                pn_results = [None] * len(dialogue)

            for res in pn_results:
                if isinstance(res, dict) and res.get('success', False) and res.get('parsed_response'):
                    dialogue_few_shot.append(str(res['parsed_response']['result']))
                else:
                    dialogue_few_shot.append(None)

            few_shot_examples_list.append(dialogue_few_shot)

        print("Positive/negative sample generation completed")
    else:
        print("Skipping positive/negative sample generation (ablation study)")

    print("\nSaving pre-prediction results to JSON file...")
    save_prediction_results_to_json(
        test_samples,
        None,
        handshake_results,
        few_shot_examples_list,
        similarity_examples_list
    )

    print("\n=== Step 4: Topic Segmentation Prediction ===")
    print("Starting topic segmentation prediction...")
    dts_agent = DTSAgent(test_samples, api_key, base_url, model, window_size=window_size)
    prediction_results = dts_agent.perform_dialogue_topic_segmentation(
        max_turns=effective_turns,
        num_threads=num_threads,
        handshake_results=handshake_results,
        few_shot_examples=few_shot_examples_list,
        similarity_examples=similarity_examples_list
    )

    reassessed_data = None
    if enable_reassessment:
        print("\n=== Step 5: LLM Reassessment of Consecutive 1s ===")
        print("Starting LLM reassessment...")
        reassessment_agent = create_reassessment_agent()
        reassessment_data = []
        for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
            total_length = len(utterance.utterances)
            hypothesis = convert_predictions_to_boundary(predictions, total_length)

            reassessment_data.append({
                'dialogue_id': utterance.dial_id,
                'utterances': utterance.utterances,
                'prediction': hypothesis
            })

        reassessed_data = reassessment_agent.batch_reassess(reassessment_data, num_threads=8)
        total_changes = 0
        dialogues_with_changes = 0
        for data in reassessed_data:
            if data.get('changes_made', False):
                dialogues_with_changes += 1
                total_changes += data.get('num_changes', 0)

        print(
            f"Reassessment completed: {dialogues_with_changes}/{len(reassessed_data)} dialogues changed, {total_changes} prediction points modified in total")

        updated_prediction_results = []
        for i, (utterance, data) in enumerate(zip(test_samples, reassessed_data)):
            if data.get('changes_made', False):
                optimized_prediction = data['optimized_prediction']
                updated_predictions = []

                for j, pred in enumerate(optimized_prediction):
                    if j < len(utterance.utterances) - 1:
                        if pred == 1:
                            updated_predictions.append({
                                'success': True,
                                'parsed_response': {'result': 'SEGMENT'}
                            })
                        else:
                            updated_predictions.append({
                                'success': True,
                                'parsed_response': {'result': 'NO_SEGMENT'}
                            })
                    else:
                        updated_predictions.append({
                            'success': True,
                            'parsed_response': {'result': 'NO_SEGMENT'}
                        })

                updated_prediction_results.append(updated_predictions)
            else:
                updated_prediction_results.append(prediction_results[i])

        prediction_results = updated_prediction_results
    else:
        print("\nSkipping LLM reassessment (ablation study)")
    print("\nStarting evaluation...")
    all_metrics = []
    results_data = []

    for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
        metrics = evaluate_single_dialogue(utterance, predictions)
        all_metrics.append(metrics)
        result_row = {
            'dial_id': utterance.dial_id,
            'PK': metrics['PK'],
            'WD': metrics['WD'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1']
        }
        results_data.append(result_row)

    print("\n" + "=" * 80)
    print("Evaluation completed, printing detailed information...")
    print("=" * 80)

    for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
        print(f"\n=== Dialogue {i} (dial_id: {utterance.dial_id}) ===")

        reassessment_info = None
        if reassessed_data is not None and i < len(reassessed_data):
            reassessment_info = reassessed_data[i]

        if reassessment_info and reassessment_info.get('changes_made', False):
            print(f"\n--- LLM Reassessment Results ---")
            print(f"Consecutive 1 ranges: {reassessment_info.get('consecutive_ranges', [])}")
            print(f"Number of changes: {reassessment_info.get('num_changes', 0)}")

            original_pred = reassessment_info.get('original_prediction', [])
            optimized_pred = reassessment_info.get('optimized_prediction', [])

            if original_pred and optimized_pred:
                print("Original prediction: ", " ".join(str(x) for x in original_pred))
                print("Optimized prediction: ", " ".join(str(x) for x in optimized_pred))
                print("Change markers: ",
                      " ".join("^" if orig != opt else " " for orig, opt in zip(original_pred, optimized_pred)))

        if len(utterance.utterances) > 0:
            print(f"\n--- DTS Prompt for First Utterance ---")
            first_context = utterance.load_index(0, dts_agent.window_size)

            if handshake_results:
                first_context = dts_agent._add_handshake_tags(first_context, utterance, 0, handshake_results)
            formatted_prompt = dts_agent.prompt.format_prompt(first_context, None, None)
            print(formatted_prompt)
            print("=" * 80)

        print(f"\n--- Prediction vs Reference Comparison ---")
        total_length = len(utterance.utterances)
        reference = convert_segments_to_boundary(utterance.segments, total_length)
        hypothesis = convert_predictions_to_boundary(predictions)
        min_length = min(len(reference), len(hypothesis))
        reference = reference[:min_length]
        hypothesis = hypothesis[:min_length]

        original_prediction = None
        if reassessment_info and reassessment_info.get('changes_made', False):
            original_prediction = reassessment_info.get('original_prediction', [])
            if original_prediction:
                original_prediction = original_prediction[:min_length]

        if original_prediction:
            print("utt_id | pred | real | reassess | diff")
            print("-" * 40)
        else:
            print("utt_id | pred | real | diff")
            print("-" * 25)

        for utt_id in range(min_length):
            pred = hypothesis[utt_id]
            real = reference[utt_id]
            diff = "✓" if pred == real else "✗"
            
            if original_prediction and utt_id < len(original_prediction):
                orig_pred = original_prediction[utt_id]
                if orig_pred != pred:
                    reassess_str = f"{orig_pred}->{pred}"
                else:
                    reassess_str = "-"
                print(f"{utt_id:5d} | {pred:4d} | {real:4d} | {reassess_str:9s} | {diff}")
            else:
                print(f"{utt_id:5d} | {pred:4d} | {real:4d} | {diff}")

        correct = sum(1 for p, r in zip(hypothesis, reference) if p == r)
        accuracy = correct / min_length if min_length > 0 else 0
        print(f"\nAccuracy: {correct}/{min_length} = {accuracy:.4f}")

        print(f"\n--- Evaluation Metrics ---")
        for metric, value in all_metrics[i].items():
            print(f"{metric}: {value:.4f}")

        print("\n" + "=" * 80)

    print(f"\n=== Overall Average ===")
    avg_metrics = {}
    for metric in ['PK', 'WD', 'Precision', 'Recall', 'F1']:
        avg_value = np.mean([m[metric] for m in all_metrics])
        avg_metrics[metric] = avg_value
        print(f"{metric}: {avg_value:.4f}")

    csv_filename = "evaluation_results.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dial_id', 'PK', 'WD', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results_data:
            writer.writerow(row)

        avg_row = {'dial_id': 'average'}
        avg_row.update(avg_metrics)
        writer.writerow(avg_row)

    print(f"\nResults saved to: {csv_filename}")
    print("=" * 50)

    return all_metrics, avg_metrics


if __name__ == "__main__":
    import argparse
    
    def str_to_bool(v):
        """Convert string to boolean value"""
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError(f'Boolean value expected, but received: {v}')
    
    parser = argparse.ArgumentParser(description='Dialogue topic segmentation evaluation script (supports ablation studies)')
    parser.add_argument('--dataset', type=str, default='vfh', help='vfh | dialseg_711 | doc2dial or path to json')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--hs', type=str_to_bool, nargs='?', const=True, default=True, 
                       help='Whether to use handshake detection (ablation study parameter, default True, can pass True/False)')
    parser.add_argument('--re', type=str_to_bool, nargs='?', const=True, default=True,
                       help='Whether to use reassessment (ablation study parameter, default True, can pass True/False)')
    parser.add_argument('--ds', type=str_to_bool, nargs='?', const=True, default=True,
                       help='Whether to use dialogue similarity (ablation study parameter, default True, can pass True/False)')
    parser.add_argument('--pn', type=str_to_bool, nargs='?', const=True, default=True,
                       help='Whether to use positive negative samples (ablation study parameter, default True, can pass True/False)')
    
    args = parser.parse_args()

    run_evaluation(
        num_samples=args.num_samples, 
        dataset_name_or_path=args.dataset,
        hs=args.hs,
        re=args.re,
        ds=args.ds,
        pn=args.pn
    )
