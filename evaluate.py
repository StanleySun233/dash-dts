import csv
import json

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


def save_prediction_results_to_json(test_samples, prediction_results, handshake_results, few_shot_examples_list,
                                    similarity_examples_list):
    results = []

    for dialogue_idx, dialogue in enumerate(test_samples):
        dialogue_results = []

        # Get real segmentation boundaries
        real_boundaries = convert_segments_to_boundary(dialogue.segments, len(dialogue.utterances))

        for utterance_idx, utterance in enumerate(dialogue.utterances):
            # Get input content with context (previous, current, next), fixed window size 3
            input_content = dialogue.load_index(utterance_idx, 3)

            # Get handshake tag
            handshake_tag = "O"  # Default tag
            if handshake_results and dialogue_idx < len(handshake_results):
                handshake_dialogue = handshake_results[dialogue_idx]
                if utterance_idx < len(handshake_dialogue):
                    handshake_result = handshake_dialogue[utterance_idx]
                    if isinstance(handshake_result, dict) and handshake_result.get('success', False):
                        parsed = handshake_result.get('parsed_response', {})
                        if parsed and 'result' in parsed:
                            handshake_tag = parsed['result']

            # Get positive/negative samples
            pos_neg_sample = None
            if few_shot_examples_list and dialogue_idx < len(few_shot_examples_list):
                dialogue_few_shot = few_shot_examples_list[dialogue_idx]
                if utterance_idx < len(dialogue_few_shot):
                    pos_neg_sample = dialogue_few_shot[utterance_idx]

            # Get similarity samples
            sim_sample = None
            if similarity_examples_list and dialogue_idx < len(similarity_examples_list):
                sim_sample = similarity_examples_list[dialogue_idx]

            # Create result dictionary
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

    # 保存到JSON文件
    output_file = "prediction_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"预测结果已保存到: {output_file}")


def convert_segments_to_boundary(segments, total_length):
    boundary = [0] * total_length

    current_pos = 0
    for i, segment_length in enumerate(segments):
        if segment_length > 0:
            current_pos += segment_length
            # 除了最后一段，每段结束后标记为分割点
            if i < len(segments) - 1 and current_pos < total_length:
                boundary[current_pos - 1] = 1

    return boundary


def convert_predictions_to_boundary(predictions, total_length=None):
    boundary = []

    for i, pred in enumerate(predictions):
        # 如果是最后一个utterance，直接设为0（不分割）
        if total_length is not None and i == total_length - 1:
            boundary.append(0)
            continue

        if isinstance(pred, dict) and pred.get('success', False):
            parsed = pred.get('parsed_response', {})
            if parsed and parsed.get('result') == 'SEGMENT':
                boundary.append(1)
            else:
                boundary.append(0)
        else:
            # 预测失败时默认为0
            boundary.append(0)

    return boundary


def evaluate_single_dialogue(utterance, prediction_results):
    # 获取金标边界序列
    total_length = len(utterance.utterances)
    reference = convert_segments_to_boundary(utterance.segments, total_length)

    # 获取预测边界序列
    hypothesis = convert_predictions_to_boundary(prediction_results, total_length)

    # 确保长度一致
    min_length = min(len(reference), len(hypothesis))
    reference = reference[:min_length]
    hypothesis = hypothesis[:min_length]

    # 计算指标
    metrics = evaluate_segmentation(reference, hypothesis)

    return metrics


def run_evaluation(num_samples=None, dataset_name_or_path='vfh'):
    """
    主评估流程
    
    Args:
        num_samples: 评估样本数量，如果为None则从config读取
    """
    # 加载配置和数据
    config = load_config("config.yaml")

    # 从config读取num_samples，如果没有则使用默认值
    if num_samples is None:
        num_samples = config.get("num_samples", 4)

    print(f"=== {dataset_name_or_path} 对话主题分割评估 ===")

    dataset = DialogueDataset(resolve_dataset_path(dataset_name_or_path))

    # 样本选择：-1 表示全量
    if num_samples == -1:
        test_samples = dataset
    else:
        test_samples = dataset[:num_samples]

    effective_turns = len(test_samples)
    if num_samples == -1:
        print(f"评估样本数量: {effective_turns} (全量)")
    else:
        print(f"评估样本数量: {effective_turns}")
    print("=" * 50)

    # 从config中读取所有配置
    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]
    window_size = config.get("window_size", 3)
    num_threads = config.get("num_threads", 8)

    # 消融实验配置
    enable_similarity_examples = config.get("enable_similarity_examples", True)
    enable_few_shot_examples = config.get("enable_few_shot_examples", True)
    enable_handshake_results = config.get("enable_handshake_results", True)

    # 第一步：先获取embeddings，并预计算相似样本
    print("=== 第一步：获取Segment Embeddings ===")
    ds_agent = DSAgent(dataset)  # 传递完整的数据集对象，而不是样本列表
    # 尝试加载已有的segment embeddings，如果不存在则生成
    print("尝试加载已有的segment embeddings...")
    if not ds_agent.load_segment_embeddings():
        print("未找到已有的embeddings，开始生成segment embeddings...")
        ds_agent.generate_segment_embeddings()
    else:
        print("成功加载已有的segment embeddings")

    # 预计算每个对话的相似样本（供DTSAgent直接使用）
    similarity_examples_list = None
    if enable_similarity_examples:
        print("预计算相似样本...")
        similarity_examples_list = []
        for dialogue in tqdm(test_samples, desc="Generating similarity examples", unit="dialogue"):
            try:
                # 获取对话的所有segments
                segments = dialogue.get_segments()
                segment_info = dialogue.get_segment_info()

                # 为每个可能的segment组合找到最相似的样本
                # 这里我们简化处理：为每个segment单独找最相似的，然后选择最好的
                best_overall_result = None
                best_overall_score = -1

                for seg_idx, segment in enumerate(segments):
                    if (seg_idx >= len(dialogue.segment_embeddings) or
                            dialogue.segment_embeddings[seg_idx] is None):
                        continue

                    # 使用新的方法，排除所有相关的segments
                    # 这里我们假设当前segment是主要关注的，所以排除它
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
        print("相似样本预计算完成")
    else:
        print("跳过相似样本生成（消融实验）")

    # 第二步：执行Handshake检测
    print("\n=== 第二步：Handshake检测 ===")
    handshake_results = None
    if enable_handshake_results:
        print("开始进行Handshake检测...")
        hs_agent = HSAgent(test_samples, api_key, base_url, model, window_size=window_size)
        handshake_results = hs_agent.generate_handshake(max_turns=effective_turns, num_threads=num_threads)
        print("Handshake检测完成")
    else:
        print("跳过Handshake检测（消融实验）")

    # 第三步：生成正负样本（few-shot示例，预计算传入DTSAgent）
    print("\n=== 第三步：生成正负样本 ===")
    few_shot_examples_list = None
    if enable_few_shot_examples:
        print("开始生成正负样本...")
        pn_agent = PNAgent(test_samples, api_key, base_url, model, window_size=7)
        few_shot_examples_list = []

        for dialogue_idx, dialogue in enumerate(
                tqdm(test_samples, desc="Generating few-shot examples", unit="dialogue")):
            dialogue_few_shot = []

            # 多线程为该对话的每个 utterance 生成正负样本
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

        print("正负样本生成完成")
    else:
        print("跳过正负样本生成（消融实验）")

    # 保存预测前的结果到JSON文件（使用真实分割数据）
    print("\n保存预测前结果到JSON文件...")
    save_prediction_results_to_json(
        test_samples,
        None,  # 不使用预测结果
        handshake_results,
        few_shot_examples_list,
        similarity_examples_list
    )

    # 第四步：开始进行主题分割预测
    print("\n=== 第四步：主题分割预测 ===")
    print("开始进行主题分割预测...")
    # 使用handshake结果初始化DTSAgent
    dts_agent = DTSAgent(test_samples, api_key, base_url, model, window_size=window_size)

    # 进行预测（DTSAgent直接使用外部传入的few-shot与similarity示例）
    prediction_results = dts_agent.perform_dialogue_topic_segmentation(
        max_turns=effective_turns,
        num_threads=num_threads,
        handshake_results=handshake_results,
        few_shot_examples=few_shot_examples_list,
        similarity_examples=similarity_examples_list
    )

    # 第五步：LLM重新评估连续1的情况
    print("\n=== 第五步：LLM重新评估连续1情况 ===")
    print("开始进行LLM重新评估...")

    # 创建重新评估代理
    reassessment_agent = create_reassessment_agent()

    # 准备重新评估的数据
    reassessment_data = []
    for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
        # 获取预测的边界序列
        total_length = len(utterance.utterances)
        hypothesis = convert_predictions_to_boundary(predictions, total_length)

        reassessment_data.append({
            'dialogue_id': utterance.dial_id,
            'utterances': utterance.utterances,
            'prediction': hypothesis
        })

    # 批量重新评估
    reassessed_data = reassessment_agent.batch_reassess(reassessment_data, num_threads=8)

    # 统计重新评估结果
    total_changes = 0
    dialogues_with_changes = 0
    for data in reassessed_data:
        if data.get('changes_made', False):
            dialogues_with_changes += 1
            total_changes += data.get('num_changes', 0)

    print(
        f"重新评估完成：{dialogues_with_changes}/{len(reassessed_data)} 个对话发生变化，总共 {total_changes} 个预测点被修改")

    # 更新预测结果
    updated_prediction_results = []
    for i, (utterance, data) in enumerate(zip(test_samples, reassessed_data)):
        if data.get('changes_made', False):
            # 将优化后的预测转换回DTSAgent格式
            optimized_prediction = data['optimized_prediction']
            updated_predictions = []

            for j, pred in enumerate(optimized_prediction):
                if j < len(utterance.utterances) - 1:  # 不是最后一个utterance
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
                    # 最后一个utterance总是NO_SEGMENT
                    updated_predictions.append({
                        'success': True,
                        'parsed_response': {'result': 'NO_SEGMENT'}
                    })

            updated_prediction_results.append(updated_predictions)
        else:
            # 没有变化，保持原预测
            updated_prediction_results.append(prediction_results[i])

    # 使用更新后的预测结果
    prediction_results = updated_prediction_results

    print("\n开始评估...")
    # 评估每个对话
    all_metrics = []
    results_data = []

    # 先进行所有评估，不打印详细信息
    for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
        # 评估单个对话
        metrics = evaluate_single_dialogue(utterance, predictions)
        all_metrics.append(metrics)

        # 保存到结果数据
        result_row = {
            'dial_id': utterance.dial_id,
            'PK': metrics['PK'],
            'WD': metrics['WD'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1': metrics['F1']
        }
        results_data.append(result_row)

    # tqdm结束后，打印详细信息
    print("\n" + "=" * 80)
    print("评估完成，开始打印详细信息...")
    print("=" * 80)

    for i, (utterance, predictions) in enumerate(zip(test_samples, prediction_results)):
        print(f"\n=== Dialogue {i} (dial_id: {utterance.dial_id}) ===")

        # 检查是否有重新评估结果
        reassessment_info = None
        if i < len(reassessed_data):
            reassessment_info = reassessed_data[i]

        # 打印重新评估信息
        if reassessment_info and reassessment_info.get('changes_made', False):
            print(f"\n--- LLM重新评估结果 ---")
            print(f"连续1范围: {reassessment_info.get('consecutive_ranges', [])}")
            print(f"修改数量: {reassessment_info.get('num_changes', 0)}")

            # 显示原始预测和优化后预测的对比
            original_pred = reassessment_info.get('original_prediction', [])
            optimized_pred = reassessment_info.get('optimized_prediction', [])

            if original_pred and optimized_pred:
                print("原始预测: ", " ".join(str(x) for x in original_pred))
                print("优化预测: ", " ".join(str(x) for x in optimized_pred))
                print("变化标记: ",
                      " ".join("^" if orig != opt else " " for orig, opt in zip(original_pred, optimized_pred)))

        # 打印第一个utterance的DTS提示词
        if len(utterance.utterances) > 0:
            print(f"\n--- 第一个utterance的DTS提示词 ---")
            first_context = utterance.load_index(0, dts_agent.window_size)

            # 如果有handshake结果，添加handshake标签
            if handshake_results:
                first_context = dts_agent._add_handshake_tags(first_context, utterance, 0, handshake_results)

            # 生成提示词
            formatted_prompt = dts_agent.prompt.format_prompt(first_context, None, None)
            print(formatted_prompt)
            print("=" * 80)

        # 打印预测结果和真实结果的差异
        print(f"\n--- 预测结果 vs 真实结果对比 ---")
        total_length = len(utterance.utterances)
        reference = convert_segments_to_boundary(utterance.segments, total_length)
        hypothesis = convert_predictions_to_boundary(predictions)

        # 确保长度一致
        min_length = min(len(reference), len(hypothesis))
        reference = reference[:min_length]
        hypothesis = hypothesis[:min_length]

        print("utt_id | pred | real | 差异")
        print("-" * 25)
        for utt_id in range(min_length):
            pred = hypothesis[utt_id]
            real = reference[utt_id]
            diff = "✓" if pred == real else "✗"
            print(f"{utt_id:5d} | {pred:4d} | {real:4d} | {diff}")

        # 计算准确率
        correct = sum(1 for p, r in zip(hypothesis, reference) if p == r)
        accuracy = correct / min_length if min_length > 0 else 0
        print(f"\n准确率: {correct}/{min_length} = {accuracy:.4f}")

        # 打印结果
        print(f"\n--- 评估指标 ---")
        for metric, value in all_metrics[i].items():
            print(f"{metric}: {value:.4f}")

        print("\n" + "=" * 80)

    # 计算总体平均指标
    print(f"\n=== Overall Average ===")
    avg_metrics = {}
    for metric in ['PK', 'WD', 'Precision', 'Recall', 'F1']:
        avg_value = np.mean([m[metric] for m in all_metrics])
        avg_metrics[metric] = avg_value
        print(f"{metric}: {avg_value:.4f}")

    # 保存结果到CSV
    csv_filename = "evaluation_results.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['dial_id', 'PK', 'WD', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in results_data:
            writer.writerow(row)

        # 添加平均行
        avg_row = {'dial_id': 'average'}
        avg_row.update(avg_metrics)
        writer.writerow(avg_row)

    print(f"\n结果已保存到: {csv_filename}")
    print("=" * 50)

    return all_metrics, avg_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vfh', help='vfh | dialseg_711 | doc2dial or path to json')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    # 从config读取配置进行评估
    run_evaluation(num_samples=args.num_samples, dataset_name_or_path=args.dataset)
