import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

from llm_api import LLMAPI
from prompt.llm_reassessment import LLMReassessmentPrompt
from utils import load_config
from tqdm import tqdm


class LLMReassessmentAgent:

    def __init__(self, api_key: str, base_url: str, model: str, window_size: int = 3):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm_api = LLMAPI(api_key, base_url, model)
        self.prompt = LLMReassessmentPrompt()
        # 窗口大小：指在连续范围两侧各取多少条上下文（不含连续范围内部）
        self.window_size = window_size

    def detect_consecutive_ones(self, prediction: List[int]) -> List[Tuple[int, int]]:
        consecutive_ranges = []
        start = None

        for i, val in enumerate(prediction):
            if val == 1:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= 1:  # At least 2 consecutive 1s
                        consecutive_ranges.append((start, i - 1))
                    start = None

        # Handle consecutive 1s at sequence end
        if start is not None and len(prediction) - start >= 1:
            consecutive_ranges.append((start, len(prediction) - 1))

        return consecutive_ranges

    def reassess_consecutive_ones(self, utterances: List[str], prediction: List[int],
                                  consecutive_ranges: List[Tuple[int, int]], num_threads: int = 8) -> List[int]:
        """
        使用LLM重新评估连续1的情况（多线程版本）
        
        Args:
            utterances: 对话序列
            prediction: 原始预测
            consecutive_ranges: 连续1的位置范围
            num_threads: 线程数量，默认8
            
        Returns:
            优化后的预测序列
        """
        if not consecutive_ranges:
            return prediction.copy()

        # 创建优化后的预测
        optimized_prediction = prediction.copy()

        # 使用多线程处理连续1范围
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {
                executor.submit(
                    self._process_single_consecutive_range,
                    utterances,
                    prediction,
                    start,
                    end
                ): idx
                for idx, (start, end) in enumerate(consecutive_ranges)
            }

            results = {}
            with tqdm(total=len(consecutive_ranges), desc="Reassessing consecutive ranges", unit="range") as pbar:
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[idx] = result
                    except Exception as exc:
                        print(f'Error processing consecutive range {idx}: {exc}')
                        # 使用启发式方法作为后备
                        start, end = consecutive_ranges[idx]
                        results[idx] = {
                            'context_start': max(0, start - self.window_size),
                            'context_end': min(len(utterances), end + self.window_size + 1),
                            'optimized_segment': self._heuristic_optimization_segment(prediction, start, end),
                            'success': False
                        }
                    finally:
                        pbar.update(1)

        # 按顺序应用结果
        for idx in range(len(consecutive_ranges)):
            if idx in results:
                result = results[idx]
                context_start = result['context_start']
                context_end = result['context_end']
                optimized_segment = result['optimized_segment']

                # 更新优化后的预测
                for i, val in enumerate(optimized_segment):
                    if context_start + i < len(optimized_prediction):
                        optimized_prediction[context_start + i] = val

        return optimized_prediction

    def _process_single_consecutive_range(self, utterances: List[str], prediction: List[int],
                                          start: int, end: int) -> Dict:
        """
        处理单个连续1范围
        
        Args:
            utterances: 对话序列
            prediction: 原始预测
            start: 连续1的起始位置
            end: 连续1的结束位置
            
        Returns:
            包含优化结果的字典
        """
        # 提取相关对话片段
        context_start = max(0, start - self.window_size)
        context_end = min(len(utterances), end + self.window_size + 1)
        context_utterances = utterances[context_start:context_end]
        context_prediction = prediction[context_start:context_end]

        # 构建提示词
        prompt = self._build_reassessment_prompt(
            context_utterances,
            context_prediction,
            (start - context_start, end - context_start),
            utterances[max(0, context_start - self.window_size):context_start] if context_start > 0 else None,
            utterances[start:end + 1],
            utterances[end + 1:min(len(utterances), end + 1 + self.window_size)]
        )

        try:
            # 调用LLM进行重新评估
            response = self.llm_api.generate_response(prompt)

            # 解析LLM响应
            optimized_segment = self._parse_llm_response(response, len(context_prediction))

            return {
                'context_start': context_start,
                'context_end': context_end,
                'optimized_segment': optimized_segment,
                'success': True
            }

        except Exception as e:
            print(f"Warning: LLM重新评估失败: {e}")
            # 如果LLM调用失败，使用简单的启发式方法
            optimized_segment = self._heuristic_optimization_segment(prediction, start, end)
            return {
                'context_start': context_start,
                'context_end': context_end,
                'optimized_segment': optimized_segment,
                'success': False
            }

    def _heuristic_optimization_segment(self, prediction: List[int], start: int, end: int) -> List[int]:
        """
        启发式优化方法，为单个连续1范围生成优化后的片段
        
        Args:
            prediction: 预测序列
            start: 连续1的起始位置
            end: 连续1的结束位置
            
        Returns:
            优化后的片段
        """
        # 计算上下文范围
        context_start = max(0, start - self.window_size)
        context_end = min(len(prediction), end + self.window_size + 1)
        context_length = context_end - context_start

        # 创建优化后的片段
        optimized_segment = [0] * context_length

        # 简单策略：保留中间位置的1，其他设为0
        if end > start:
            middle = (start + end) // 2
            segment_middle = middle - context_start
            if 0 <= segment_middle < context_length:
                optimized_segment[segment_middle] = 1

        return optimized_segment

    def _build_reassessment_prompt(self, utterances: List[str], prediction: List[int],
                                   consecutive_range: Tuple[int, int], previous_context: List[str] = None,
                                   consecutive_context: List[str] = None, next_context: List[str] = None) -> str:
        """构建重新评估的提示词"""
        return self.prompt.format_prompt(
            utterances,
            prediction,
            consecutive_range,
            previous_context,
            consecutive_context,
            next_context
        )

    def _parse_llm_response(self, response: str, expected_length: int) -> List[int]:
        """解析LLM响应，提取优化后的预测序列"""
        try:
            # 尝试解析JSON响应
            json_match = re.search(r'\{[^}]*"optimized_prediction"[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                if 'optimized_prediction' in result:
                    prediction = result['optimized_prediction']
                    if len(prediction) == expected_length:
                        return prediction

            # 如果JSON解析失败，尝试从响应中提取预测序列
            # 查找类似 [0, 1, 0, 0, 1, 0] 的模式
            pattern = r'\[([0-9,\s]+)\]'
            matches = re.findall(pattern, response)

            if matches:
                # 取第一个匹配的序列
                prediction_str = matches[0]
                prediction = [int(x.strip()) for x in prediction_str.split(',')]

                # 确保长度正确
                if len(prediction) == expected_length:
                    return prediction

            # 如果解析失败，尝试其他模式
            lines = response.split('\n')
            for line in lines:
                if 'optimized_prediction' in line or '预测' in line:
                    # 查找数字序列
                    numbers = re.findall(r'\b[01]\b', line)
                    if len(numbers) == expected_length:
                        return [int(x) for x in numbers]

            # 如果都失败了，返回全0序列
            print(f"Warning: 无法解析LLM响应，使用默认值")
            return [0] * expected_length

        except Exception as e:
            print(f"Warning: 解析LLM响应时出错: {e}")
            return [0] * expected_length

    def _heuristic_optimization(self, prediction: List[int], start: int, end: int) -> List[int]:
        """
        启发式优化方法，当LLM调用失败时使用
        
        Args:
            prediction: 预测序列
            start: 连续1的起始位置
            end: 连续1的结束位置
            
        Returns:
            优化后的预测序列
        """
        optimized = prediction.copy()

        # 简单策略：保留中间位置的1，其他设为0
        if end > start:
            middle = (start + end) // 2
            for i in range(start, end + 1):
                optimized[i] = 1 if i == middle else 0

        return optimized

    def reassess_dialogue(self, utterances: List[str], prediction: List[int], num_threads: int = 8) -> Dict:
        """
        重新评估单个对话的分割预测
        
        Args:
            utterances: 对话序列
            prediction: 原始预测
            num_threads: 线程数量，默认8
            
        Returns:
            包含优化结果的字典
        """
        # 检测连续1
        consecutive_ranges = self.detect_consecutive_ones(prediction)

        if not consecutive_ranges:
            return {
                'original_prediction': prediction,
                'optimized_prediction': prediction.copy(),
                'consecutive_ranges': [],
                'changes_made': False,
                'num_changes': 0
            }

        # 使用LLM重新评估
        optimized_prediction = self.reassess_consecutive_ones(utterances, prediction, consecutive_ranges, num_threads)

        # 计算变化
        changes = sum(1 for orig, opt in zip(prediction, optimized_prediction) if orig != opt)

        return {
            'original_prediction': prediction,
            'optimized_prediction': optimized_prediction,
            'consecutive_ranges': consecutive_ranges,
            'changes_made': changes > 0,
            'num_changes': changes
        }

    def batch_reassess(self, dialogues: List[Dict], num_threads: int = 8) -> List[Dict]:
        """
        批量重新评估多个对话
        
        Args:
            dialogues: 对话列表，每个对话包含utterances和prediction
            num_threads: 线程数量，默认8
            
        Returns:
            重新评估后的对话列表
        """
        results = []

        for dialogue in dialogues:
            utterances = dialogue.get('utterances', [])
            prediction = dialogue.get('prediction', [])

            if not utterances or not prediction:
                results.append(dialogue)
                continue

            # 重新评估
            reassessment_result = self.reassess_dialogue(utterances, prediction, num_threads)

            # 更新对话结果
            updated_dialogue = dialogue.copy()
            updated_dialogue['original_prediction'] = reassessment_result['original_prediction']
            updated_dialogue['optimized_prediction'] = reassessment_result['optimized_prediction']
            updated_dialogue['consecutive_ranges'] = reassessment_result['consecutive_ranges']
            updated_dialogue['changes_made'] = reassessment_result['changes_made']
            updated_dialogue['num_changes'] = reassessment_result['num_changes']

            results.append(updated_dialogue)

        return results


def create_reassessment_agent(config_path: str = "config.yaml") -> LLMReassessmentAgent:
    """
    从配置文件创建重新评估代理
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LLMReassessmentAgent实例
    """
    config = load_config(config_path)
    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]

    return LLMReassessmentAgent(api_key, base_url, model)


# 示例使用
if __name__ == "__main__":
    # 创建重新评估代理
    agent = create_reassessment_agent()

    # 示例对话和预测
    utterances = [
        "Hello, how are you?",
        "I'm fine, thank you.",
        "What's the weather like?",
        "It's sunny today.",
        "Let's go for a walk.",
        "That sounds great!"
    ]

    # 包含连续1的预测
    prediction = [0, 1, 1, 0, 1, 0]

    print("原始预测:", prediction)
    print("对话内容:")
    for i, utt in enumerate(utterances):
        print(f"{i}: {utt}")

    # 重新评估
    result = agent.reassess_dialogue(utterances, prediction)

    print("\n重新评估结果:")
    print("优化后预测:", result['optimized_prediction'])
    print("连续1范围:", result['consecutive_ranges'])
    print("是否发生变化:", result['changes_made'])
    print("变化数量:", result['num_changes'])
