import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm_api import LLMAPI
from prompt.positive_negative import PositiveNegativePrompt
import tiktoken


class PNAgent:
    def __init__(self, dataset, api_key, base_url, model, window_size=7):
        self.dataset = dataset
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = LLMAPI(self.api_key, self.base_url, self.model)
        self.window_size = window_size
        self.prompt = PositiveNegativePrompt()

    def generate_positive_negative_samples(self, max_turns=5, num_threads=8):
        responses = []
        total_turns = self.dataset[:min(max_turns, len(self.dataset))]

        total_input_tokens = 0
        total_output_tokens = 0
        total_attempts = 0
        success_count = 0
        error_count = 0

        for turn_idx, dialogue in enumerate(
                tqdm(total_turns, desc="Generating positive/negative samples", unit="round")):
            single_dialogue = self._process_single_dialogue(dialogue, turn_idx, num_threads)
            responses.append(single_dialogue)

            for result in single_dialogue:
                if isinstance(result, dict):
                    total_input_tokens += result.get('input_tokens', 0)
                    total_output_tokens += result.get('output_tokens', 0)
                    total_attempts += result.get('attempts', 1)
                    if result.get('success', False):
                        success_count += 1
                    else:
                        error_count += 1

        print("\n" + "=" * 50)
        print("Positive/Negative Sample Generation Statistics:")
        print("=" * 50)
        print(f"Total dialogue rounds: {len(responses)}")
        print(f"Total utterances: {sum(len(dialogue) for dialogue in responses)}")
        print(f"Successful utterances: {success_count}")
        print(f"Failed utterances: {error_count}")
        print(f"Success rate: {success_count / (success_count + error_count) * 100:.2f}%" if (
                                                                                                     success_count + error_count) > 0 else "Success rate: 0%")
        print(f"Total attempts: {total_attempts}")
        print(f"Average attempts: {total_attempts / (success_count + error_count):.2f}" if (
                                                                                                   success_count + error_count) > 0 else "Average attempts: 0")
        print(f"Total input tokens: {total_input_tokens:,}")
        print(f"Total output tokens: {total_output_tokens:,}")
        print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
        print("=" * 50)

        return responses

    def _process_single_dialogue(self, dialogue, turn_idx, num_threads):
        single_dialogue = []
        dialogue_length = len(dialogue)

        tasks = [(item_idx, dialogue) for item_idx in range(dialogue_length)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {
                executor.submit(self._generate_single_response, task[0], task[1]): task[0]
                for task in tasks
            }

            results = {}

            with tqdm(total=dialogue_length, desc=f"Generating PN samples (round {turn_idx + 1})",
                      unit="utterance") as pbar:
                for future in as_completed(future_to_index):
                    item_idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[item_idx] = result
                    except Exception as exc:
                        print(f'Error generating PN samples (round {turn_idx + 1}, utterance {item_idx}): {exc}')
                        results[item_idx] = {
                            'response': f"Thread execution error: {exc}",
                            'input_tokens': 0,
                            'output_tokens': 0,
                            'attempts': 1,
                            'success': False,
                            'error': str(exc)
                        }
                    finally:
                        pbar.update(1)

        for item_idx in range(dialogue_length):
            single_dialogue.append(results[item_idx])

        return single_dialogue

    def _generate_single_response(self, item_idx, dialogue, max_retries=3):
        conversation_context = dialogue.load_index(item_idx, self.window_size)
        formatted_prompt = self.prompt.format_prompt(conversation_context)

        input_tokens = self._count_tokens(formatted_prompt)

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate_response(formatted_prompt)
                output_tokens = self._count_tokens(response)

                # 验证响应格式
                parsed_response = self._validate_and_parse_response(response)

                return {
                    'response': response,
                    'parsed_response': parsed_response,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'attempts': attempt + 1,
                    'success': True
                }
            except Exception as exc:
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    error_msg = f"Failed after {max_retries} retries: {exc}"
                    return {
                        'response': error_msg,
                        'parsed_response': None,
                        'input_tokens': input_tokens,
                        'output_tokens': 0,
                        'attempts': max_retries + 1,
                        'success': False,
                        'error': str(exc)
                    }

    def _validate_and_parse_response(self, response):
        """
        验证并解析LLM响应，确保格式正确
        """
        try:
            # 尝试提取JSON部分
            if '```json' in response:
                json_start = response.find('```json') + 7
                json_end = response.find('```', json_start)
                if json_end != -1:
                    json_str = response[json_start:json_end].strip()
                else:
                    json_str = response[json_start:].strip()
            elif '{' in response and '}' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("No valid JSON found in response")

            parsed = json.loads(json_str)

            # 验证必需字段
            if 'result' not in parsed:
                raise ValueError("Missing 'result' field")

            result = parsed['result']
            if 'positive' not in result or 'negative' not in result:
                raise ValueError("Missing 'positive' or 'negative' field")

            # 验证正样本结构
            positive = result['positive']
            if not all(key in positive for key in ['previous', 'current', 'next', 'reason']):
                raise ValueError("Positive sample missing required fields")

            # 验证负样本结构
            negative = result['negative']
            if not all(key in negative for key in ['previous', 'current', 'next', 'reason']):
                raise ValueError("Negative sample missing required fields")

            # 验证句子数量（每个样本应该有7个句子：3+1+3）
            if len(positive['previous']) != 3 or len(positive['next']) != 3:
                raise ValueError("Positive sample should have 3 previous and 3 next sentences")

            if len(negative['previous']) != 3 or len(negative['next']) != 3:
                raise ValueError("Negative sample should have 3 previous and 3 next sentences")

            return parsed

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise ValueError(f"Response validation failed: {e}")

    def _count_tokens(self, text):
        if tiktoken is None:
            return len(text) // 4

        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            try:
                encoding = tiktoken.get_encoding("p50k_base")
                return len(encoding.encode(text))
            except Exception:
                return len(text) // 4
