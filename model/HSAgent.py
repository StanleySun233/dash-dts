from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm_api import LLMAPI
from prompt.handshake import HandShakePrompt

try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not installed, will use character count estimation for token count")
    print("Please run: pip install tiktoken")
    tiktoken = None
import time


class HSAgent:
    def __init__(self, dataset, api_key, base_url, model, window_size=3):
        self.dataset = dataset
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = LLMAPI(self.api_key, self.base_url, self.model)
        self.window_size = window_size
        self.prompt = HandShakePrompt()

    def generate_handshake(self, max_turns=5, num_threads=8):
        responses = []
        total_turns = self.dataset[:min(max_turns, len(self.dataset))]

        total_input_tokens = 0
        total_output_tokens = 0
        total_attempts = 0
        success_count = 0
        error_count = 0

        for turn_idx, dialogue in enumerate(tqdm(total_turns, desc="Generating handshake dialogue", unit="round")):
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
        print("Execution completion statistics:")
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

            with tqdm(total=dialogue_length, desc=f"Generating handshake utterances (round {turn_idx + 1})",
                      unit="utterance") as pbar:
                for future in as_completed(future_to_index):
                    item_idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[item_idx] = result
                    except Exception as exc:
                        print(
                            f'Error generating handshake utterance (round {turn_idx + 1}, utterance {item_idx}): {exc}')
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

                return {
                    'response': response,
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
                        'input_tokens': input_tokens,
                        'output_tokens': 0,
                        'attempts': max_retries + 1,
                        'success': False,
                        'error': str(exc)
                    }

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
