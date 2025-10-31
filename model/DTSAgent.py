import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from llm_api import LLMAPI
from prompt.dts import DTSPrompt

try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken not installed, will use character count estimation for token count")
    print("Please run: pip install tiktoken")
    tiktoken = None
import time


class DTSAgent:
    def __init__(self, dataset, api_key, base_url, model, window_size=7):
        self.dataset = dataset
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.llm = LLMAPI(self.api_key, self.base_url, self.model)
        self.window_size = window_size
        self.prompt = DTSPrompt()

    def perform_dialogue_topic_segmentation(self, max_turns=5, num_threads=8, few_shot_examples=None,
                                            similarity_examples=None, handshake_results=None):
        responses = []
        total_turns = self.dataset[:min(max_turns, len(self.dataset))]

        total_input_tokens = 0
        total_output_tokens = 0
        total_attempts = 0
        success_count = 0
        error_count = 0

        for turn_idx, dialogue in enumerate(tqdm(total_turns, desc="Performing DTS", unit="round")):
            # Select few_shot and similarity for current dialogue
            current_few_shot = None
            if isinstance(few_shot_examples, list) and turn_idx < len(few_shot_examples):
                # Handle 2D list (multiple utterances per dialogue)
                if isinstance(few_shot_examples[turn_idx], list):
                    current_few_shot = few_shot_examples[turn_idx]
                else:
                    current_few_shot = few_shot_examples[turn_idx]
            elif isinstance(few_shot_examples, dict):
                current_few_shot = few_shot_examples.get(getattr(dialogue, 'dial_id', turn_idx))
            else:
                current_few_shot = few_shot_examples

            current_similarity = None
            if isinstance(similarity_examples, list) and turn_idx < len(similarity_examples):
                current_similarity = similarity_examples[turn_idx]
            elif isinstance(similarity_examples, dict):
                current_similarity = similarity_examples.get(getattr(dialogue, 'dial_id', turn_idx))
            else:
                current_similarity = similarity_examples

            single_dialogue = self._process_single_dialogue(dialogue, turn_idx, num_threads, current_few_shot,
                                                            current_similarity, handshake_results)
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
        print("Dialogue Topic Segmentation Statistics:")
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

    def _process_single_dialogue(self, dialogue, turn_idx, num_threads, few_shot_examples=None,
                                 similarity_examples=None, handshake_results=None):
        single_dialogue = []
        dialogue_length = len(dialogue)

        tasks = [(item_idx, dialogue) for item_idx in range(dialogue_length)]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_index = {
                executor.submit(self._generate_single_response, task[0], task[1], few_shot_examples,
                                similarity_examples, handshake_results): task[0]
                for task in tasks
            }

            results = {}

            with tqdm(total=dialogue_length, desc=f"Performing DTS (round {turn_idx + 1})", unit="utterance") as pbar:
                for future in as_completed(future_to_index):
                    item_idx = future_to_index[future]
                    try:
                        result = future.result()
                        results[item_idx] = result
                    except Exception as exc:
                        import traceback
                        error_details = traceback.format_exc()
                        print(f'Error performing DTS (round {turn_idx + 1}, utterance {item_idx}): {exc}')
                        print(f'Error details: {error_details}')
                        results[item_idx] = {
                            'response': f"Thread execution error: {exc}",
                            'input_tokens': 0,
                            'output_tokens': 0,
                            'attempts': 1,
                            'success': False,
                            'error': str(exc),
                            'error_details': error_details
                        }
                    finally:
                        pbar.update(1)

        for item_idx in range(dialogue_length):
            single_dialogue.append(results[item_idx])

        return single_dialogue

    def _generate_single_response(self, item_idx, dialogue, few_shot_examples=None, similarity_examples=None,
                                  handshake_results=None, max_retries=3):
        conversation_context = dialogue.load_index(item_idx, self.window_size)

        # Integrate handshake results if available
        if handshake_results:
            conversation_context = self._add_handshake_tags(conversation_context, dialogue, item_idx, handshake_results)

        # Use external few-shot and similarity examples (ablation study)
        # Select few-shot examples for current utterance
        current_few_shot = None
        if isinstance(few_shot_examples, list) and item_idx < len(few_shot_examples):
            current_few_shot = few_shot_examples[item_idx]
        else:
            current_few_shot = few_shot_examples

        current_similarity = similarity_examples

        formatted_prompt = self.prompt.format_prompt(conversation_context, current_few_shot, current_similarity)

        input_tokens = self._count_tokens(formatted_prompt)

        for attempt in range(max_retries + 1):
            try:
                response = self.llm.generate_response(formatted_prompt)
                output_tokens = self._count_tokens(response)

                # Validate response format
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
                import traceback
                error_details = traceback.format_exc()
                if attempt < max_retries:
                    time.sleep(1 * (attempt + 1))
                    continue
                else:
                    error_msg = f"Failed after {max_retries + 1} attempts: {exc}"
                    # Log detailed error for debugging
                    print(f"Error in _generate_single_response (utterance {item_idx}): {error_msg}")
                    print(f"Error details: {error_details}")
                    if hasattr(exc, 'args') and len(exc.args) > 0:
                        print(f"Exception args: {exc.args}")
                    return {
                        'response': error_msg,
                        'parsed_response': None,
                        'input_tokens': input_tokens,
                        'output_tokens': 0,
                        'attempts': max_retries + 1,
                        'success': False,
                        'error': str(exc),
                        'error_details': error_details
                    }

    def _validate_and_parse_response(self, response):
        try:
            # Extract JSON part
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

            # Validate required fields
            if 'result' not in parsed:
                raise ValueError("Missing 'result' field")

            if 'score' not in parsed:
                raise ValueError("Missing 'score' field")

            if 'reason' not in parsed:
                raise ValueError("Missing 'reason' field")

            # Validate result value
            if parsed['result'] not in ['SEGMENT', 'NO_SEGMENT']:
                raise ValueError("Invalid result value, must be 'SEGMENT' or 'NO_SEGMENT'")

            # Validate score value
            score = parsed['score']
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                raise ValueError("Score must be a number between 0.0 and 1.0")

            return parsed

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {e}. Response was: {response[:200]}..."
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Response validation failed: {e}. Response was: {response[:200]}..."
            raise ValueError(error_msg)

    def _add_handshake_tags(self, conversation_context, dialogue, current_idx, handshake_results):
        # Get handshake results for current dialogue
        handshake_dialogue = None
        
        if handshake_results is None:
            return conversation_context
        
        # Handle both list and dictionary formats
        if isinstance(handshake_results, dict):
            # Dictionary format: key is dial_id
            handshake_dialogue = handshake_results.get(dialogue.dial_id)
        elif isinstance(handshake_results, list):
            # List format: index corresponds to dialogue index in dataset
            dialogue_idx = None
            for i, d in enumerate(self.dataset):
                if d.dial_id == dialogue.dial_id:
                    dialogue_idx = i
                    break
            
            if dialogue_idx is None or dialogue_idx >= len(handshake_results):
                return conversation_context
            
            handshake_dialogue = handshake_results[dialogue_idx]
        else:
            return conversation_context
        
        # Check if handshake_dialogue is valid
        if handshake_dialogue is None or (isinstance(handshake_dialogue, list) and len(handshake_dialogue) == 0):
            return conversation_context
        
        # Ensure handshake_dialogue is a list for index-based access
        if not isinstance(handshake_dialogue, list):
            return conversation_context

        # Add handshake tags to previous utterances
        tagged_previous = []
        # Count padding tokens at the beginning
        padding_count = 0
        for utt in conversation_context["previous"]:
            if utt in ["<dialogue_start>", "<dialogue_end>"]:
                padding_count += 1
            else:
                break
        
        # Calculate starting index of actual utterances (excluding padding)
        actual_start_idx = current_idx - (len(conversation_context["previous"]) - padding_count)
        
        for i, utterance in enumerate(conversation_context["previous"]):
            if utterance in ["<dialogue_start>", "<dialogue_end>"]:
                tagged_previous.append(utterance)
            else:
                # Calculate actual index in dialogue (skip padding)
                actual_idx = actual_start_idx + (i - padding_count)
                try:
                    if isinstance(handshake_dialogue, list) and 0 <= actual_idx < len(handshake_dialogue):
                        handshake_result = handshake_dialogue[actual_idx]
                        if isinstance(handshake_result, dict) and handshake_result.get('success', False):
                            parsed = handshake_result.get('parsed_response', {})
                            if parsed and 'result' in parsed:
                                tag = parsed['result']
                                tagged_utterance = f"[{tag}] {utterance}"
                            else:
                                tagged_utterance = f"[O] {utterance}"
                        else:
                            tagged_utterance = f"[O] {utterance}"
                    else:
                        tagged_utterance = f"[O] {utterance}"
                except (IndexError, KeyError, TypeError) as e:
                    # Fallback to default tag if index access fails
                    tagged_utterance = f"[O] {utterance}"
                tagged_previous.append(tagged_utterance)

        # Add handshake tag to current utterance
        try:
            if isinstance(handshake_dialogue, list) and 0 <= current_idx < len(handshake_dialogue):
                handshake_result = handshake_dialogue[current_idx]
                if isinstance(handshake_result, dict) and handshake_result.get('success', False):
                    parsed = handshake_result.get('parsed_response', {})
                    if parsed and 'result' in parsed:
                        tag = parsed['result']
                        tagged_current = f"[{tag}] {conversation_context['current']}"
                    else:
                        tagged_current = f"[O] {conversation_context['current']}"
                else:
                    tagged_current = f"[O] {conversation_context['current']}"
            else:
                tagged_current = f"[O] {conversation_context['current']}"
        except (IndexError, KeyError, TypeError) as e:
            # Fallback to default tag if index access fails
            tagged_current = f"[O] {conversation_context['current']}"

        # Add handshake tags to next utterances
        tagged_next = []
        for i, utterance in enumerate(conversation_context["next"]):
            if utterance in ["<dialogue_start>", "<dialogue_end>"]:
                tagged_next.append(utterance)
            else:
                # Calculate actual index in dialogue (skip padding)
                actual_idx = current_idx + 1 + i
                try:
                    if isinstance(handshake_dialogue, list) and 0 <= actual_idx < len(handshake_dialogue):
                        handshake_result = handshake_dialogue[actual_idx]
                        if isinstance(handshake_result, dict) and handshake_result.get('success', False):
                            parsed = handshake_result.get('parsed_response', {})
                            if parsed and 'result' in parsed:
                                tag = parsed['result']
                                tagged_utterance = f"[{tag}] {utterance}"
                            else:
                                tagged_utterance = f"[O] {utterance}"
                        else:
                            tagged_utterance = f"[O] {utterance}"
                    else:
                        tagged_utterance = f"[O] {utterance}"
                except (IndexError, KeyError, TypeError) as e:
                    # Fallback to default tag if index access fails
                    tagged_utterance = f"[O] {utterance}"
                tagged_next.append(tagged_utterance)

        return {
            "previous": tagged_previous,
            "current": tagged_current,
            "next": tagged_next
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
