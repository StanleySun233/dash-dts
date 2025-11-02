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
        # Window size: number of context utterances to take on each side of consecutive range (excluding the consecutive range itself)
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
        Reassess consecutive 1s using LLM (multi-threaded version)
        
        Args:
            utterances: Dialogue sequence
            prediction: Original prediction
            consecutive_ranges: Position ranges of consecutive 1s
            num_threads: Number of threads, default 8
            
        Returns:
            Optimized prediction sequence
        """
        if not consecutive_ranges:
            return prediction.copy()

        # Create optimized prediction
        optimized_prediction = prediction.copy()

        # Process consecutive 1 ranges using multi-threading
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
                        # Use heuristic method as fallback
                        start, end = consecutive_ranges[idx]
                        results[idx] = {
                            'context_start': max(0, start - self.window_size),
                            'context_end': min(len(utterances), end + self.window_size + 1),
                            'optimized_segment': self._heuristic_optimization_segment(prediction, start, end),
                            'reason': 'Heuristic fallback due to processing error',
                            'score': 0.5,
                            'success': False
                        }
                    finally:
                        pbar.update(1)

        # Collect reason and score information for each position
        position_reason = {}  # Map position -> reason
        position_score = {}   # Map position -> score

        # Apply results in order
        for idx in range(len(consecutive_ranges)):
            if idx in results:
                result = results[idx]
                context_start = result['context_start']
                context_end = result['context_end']
                optimized_segment = result['optimized_segment']
                reason = result.get('reason', 'No reason provided')
                score = result.get('score', 0.5)

                # Update optimized prediction and store reason/score
                start, end = consecutive_ranges[idx]
                for i, val in enumerate(optimized_segment):
                    pos = context_start + i
                    if pos < len(optimized_prediction):
                        optimized_prediction[pos] = val
                        # Store reason and score for all positions in the consecutive range
                        # This explains why the reassessment was made for this range
                        if start <= pos <= end:
                            position_reason[pos] = reason
                            position_score[pos] = score

        return optimized_prediction, position_reason, position_score

    def _process_single_consecutive_range(self, utterances: List[str], prediction: List[int],
                                          start: int, end: int) -> Dict:
        # Extract relevant dialogue segment
        context_start = max(0, start - self.window_size)
        context_end = min(len(utterances), end + self.window_size + 1)
        context_utterances = utterances[context_start:context_end]
        context_prediction = prediction[context_start:context_end]

        # Build prompt
        prompt = self._build_reassessment_prompt(
            context_utterances,
            context_prediction,
            (start - context_start, end - context_start),
            utterances[max(0, context_start - self.window_size):context_start] if context_start > 0 else None,
            utterances[start:end + 1],
            utterances[end + 1:min(len(utterances), end + 1 + self.window_size)]
        )

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            # Call LLM for reassessment
            response = self.llm_api.generate_response(prompt)

            # Parse LLM response - returns None on failure, may raise exception
            parse_result = self._parse_llm_response(response, len(context_prediction))
            
            # Check if parsing was successful
            if parse_result is not None:
                optimized_segment = parse_result['prediction']
                reason = parse_result.get('reason', 'No reason provided')
                score = parse_result.get('score', 0.5)

                return {
                    'context_start': context_start,
                    'context_end': context_end,
                    'optimized_segment': optimized_segment,
                    'reason': reason,
                    'score': score,
                    'success': True
                }
            
            # If this is not the last attempt, print warning and continue
            if attempt < max_retries - 1:
                print(f"Warning: LLM response parsing failed (attempt {attempt + 1}/{max_retries}), retrying...")
        
        # All retries failed, use heuristic fallback
        print(f"Warning: LLM reassessment failed after {max_retries} attempts, using heuristic fallback")
        optimized_segment = self._heuristic_optimization_segment(prediction, start, end)
        return {
            'context_start': context_start,
            'context_end': context_end,
            'optimized_segment': optimized_segment,
            'reason': f'Heuristic fallback: Failed after {max_retries} attempts',
            'score': 0.5,
            'success': False
        }

    def _heuristic_optimization_segment(self, prediction: List[int], start: int, end: int) -> List[int]:
        """
        Heuristic optimization method to generate optimized segment for a single consecutive 1 range
        
        Args:
            prediction: Prediction sequence
            start: Start position of consecutive 1s
            end: End position of consecutive 1s
            
        Returns:
            Optimized segment
        """
        # Calculate context range
        context_start = max(0, start - self.window_size)
        context_end = min(len(prediction), end + self.window_size + 1)
        context_length = context_end - context_start

        # Create optimized segment
        optimized_segment = [0] * context_length

        # Simple strategy: keep 1 at middle position, set others to 0
        if end > start:
            middle = (start + end) // 2
            segment_middle = middle - context_start
            if 0 <= segment_middle < context_length:
                optimized_segment[segment_middle] = 1

        return optimized_segment

    def _build_reassessment_prompt(self, utterances: List[str], prediction: List[int],
                                   consecutive_range: Tuple[int, int], previous_context: List[str] = None,
                                   consecutive_context: List[str] = None, next_context: List[str] = None) -> str:
        """Build reassessment prompt"""
        return self.prompt.format_prompt(
            utterances,
            prediction,
            consecutive_range,
            previous_context,
            consecutive_context,
            next_context
        )

    def _parse_llm_response(self, response: str, expected_length: int) -> Dict:
        """Parse LLM response to extract optimized prediction sequence, reason, and score
        Only extracts JSON from ```json``` code blocks
        Returns None if parsing fails
        """
        # Extract JSON from ```json``` code block only
        json_str = None
        
        if '```json' in response:
            json_start = response.find('```json') + 7
            json_end = response.find('```', json_start)
            if json_end != -1:
                json_str = response[json_start:json_end].strip()
        
        # If no ```json block found, try regular ``` block
        if json_str is None and '```' in response:
            # Find first ``` block
            code_start = response.find('```') + 3
            code_end = response.find('```', code_start)
            if code_end != -1:
                json_str = response[code_start:code_end].strip()
        
        # If no code block found, return None
        if json_str is None:
            print(f"Warning: No JSON code block found in response")
            return None
        
        # Parse JSON - if parsing fails, return None
        result = json.loads(json_str)
        
        # Validate required fields
        if 'optimized_prediction' not in result:
            print(f"Warning: Missing 'optimized_prediction' field in JSON")
            return None
        
        prediction = result['optimized_prediction']
        if not isinstance(prediction, list):
            print(f"Warning: 'optimized_prediction' is not a list")
            return None
        
        if len(prediction) != expected_length:
            print(f"Warning: Prediction length {len(prediction)} != expected length {expected_length}")
            return None
        
        # Extract reason
        reason = result.get('reason', result.get('reasoning', 'No reason provided'))
        if not isinstance(reason, str):
            reason = str(reason)
        
        # Extract score
        score = result.get('score', result.get('confidence', 0.5))
        if not isinstance(score, (int, float)):
            score = float(score)
        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, float(score)))
        
        return {
            'prediction': prediction,
            'reason': reason,
            'score': score
        }

    def _heuristic_optimization(self, prediction: List[int], start: int, end: int) -> List[int]:
        """
        Heuristic optimization method used when LLM call fails
        
        Args:
            prediction: Prediction sequence
            start: Start position of consecutive 1s
            end: End position of consecutive 1s
            
        Returns:
            Optimized prediction sequence
        """
        optimized = prediction.copy()

        # Simple strategy: keep 1 at middle position, set others to 0
        if end > start:
            middle = (start + end) // 2
            for i in range(start, end + 1):
                optimized[i] = 1 if i == middle else 0

        return optimized

    def reassess_dialogue(self, utterances: List[str], prediction: List[int], num_threads: int = 8) -> Dict:
        """
        Reassess segmentation prediction for a single dialogue
        
        Args:
            utterances: Dialogue sequence
            prediction: Original prediction
            num_threads: Number of threads, default 8
            
        Returns:
            Dictionary containing optimization results
        """
        # Detect consecutive 1s
        consecutive_ranges = self.detect_consecutive_ones(prediction)

        if not consecutive_ranges:
            return {
                'original_prediction': prediction,
                'optimized_prediction': prediction.copy(),
                'consecutive_ranges': [],
                'changes_made': False,
                'num_changes': 0,
                'position_reason': {},  # Empty reason map
                'position_score': {}    # Empty score map
            }

        # Use LLM for reassessment
        optimized_prediction, position_reason, position_score = self.reassess_consecutive_ones(utterances, prediction, consecutive_ranges, num_threads)

        # Calculate changes
        changes = sum(1 for orig, opt in zip(prediction, optimized_prediction) if orig != opt)

        return {
            'original_prediction': prediction,
            'optimized_prediction': optimized_prediction,
            'consecutive_ranges': consecutive_ranges,
            'changes_made': changes > 0,
            'num_changes': changes,
            'position_reason': position_reason,  # Map of position -> reason
            'position_score': position_score     # Map of position -> score
        }

    def batch_reassess(self, dialogues: List[Dict], num_threads: int = 8) -> List[Dict]:
        """
        Batch reassess multiple dialogues
        
        Args:
            dialogues: List of dialogues, each containing utterances and prediction
            num_threads: Number of threads, default 8
            
        Returns:
            List of reassessed dialogues
        """
        results = []

        for dialogue in dialogues:
            utterances = dialogue.get('utterances', [])
            prediction = dialogue.get('prediction', [])

            if not utterances or not prediction:
                results.append(dialogue)
                continue

            # Reassess
            reassessment_result = self.reassess_dialogue(utterances, prediction, num_threads)

            # Update dialogue results
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
    Create reassessment agent from config file
    
    Args:
        config_path: Path to config file
        
    Returns:
        LLMReassessmentAgent instance
    """
    config = load_config(config_path)
    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]

    return LLMReassessmentAgent(api_key, base_url, model)


# Example usage
if __name__ == "__main__":
    # Create reassessment agent
    agent = create_reassessment_agent()

    # Example dialogue and prediction
    utterances = [
        "Hello, how are you?",
        "I'm fine, thank you.",
        "What's the weather like?",
        "It's sunny today.",
        "Let's go for a walk.",
        "That sounds great!"
    ]

    # Prediction containing consecutive 1s
    prediction = [0, 1, 1, 0, 1, 0]

    print("Original prediction:", prediction)
    print("Dialogue content:")
    for i, utt in enumerate(utterances):
        print(f"{i}: {utt}")

    # Reassess
    result = agent.reassess_dialogue(utterances, prediction)

    print("\nReassessment results:")
    print("Optimized prediction:", result['optimized_prediction'])
    print("Consecutive 1 ranges:", result['consecutive_ranges'])
    print("Changes made:", result['changes_made'])
    print("Number of changes:", result['num_changes'])
