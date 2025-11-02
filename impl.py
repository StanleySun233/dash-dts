from flask import jsonify
from tqdm import tqdm

from dialogue_dataset import Utterance
from metrics import evaluate_segmentation
from utils import (
    parse_llm_response,
    convert_numpy_types,
    convert_segments_to_boundary,
    convert_predictions_to_boundary,
)


def list_dialogues(dataset):
    dialogues = []
    for i, dialogue in enumerate(dataset):
        dialogues.append({
            'dial_id': dialogue.dial_id,
            'num_utterances': len(dialogue.utterances),
            'index': i
        })
    return {"success": True, "dialogues": dialogues}


def get_dialogue_info(dataset, dial_id):
    target_dialogue = None
    for dialogue in dataset:
        if dialogue.dial_id == dial_id:
            target_dialogue = dialogue
            break

    if target_dialogue is None:
        return None, 404

    reference = convert_segments_to_boundary(target_dialogue.segments, len(target_dialogue.utterances))
    return {
        "success": True,
        "dial_id": target_dialogue.dial_id,
        "utterances": target_dialogue.utterances,
        "reference": reference
    }, 200


def run_handshake(hs_agent, data):
    test_utterances = data["previous"] + [data["current"]] + data["next"]
    test_utterance = Utterance(
        dial_id="test_handshake",
        utterances=test_utterances,
        segments=[len(test_utterances)],
        utt_lst=list(range(len(test_utterances)))
    )

    current_idx = len(data["previous"])
    result = hs_agent._generate_single_response(current_idx, test_utterance)

    if result.get('success', False):
        response_text = result['response']
        classifications = parse_llm_response(response_text)
        return {"result": classifications, "confidence": result.get('output_tokens', 0), "success": True}, 200
    else:
        return {"error": result.get('error', 'Handshake detection failed')}, 500


def run_posneg(pn_agent, data):
    test_utterances = data["dialogue"]
    test_utterance = Utterance(
        dial_id="test_posneg",
        utterances=test_utterances,
        segments=[len(test_utterances)],
        utt_lst=list(range(len(test_utterances)))
    )

    current_idx = len(test_utterances) // 2
    result = pn_agent._generate_single_response(current_idx, test_utterance)

    if result.get('success', False):
        parsed_response = result.get('parsed_response')
        if parsed_response and 'result' in parsed_response:
            return {"result": parsed_response["result"], "success": True}, 200
        else:
            response_text = result['response']
            parsed_result = parse_llm_response(response_text)
            return {"result": parsed_result, "success": True}, 200
    else:
        return {"error": result.get('error', 'Positive/Negative sample generation failed')}, 500


def run_dts(dts_agent, data):
    test_utterances = data["previous"] + [data["current"]] + data["next"]
    test_utterance = Utterance(
        dial_id="test_dts",
        utterances=test_utterances,
        segments=[len(test_utterances)],
        utt_lst=list(range(len(test_utterances)))
    )

    current_idx = len(data["previous"])
    result = dts_agent._generate_single_response(current_idx, test_utterance)

    if result.get('success', False):
        parsed_response = result.get('parsed_response')
        if parsed_response:
            return {
                "result": parsed_response["result"],
                "score": parsed_response["score"],
                "reason": parsed_response["reason"],
                "success": True
            }, 200
        else:
            response_text = result['response']
            parsed_result = parse_llm_response(response_text)
            return {"result": parsed_result, "success": True}, 200
    else:
        return {"error": result.get('error', 'Dialogue topic segmentation failed')}, 500


def run_similarity(ds_agent, dialogue):
    print("Loading existing segment embeddings...")
    if not ds_agent.load_segment_embeddings():
        print("Generating new segment embeddings...")
        ds_agent.generate_segment_embeddings()
    else:
        print("Loaded existing segment embeddings")

    target_dialogue = Utterance(
        dial_id="query_dialogue",
        utterances=dialogue,
        segments=[len(dialogue)],
        utt_lst=list(range(len(dialogue)))
    )

    similarity_results = []
    window_size = 3
    total_utts = len(target_dialogue.utterances)
    with tqdm(total=total_utts, desc="Transformer similarity", unit="utt") as pbar:
        for utterance_idx in range(total_utts):
            try:
                context = target_dialogue.load_index(utterance_idx, window_size)

                temp_utterance = Utterance(
                    dial_id="temp_query",
                    utterances=context["previous"] + [context["current"]] + context["next"],
                    segments=[len(context["previous"]) + 1 + len(context["next"])],
                    utt_lst=list(range(len(context["previous"]) + 1 + len(context["next"])) )
                )

                temp_segments = temp_utterance.get_segments()
                temp_utterance.segment_embeddings = []
                for segment in temp_segments:
                    if segment:
                        segment_embedding = ds_agent.model.encode(segment)
                        temp_utterance.segment_embeddings.append(segment_embedding)
                    else:
                        temp_utterance.segment_embeddings.append(None)

                most_similar_utterance, most_similar_segment_id, similarity_score = ds_agent.find_most_similar_segment(
                    temp_utterance, 0)

                if most_similar_utterance is not None:
                    similar_segment = most_similar_utterance.get_segments()[most_similar_segment_id]
                    similarity_results.append({
                        "utterance_idx": utterance_idx,
                        "context": context,
                        "similar_dialogue": similar_segment,
                        "similarity_score": float(similarity_score),
                        "dial_id": most_similar_utterance.dial_id
                    })
                else:
                    similarity_results.append({
                        "utterance_idx": utterance_idx,
                        "context": context,
                        "similar_dialogue": None,
                        "similarity_score": 0.0,
                        "dial_id": None
                    })
            except Exception as e:
                print(f"Error processing utterance {utterance_idx}: {e}")
                similarity_results.append({
                    "utterance_idx": utterance_idx,
                    "context": target_dialogue.load_index(utterance_idx, window_size),
                    "similar_dialogue": None,
                    "similarity_score": 0.0,
                    "dial_id": None,
                    "error": str(e)
                })
            finally:
                pbar.update(1)

    return {"success": True, "total_utterances": len(target_dialogue.utterances), "similarity_results": similarity_results}, 200


def compute_metrics(reference, hypothesis):
    results = evaluate_segmentation(reference, hypothesis)
    return {
        "success": True,
        "metrics": convert_numpy_types(results),
        "data_info": {
            "sequence_length": len(reference),
            "reference_segments": sum(reference),
            "hypothesis_segments": sum(hypothesis)
        }
    }, 200


def run_reassess(reassessment_agent, content, prediction):
    result = reassessment_agent.reassess_dialogue(content, prediction, num_threads=8)
    return {
        "original_prediction": result['original_prediction'],
        "optimized_prediction": result['optimized_prediction']
    }, 200


def run_segment(dataset, dts_agent, dial_id, handshake_results, few_shot_examples, similarity_examples):
    target_dialogue = None
    target_dialogue_idx = None
    for idx, dialogue in enumerate(dataset):
        if dialogue.dial_id == dial_id:
            target_dialogue = dialogue
            target_dialogue_idx = idx
            break

    if target_dialogue is None:
        return {"error": f"Dialogue {dial_id} not found"}, 404

    # Extract examples for this specific dialogue
    # Handle both single-dialogue format (from API) and multi-dialogue format (from evaluation)
    current_handshake = None
    if handshake_results:
        if isinstance(handshake_results, dict):
            # Dictionary format: key is dial_id, value is list of utterance results
            current_handshake = handshake_results  # Already in dict format
        elif isinstance(handshake_results, list):
            # Check if this is a single dialogue result (list of utterance results)
            # or multi-dialogue format (list of dialogues, each containing utterance results)
            if len(handshake_results) > 0:
                # If first element is a dict with 'success' or 'parsed_response', it's single dialogue format
                if isinstance(handshake_results[0], dict) and ('success' in handshake_results[0] or 'parsed_response' in handshake_results[0]):
                    # Single dialogue format: convert to dict with dial_id as key
                    current_handshake = {dial_id: handshake_results}
                elif target_dialogue_idx is not None and target_dialogue_idx < len(handshake_results):
                    # Multi-dialogue format: extract specific dialogue
                    current_handshake = {dial_id: handshake_results[target_dialogue_idx]}
                else:
                    # Try to use as-is (might be single dialogue format)
                    current_handshake = {dial_id: handshake_results}
        else:
            current_handshake = handshake_results

    current_few_shot = None
    if few_shot_examples:
        if isinstance(few_shot_examples, dict):
            current_few_shot = few_shot_examples  # Already in dict format
        elif isinstance(few_shot_examples, list):
            # Check if this is a single dialogue result (list of utterance strings)
            # or multi-dialogue format (list of dialogues)
            if len(few_shot_examples) > 0:
                # If first element is a string or None, it's single dialogue format (list of utterance examples)
                if isinstance(few_shot_examples[0], (str, type(None))):
                    # Single dialogue format: use directly (it's already per-utterance)
                    current_few_shot = few_shot_examples
                elif target_dialogue_idx is not None and target_dialogue_idx < len(few_shot_examples):
                    # Multi-dialogue format: extract specific dialogue
                    current_few_shot = few_shot_examples[target_dialogue_idx]
                else:
                    # Try to use as-is (might be single dialogue format)
                    current_few_shot = few_shot_examples
        else:
            current_few_shot = few_shot_examples

    current_similarity = None
    if similarity_examples:
        if isinstance(similarity_examples, dict):
            current_similarity = similarity_examples  # Already in dict format
        elif isinstance(similarity_examples, list):
            # Check if this is a single dialogue result or multi-dialogue format
            if len(similarity_examples) > 0:
                # If first element is a string, it's likely single dialogue format
                if isinstance(similarity_examples[0], str):
                    # Single dialogue format: might be a string representation
                    current_similarity = similarity_examples[0] if len(similarity_examples) == 1 else str(similarity_examples)
                elif target_dialogue_idx is not None and target_dialogue_idx < len(similarity_examples):
                    # Multi-dialogue format: extract specific dialogue
                    current_similarity = similarity_examples[target_dialogue_idx]
                else:
                    # Try to use as-is
                    current_similarity = str(similarity_examples) if not isinstance(similarity_examples, str) else similarity_examples
        else:
            # String or other format: use directly
            current_similarity = similarity_examples

    # Process the single dialogue directly using _process_single_dialogue
    try:
        predictions = dts_agent._process_single_dialogue(
            target_dialogue,
            turn_idx=0,
            num_threads=8,
            few_shot_examples=current_few_shot,
            similarity_examples=current_similarity,
            handshake_results=current_handshake
        )
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in run_segment._process_single_dialogue: {e}")
        print(f"Error details: {error_details}")
        return {"error": f"Failed to process dialogue: {str(e)}", "details": error_details}, 500

    if not predictions or len(predictions) == 0:
        return {"error": "Failed to generate predictions"}, 500

    prediction_boundary = convert_predictions_to_boundary(predictions, len(target_dialogue.utterances))

    prediction_details = []
    for i, pred in enumerate(predictions):
        if isinstance(pred, dict) and pred.get('success', False):
            parsed = pred.get('parsed_response', {})
            if parsed:
                prediction_details.append({
                    "boundary": prediction_boundary[i],
                    "score": parsed.get('score', 0.0),
                    "reason": parsed.get('reason', 'No reason provided')
                })
            else:
                prediction_details.append({
                    "boundary": prediction_boundary[i],
                    "score": 0.0,
                    "reason": "Failed to parse response"
                })
        else:
            prediction_details.append({
                "boundary": prediction_boundary[i],
                "score": 0.0,
                "reason": "Prediction failed"
            })

    reference = convert_segments_to_boundary(target_dialogue.segments, len(target_dialogue.utterances))
    metrics = evaluate_segmentation(reference, prediction_boundary)

    return {
        "success": True,
        "prediction": prediction_boundary,
        "prediction_details": prediction_details,
        "metrics": convert_numpy_types(metrics)
    }, 200


def run_reassess_by_id(dataset, reassessment_agent, dial_id, prediction):
    target_dialogue = None
    for dialogue in dataset:
        if dialogue.dial_id == dial_id:
            target_dialogue = dialogue
            break

    if target_dialogue is None:
        return {"error": f"Dialogue {dial_id} not found"}, 404

    result = reassessment_agent.reassess_dialogue(target_dialogue.utterances, prediction, num_threads=8)

    reassess_details = []
    optimized_prediction = result['optimized_prediction']
    position_reason = result.get('position_reason', {})  # Get reason from LLM
    position_score = result.get('position_score', {})    # Get score from LLM
    
    for i, (original, optimized) in enumerate(zip(prediction, optimized_prediction)):
        # Use LLM reason and score if available for this position, otherwise use fallback
        if i in position_reason:
            reason = position_reason[i]
            score = position_score.get(i, 0.5)  # Use real score from LLM
        elif original != optimized:
            reason = f"Reassessed from {original} to {optimized} based on context analysis"
            score = 0.8 if optimized == 1 else 0.2
        else:
            reason = "No change needed - position was not in consecutive range requiring reassessment"
            score = 0.5
        reassess_details.append({
            "boundary": optimized,
            "score": score,
            "reason": reason
        })

    reference = convert_segments_to_boundary(target_dialogue.segments, len(target_dialogue.utterances))
    optimized_metrics = evaluate_segmentation(reference, result['optimized_prediction'])

    return {
        "success": True,
        "optimized_prediction": result['optimized_prediction'],
        "reassess_details": reassess_details,
        "changes_made": result.get('changes_made', False),
        "num_changes": result.get('num_changes', 0),
        "metrics": convert_numpy_types(optimized_metrics)
    }, 200


