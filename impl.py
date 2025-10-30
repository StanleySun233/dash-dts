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
    for dialogue in dataset:
        if dialogue.dial_id == dial_id:
            target_dialogue = dialogue
            break

    if target_dialogue is None:
        return {"error": f"Dialogue {dial_id} not found"}, 404

    prediction_results = dts_agent.perform_dialogue_topic_segmentation(
        max_turns=1,
        num_threads=8,
        handshake_results=handshake_results,
        few_shot_examples=few_shot_examples,
        similarity_examples=similarity_examples
    )

    if not prediction_results or len(prediction_results) == 0:
        return {"error": "Failed to generate predictions"}, 500

    predictions = prediction_results[0]
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
    for i, (original, optimized) in enumerate(zip(prediction, optimized_prediction)):
        if original != optimized:
            reason = f"Reassessed from {original} to {optimized} based on context analysis"
            score = 0.8 if optimized == 1 else 0.2
        else:
            reason = "No change needed"
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


