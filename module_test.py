import random

from dialogue_dataset import DialogueDataset
from model.DSAgent import DSAgent
from model.DTSAgent import DTSAgent
from model.HSAgent import HSAgent
from model.PNAgent import PNAgent
from utils import load_config, resolve_dataset_path


def test_hs_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2):
    print("=== Testing HSAgent (Handshake Detection) ===")

    hs_agent = HSAgent(dataset, api_key, base_url, model, window_size=3)

    try:
        print(f"Processing {max_turns} dialogue turns with {num_threads} threads...")
        results = hs_agent.generate_handshake(max_turns=max_turns, num_threads=num_threads)

        print(f"\nHSAgent Results Summary:")
        print(f"Total dialogue rounds processed: {len(results)}")

        # Analyze results
        total_utterances = 0
        successful_utterances = 0
        for dialogue_results in results:
            for utterance_result in dialogue_results:
                total_utterances += 1
                if isinstance(utterance_result, dict) and utterance_result.get('success', False):
                    successful_utterances += 1

        print(f"Total utterances: {total_utterances}")
        print(f"Successful utterances: {successful_utterances}")
        print(
            f"Success rate: {successful_utterances / total_utterances * 100:.2f}%" if total_utterances > 0 else "Success rate: 0%")

        return results

    except Exception as e:
        print(f"❌ HSAgent test failed: {e}")
        return None


def test_pn_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2):
    print("=== Testing PNAgent (Positive/Negative Sample Generation) ===")

    pn_agent = PNAgent(dataset, api_key, base_url, model, window_size=7)

    try:
        print(f"Processing {max_turns} dialogue turns with {num_threads} threads...")
        results = pn_agent.generate_positive_negative_samples(max_turns=max_turns, num_threads=num_threads)

        print(f"\nPNAgent Results Summary:")
        print(f"Total dialogue rounds processed: {len(results)}")

        # Analyze results
        total_utterances = 0
        successful_utterances = 0
        valid_samples = 0

        for dialogue_results in results:
            for utterance_result in dialogue_results:
                total_utterances += 1
                if isinstance(utterance_result, dict) and utterance_result.get('success', False):
                    successful_utterances += 1
                    # Check for valid parsed results
                    if utterance_result.get('parsed_response'):
                        valid_samples += 1

        print(f"Total utterances: {total_utterances}")
        print(f"Successful utterances: {successful_utterances}")
        print(f"Valid samples generated: {valid_samples}")
        print(
            f"Success rate: {successful_utterances / total_utterances * 100:.2f}%" if total_utterances > 0 else "Success rate: 0%")
        print(
            f"Valid sample rate: {valid_samples / successful_utterances * 100:.2f}%" if successful_utterances > 0 else "Valid sample rate: 0%")

        return results

    except Exception as e:
        print(f"❌ PNAgent test failed: {e}")
        return None


def test_dts_agent(dataset, api_key, base_url, model, ds_agent=None, max_turns=2, num_threads=2):
    print("=== Testing DTSAgent (Dialogue Topic Segmentation) ===")

    dts_agent = DTSAgent(dataset, api_key, base_url, model, window_size=7)

    # Prepare few-shot and similarity examples
    few_shot_examples = None
    similarity_examples = None

    if ds_agent:
        try:
            # Generate few-shot examples
            random_idx = random.randint(0, len(dataset) - 1)
            item = dataset[random_idx]
            conversation_context = item.load_index(8, 3)

            # Use PNAgent to generate few-shot examples
            pn_agent = PNAgent(dataset, api_key, base_url, model, window_size=7)
            # Simplified: use dialogue context directly as example
            few_shot_examples = str(conversation_context)
            print(f"Generated few-shot examples: {len(few_shot_examples)} characters")

            # Generate similarity examples
            if not hasattr(dataset[0], 'segment_embeddings') or dataset[0].segment_embeddings is None or len(
                    dataset[0].segment_embeddings) == 0:
                ds_agent.generate_segment_embeddings()
            else:
                ds_agent.load_segment_embeddings()

            target_utterance = dataset[random_idx]
            similarity_results = ds_agent.find_most_similar_segments_for_all(target_utterance)
            similarity_examples = similarity_results
            print(f"Generated similarity examples: {len(similarity_results)} segment examples")

        except Exception as e:
            print(f"Warning: Error generating examples: {e}")

    try:
        print(f"Processing {max_turns} dialogue turns with {num_threads} threads...")
        results = dts_agent.perform_dialogue_topic_segmentation(
            max_turns=max_turns,
            num_threads=num_threads,
            few_shot_examples=few_shot_examples,
            similarity_examples=similarity_examples
        )

        print(f"\nDTSAgent Results Summary:")
        print(f"Total dialogue rounds processed: {len(results)}")

        # Analyze results
        total_utterances = 0
        successful_utterances = 0
        valid_segmentations = 0
        segment_decisions = {'SEGMENT': 0, 'NO_SEGMENT': 0}

        for dialogue_results in results:
            for utterance_result in dialogue_results:
                total_utterances += 1
                if isinstance(utterance_result, dict) and utterance_result.get('success', False):
                    successful_utterances += 1
                    # Check for valid parsed results
                    parsed_response = utterance_result.get('parsed_response')
                    if parsed_response and 'result' in parsed_response:
                        valid_segmentations += 1
                        decision = parsed_response['result']
                        if decision in segment_decisions:
                            segment_decisions[decision] += 1

        print(f"Total utterances: {total_utterances}")
        print(f"Successful utterances: {successful_utterances}")
        print(f"Valid segmentations: {valid_segmentations}")
        print(
            f"Success rate: {successful_utterances / total_utterances * 100:.2f}%" if total_utterances > 0 else "Success rate: 0%")
        print(
            f"Valid segmentation rate: {valid_segmentations / successful_utterances * 100:.2f}%" if successful_utterances > 0 else "Valid segmentation rate: 0%")
        print(f"Segment decisions: {segment_decisions}")

        return results

    except Exception as e:
        print(f"❌ DTSAgent test failed: {e}")
        return None


def test_segments_cutting(dataset, test_count=3):
    print("=== Testing segments cutting functionality ===")

    test_count = min(test_count, len(dataset))
    random_indices = random.sample(range(len(dataset)), test_count)

    results = []

    for i, idx in enumerate(random_indices):
        utterance = dataset[idx]
        print(f"\n--- Dialogue {i + 1} (dial_id: {utterance.dial_id}) ---")

        print(f"Original dialogue length: {len(utterance.utterances)}")
        print(f"Segments: {utterance.segments}")
        print(f"Utt_lst: {utterance.utt_list[:20]}..." if len(
            utterance.utt_list) > 20 else f"Utt_lst: {utterance.utt_list}")

        segments = utterance.get_segments()
        segment_info = utterance.get_segment_info()

        print(f"Number of segments: {len(segments)}")
        segment_details = []

        for j, (segment, info) in enumerate(zip(segments, segment_info)):
            print(f"  Segment {j}:")
            print(f"    Length: {len(segment)}")
            print(f"    Range: [{info['start_idx']}:{info['end_idx']}]")
            print(f"    Content: {segment}")
            print()

            segment_details.append({
                'segment_id': j,
                'length': len(segment),
                'start_idx': info['start_idx'],
                'end_idx': info['end_idx'],
                'content': segment
            })

        results.append({
            'dial_id': utterance.dial_id,
            'original_length': len(utterance.utterances),
            'segments': utterance.segments,
            'segment_count': len(segments),
            'segment_details': segment_details
        })

        print("=" * 60)

    return results


def test_segment_similarity(dataset, ds_agent=None, test_count=3):
    print("\n=== Testing segment similarity matching functionality ===")

    if ds_agent is None:
        ds_agent = DSAgent(dataset)
        if not hasattr(dataset[0], 'segment_embeddings') or dataset[0].segment_embeddings is None or len(
                dataset[0].segment_embeddings) == 0:
            print("Generating segment embeddings...")
            ds_agent.generate_segment_embeddings()
        else:
            print("Loading existing segment embeddings...")
            ds_agent.load_segment_embeddings()
    else:
        print("Using provided DSAgent with existing embeddings...")

    test_count = min(test_count, len(dataset))
    random_indices = random.sample(range(len(dataset)), test_count)

    results = []

    for i, idx in enumerate(random_indices):
        target_utterance = dataset[idx]
        print(f"\n--- Dialogue {i + 1} (dial_id: {target_utterance.dial_id}) ---")

        print(f"Original dialogue length: {len(target_utterance.utterances)}")
        print(f"Segments: {target_utterance.segments}")

        segments = target_utterance.get_segments()
        segment_info = target_utterance.get_segment_info()

        print(f"Number of segments: {len(segments)}")
        for j, (segment, info) in enumerate(zip(segments, segment_info)):
            print(f"  Segment {j}: length={len(segment)}, range=[{info['start_idx']}:{info['end_idx']}]")
            print(f"    Content: {segment[:2]}..." if len(segment) > 2 else f"    Content: {segment}")

        try:
            print(f"\n--- Segment similarity matching results ---")
            segment_results = ds_agent.find_most_similar_segments_for_all(target_utterance)

            segment_matches = []
            for result in segment_results:
                seg_id = result['segment_id']
                segment = result['segment']
                similar_utterance = result['most_similar_utterance']
                similar_seg_id = result['most_similar_segment_id']
                similarity_score = result['similarity_score']

                print(f"\nSegment {seg_id}:")
                print(f"  Content: {segment[:2]}..." if len(segment) > 2 else f"  Content: {segment}")

                if similar_utterance is not None:
                    similar_segment = similar_utterance.get_segments()[similar_seg_id]
                    print(f"  Most similar dialogue: dial_id={similar_utterance.dial_id}, segment={similar_seg_id}")
                    print(f"  Similarity score: {similarity_score:.4f}")
                    print(f"  Similar segment content: {similar_segment[:2]}..." if len(
                        similar_segment) > 2 else f"  Similar segment content: {similar_segment}")

                    segment_matches.append({
                        'segment_id': seg_id,
                        'segment_content': segment,
                        'similar_dial_id': similar_utterance.dial_id,
                        'similar_segment_id': similar_seg_id,
                        'similar_segment_content': similar_segment,
                        'similarity_score': similarity_score
                    })
                else:
                    print(f"  No similar segment found")
                    segment_matches.append({
                        'segment_id': seg_id,
                        'segment_content': segment,
                        'similar_dial_id': None,
                        'similar_segment_id': None,
                        'similar_segment_content': None,
                        'similarity_score': 0.0
                    })

        except Exception as e:
            print(f"Error: {e}")
            segment_matches = []

        results.append({
            'dial_id': target_utterance.dial_id,
            'original_length': len(target_utterance.utterances),
            'segments': target_utterance.segments,
            'segment_count': len(segments),
            'segment_matches': segment_matches
        })

        print("\n" + "=" * 80)

    return results


def test_single_segment_matching(dataset, ds_agent=None):
    print("\n=== Testing single segment matching functionality ===")

    if ds_agent is None:
        ds_agent = DSAgent(dataset)
        if not hasattr(dataset[0], 'segment_embeddings') or dataset[0].segment_embeddings is None or len(
                dataset[0].segment_embeddings) == 0:
            ds_agent.generate_segment_embeddings()
        else:
            ds_agent.load_segment_embeddings()
    else:
        print("Using provided DSAgent with existing embeddings...")

    target_utterance = dataset[0]
    segments = target_utterance.get_segments()

    if len(segments) > 0:
        print(f"Target dialogue: dial_id={target_utterance.dial_id}")
        print(f"Target segment 0: {segments[0]}")

        try:
            most_similar_utterance, most_similar_segment_id, similarity_score = ds_agent.find_most_similar_segment(
                target_utterance, target_segment_id=0
            )

            if most_similar_utterance is not None:
                similar_segment = most_similar_utterance.get_segments()[most_similar_segment_id]
                print(f"Most similar dialogue: dial_id={most_similar_utterance.dial_id}")
                print(f"Most similar segment {most_similar_segment_id}: {similar_segment}")
                print(f"Similarity score: {similarity_score:.4f}")

                return {
                    'target_dial_id': target_utterance.dial_id,
                    'target_segment': segments[0],
                    'similar_dial_id': most_similar_utterance.dial_id,
                    'similar_segment_id': most_similar_segment_id,
                    'similar_segment': similar_segment,
                    'similarity_score': similarity_score
                }
            else:
                print("No similar segment found")
                return None

        except Exception as e:
            print(f"Error: {e}")
            return None
    else:
        print("Target dialogue has no segments")
        return None


def run_all_tests(dataset_name_or_path='vfh'):
    config = load_config("config.yaml")
    dataset = DialogueDataset(resolve_dataset_path(dataset_name_or_path))

    # Segment-related tests
    segments_cutting_results = test_segments_cutting(dataset, test_count=3)

    # Create DSAgent for similarity matching and ensure embeddings are generated
    ds_agent = DSAgent(dataset)

    # Try to load embeddings first, generate if loading fails
    print("Attempting to load existing segment embeddings...")
    load_success = ds_agent.load_segment_embeddings()

    if not load_success:
        print("No existing embeddings found, generating new ones...")
        ds_agent.generate_segment_embeddings()
    else:
        print("Successfully loaded existing segment embeddings!")

    # Use the same ds_agent for segment tests
    segment_similarity_results = test_segment_similarity(dataset, ds_agent, test_count=3)
    single_segment_result = test_single_segment_matching(dataset, ds_agent)

    # Agent tests
    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]

    # Run each Agent test
    hs_agent_result = test_hs_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2)
    pn_agent_result = test_pn_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2)
    dts_agent_result = test_dts_agent(dataset, api_key, base_url, model, ds_agent, max_turns=2, num_threads=2)

    return {
        'segments_cutting': segments_cutting_results,
        'segment_similarity': segment_similarity_results,
        'single_segment_matching': single_segment_result,
        'hs_agent': hs_agent_result,
        'pn_agent': pn_agent_result,
        'dts_agent': dts_agent_result
    }


def run_segment_tests(dataset_name_or_path='vfh'):
    print("=== Running segment-related tests ===")
    dataset = DialogueDataset(resolve_dataset_path(dataset_name_or_path))

    segments_cutting_results = test_segments_cutting(dataset, test_count=3)

    # Create DSAgent for similarity matching and ensure embeddings are generated
    ds_agent = DSAgent(dataset)

    # Try to load embeddings first, generate if loading fails
    print("Attempting to load existing segment embeddings...")
    load_success = ds_agent.load_segment_embeddings()

    if not load_success:
        print("No existing embeddings found, generating new ones...")
        ds_agent.generate_segment_embeddings()
    else:
        print("Successfully loaded existing segment embeddings!")

    segment_similarity_results = test_segment_similarity(dataset, ds_agent, test_count=3)
    single_segment_result = test_single_segment_matching(dataset, ds_agent)

    return {
        'segments_cutting': segments_cutting_results,
        'segment_similarity': segment_similarity_results,
        'single_segment_matching': single_segment_result
    }


def run_agent_tests(dataset_name_or_path='vfh'):
    """Run all Agent tests"""
    print("=== Running Agent Tests ===")
    config = load_config("config.yaml")
    dataset = DialogueDataset(resolve_dataset_path(dataset_name_or_path))

    api_key = config["api_key"]["openrouter"]
    base_url = config["base_url"]["openrouter"]
    model = config["model"]["openrouter"][0]

    # Create DSAgent for similarity matching and ensure embeddings are generated
    ds_agent = DSAgent(dataset)

    # Try to load embeddings first, generate if loading fails
    print("Attempting to load existing segment embeddings...")
    load_success = ds_agent.load_segment_embeddings()

    if not load_success:
        print("No existing embeddings found, generating new ones...")
        ds_agent.generate_segment_embeddings()
    else:
        print("Successfully loaded existing segment embeddings!")

    # Run each Agent test
    hs_agent_result = test_hs_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2)
    pn_agent_result = test_pn_agent(dataset, api_key, base_url, model, max_turns=2, num_threads=2)
    dts_agent_result = test_dts_agent(dataset, api_key, base_url, model, ds_agent, max_turns=2, num_threads=2)

    return {
        'hs_agent': hs_agent_result,
        'pn_agent': pn_agent_result,
        'dts_agent': dts_agent_result
    }


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vfh', help='vfh | dialseg_711 | doc2dial or path to json')
    parser.add_argument('mode', nargs='?', default='all', help='segments | agents | all')
    args_parsed, unknown = parser.parse_known_args()

    if args_parsed.mode == "segments":
        run_segment_tests(args_parsed.dataset)
    elif args_parsed.mode == "agents":
        run_agent_tests(args_parsed.dataset)
    else:
        run_all_tests(args_parsed.dataset)
