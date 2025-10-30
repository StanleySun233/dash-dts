from dialogue_dataset import DialogueDataset
from model.DSAgent import DSAgent
from utils import resolve_dataset_path

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vfh', help='vfh | dialseg_711 | doc2dial | all or path to json')
    return parser.parse_args()


def _run_for_dataset(dataset_key_or_path: str):
    print(f"\n===== Running for dataset: {dataset_key_or_path} =====")
    dataset = DialogueDataset(resolve_dataset_path(dataset_key_or_path))
    ds_agent = DSAgent(dataset)

    # Generate segment embeddings
    if not hasattr(dataset[0], 'segment_embeddings') or dataset[0].segment_embeddings is None or len(
            dataset[0].segment_embeddings) == 0:
        ds_agent.generate_segment_embeddings()
    else:
        print("Using existing segment embeddings")

    ds_agent.save_segment_embeddings()
    ds_agent.load_segment_embeddings()

    test_count = min(5, len(dataset))
    print(f"Testing segment similarity matching for {test_count} dialogues")

    for i in range(test_count):
        target_utterance = dataset[i]
        print(f"\nTarget Utterance (dial_id: {target_utterance.dial_id}): {target_utterance.utterances}")

        try:
            # Test segment similarity matching
            segment_results = ds_agent.find_most_similar_segments_for_all(target_utterance)

            for result in segment_results:
                seg_id = result['segment_id']
                segment = result['segment']
                similar_utterance = result['most_similar_utterance']
                similar_seg_id = result['most_similar_segment_id']
                similarity_score = result['similarity_score']

                print(f"\nSegment {seg_id}: {segment}")

                if similar_utterance is not None:
                    similar_segment = similar_utterance.get_segments()[similar_seg_id]
                    print(
                        f"Most similar (dial_id: {similar_utterance.dial_id}, segment {similar_seg_id}): {similar_segment}")
                    print(f"Similarity score: {similarity_score:.4f}")
                else:
                    print("No similar segment found")

        except Exception as e:
            print(f"Error during testing: {e}")


def main():
    args = args_parser()

    if args.dataset == 'all':
        for ds_name in ['vfh', 'dialseg_711', 'doc2dial']:
            _run_for_dataset(ds_name)
    else:
        _run_for_dataset(args.dataset)


if __name__ == "__main__":
    main()
