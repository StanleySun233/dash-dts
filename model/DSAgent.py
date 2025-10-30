import os
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class DSAgent:
    def __init__(self, dataset, model="BAAI/bge-m3", embedding_dir="./embeddings"):
        self.dataset = dataset
        self.model_name = model
        self.embedding_dir = embedding_dir

        self.model = SentenceTransformer(model)
        Path(self.embedding_dir).mkdir(parents=True, exist_ok=True)

    def _get_embedding_path(self):
        dataset_name = Path(self.dataset.data_path).stem
        return os.path.join(self.embedding_dir, f"{dataset_name}_embeddings.pkl")

    def _pad_sentences(self, utterances, max_sentences=64, max_length=128):
        padded_sentences = []
        for sentence in utterances:
            sentence_chars = list(str(sentence))
            if len(sentence_chars) > max_length:
                sentence_chars = sentence_chars[:max_length]
            else:
                sentence_chars.extend([''] * (max_length - len(sentence_chars)))
            padded_sentences.append(''.join(sentence_chars))

        if len(padded_sentences) > max_sentences:
            padded_sentences = padded_sentences[:max_sentences]
        else:
            empty_sentence = '' * max_length
            padded_sentences.extend([empty_sentence] * (max_sentences - len(padded_sentences)))

        return padded_sentences

    def generate_segment_embeddings(self):
        print("Generating segment embeddings...")
        for item in tqdm(self.dataset, desc="Processing dialogue segments"):
            segments = item.get_segments()
            item.segment_embeddings = []
            for segment in segments:
                if segment:  # Ensure segment is not empty
                    segment_embedding = self.model.encode(segment)
                    item.segment_embeddings.append(segment_embedding)
                else:
                    item.segment_embeddings.append(None)

        self.save_segment_embeddings()
        print("Segment embedding generation completed")

    def save_segment_embeddings(self):
        embedding_data = {}
        for item in self.dataset:
            if (hasattr(item, 'segment_embeddings') and
                    item.segment_embeddings is not None and
                    len(item.segment_embeddings) > 0):
                embedding_data[item.dial_id] = item.segment_embeddings

        embedding_path = self._get_embedding_path().replace('.pkl', '_segments.pkl')
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding_data, f)
        print(f"Saved {len(embedding_data)} segment embeddings to: {embedding_path}")

    def load_segment_embeddings(self):
        embedding_path = self._get_embedding_path().replace('.pkl', '_segments.pkl')

        if not os.path.exists(embedding_path):
            return False

        try:
            with open(embedding_path, 'rb') as f:
                embedding_data = pickle.load(f)

            loaded_count = 0
            for item in self.dataset:
                if item.dial_id in embedding_data:
                    item.segment_embeddings = embedding_data[item.dial_id]
                    loaded_count += 1

            print(f"Loaded {loaded_count} segment embeddings from: {embedding_path}")
            return True
        except Exception as e:
            print(f"Error loading segment embeddings: {e}")
            return False

    def find_most_similar_segment(self, target_utterance, target_segment_id=None):
        if not hasattr(target_utterance, 'segment_embeddings') or target_utterance.segment_embeddings is None:
            raise ValueError("Target utterance has no segment embeddings, please generate segment embeddings first")

        target_dial_id = target_utterance.dial_id
        max_similarity = -1
        most_similar_utterance = None
        most_similar_segment_id = None

        # Determine target segments to match
        target_segments = target_utterance.get_segments()
        if target_segment_id is not None:
            if target_segment_id >= len(target_segments):
                raise ValueError(f"Target segment ID {target_segment_id} out of range")
            target_segment_indices = [target_segment_id]
        else:
            target_segment_indices = range(len(target_segments))

        for target_seg_idx in target_segment_indices:
            if (target_seg_idx >= len(target_utterance.segment_embeddings) or
                    target_utterance.segment_embeddings[target_seg_idx] is None):
                continue

            target_embedding = target_utterance.segment_embeddings[target_seg_idx]

            # Iterate through all other dialogue segments
            for utterance in self.dataset:
                if utterance.dial_id == target_dial_id:
                    continue

                if not hasattr(utterance, 'segment_embeddings') or utterance.segment_embeddings is None:
                    continue

                # Iterate through current dialogue segments
                for seg_idx, segment_embedding in enumerate(utterance.segment_embeddings):
                    if segment_embedding is None:
                        continue

                    similarity = self._cosine_similarity(target_embedding, segment_embedding)

                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_utterance = utterance
                        most_similar_segment_id = seg_idx

        return most_similar_utterance, most_similar_segment_id, max_similarity

    def find_most_similar_segments_for_all(self, target_utterance):
        if not hasattr(target_utterance, 'segment_embeddings') or target_utterance.segment_embeddings is None:
            raise ValueError("Target utterance has no segment embeddings, please generate segment embeddings first")

        results = []
        segments = target_utterance.get_segments()

        for seg_idx, segment in enumerate(segments):
            if (seg_idx >= len(target_utterance.segment_embeddings) or
                    target_utterance.segment_embeddings[seg_idx] is None):
                results.append({
                    "segment_id": seg_idx,
                    "segment": segment,
                    "most_similar_utterance": None,
                    "most_similar_segment_id": None,
                    "similarity_score": 0.0
                })
                continue

            most_similar_utterance, most_similar_segment_id, similarity_score = self.find_most_similar_segment(
                target_utterance, seg_idx
            )

            results.append({
                "segment_id": seg_idx,
                "segment": segment,
                "most_similar_utterance": most_similar_utterance,
                "most_similar_segment_id": most_similar_segment_id,
                "similarity_score": similarity_score
            })

        return results

    def find_most_similar_for_context(self, target_utterance, context_segment_ids):
        if not hasattr(target_utterance, 'segment_embeddings') or target_utterance.segment_embeddings is None:
            raise ValueError("Target utterance has no segment embeddings, please generate segment embeddings first")

        target_dial_id = target_utterance.dial_id
        max_similarity = -1
        best_match = None

        # Calculate average embedding for target context
        target_embeddings = []
        for seg_id in context_segment_ids:
            if (seg_id < len(target_utterance.segment_embeddings) and
                    target_utterance.segment_embeddings[seg_id] is not None):
                target_embeddings.append(target_utterance.segment_embeddings[seg_id])

        if not target_embeddings:
            return None

        # Calculate average embedding
        target_avg_embedding = np.mean(target_embeddings, axis=0)

        # Iterate through all other dialogues
        for utterance in self.dataset:
            if utterance.dial_id == target_dial_id:
                continue

            if not hasattr(utterance, 'segment_embeddings') or utterance.segment_embeddings is None:
                continue

            # Iterate through current dialogue segments
            for seg_idx, segment_embedding in enumerate(utterance.segment_embeddings):
                if segment_embedding is None:
                    continue

                similarity = self._cosine_similarity(target_avg_embedding, segment_embedding)

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = {
                        "most_similar_utterance": utterance,
                        "most_similar_segment_id": seg_idx,
                        "similarity_score": similarity,
                        "context_segment_ids": context_segment_ids
                    }

        return best_match

    def find_most_similar_for_dynamic_context(self, target_utterance, current_utterance_idx, window_size):
        if not hasattr(target_utterance, 'segment_embeddings') or target_utterance.segment_embeddings is None:
            raise ValueError("Target utterance has no segment embeddings, please generate segment embeddings first")

        # Get segment information
        segment_info = target_utterance.get_segment_info()
        if not segment_info:
            return None

        # Determine which segment current utterance belongs to
        current_segment_id = None
        for seg_info in segment_info:
            if seg_info['start_idx'] <= current_utterance_idx < seg_info['end_idx']:
                current_segment_id = seg_info['segment_id']
                break

        if current_segment_id is None:
            return None

        # Determine context segments
        # Add adjacent segments based on window_size
        context_segment_ids = set()

        # Add current segment
        context_segment_ids.add(current_segment_id)

        # Add adjacent segments (simplified: add previous and next if exist)
        if current_segment_id > 0:
            context_segment_ids.add(current_segment_id - 1)
        if current_segment_id < len(segment_info) - 1:
            context_segment_ids.add(current_segment_id + 1)

        context_segment_ids = list(context_segment_ids)

        # Use existing method to find most similar sample
        return self.find_most_similar_for_context(target_utterance, context_segment_ids)

    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        if len(vec1.shape) == 2:
            vec1 = np.mean(vec1, axis=0)
        if len(vec2.shape) == 2:
            vec2 = np.mean(vec2, axis=0)

        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return similarity
