import json
from typing import List

from torch.utils.data import Dataset
from tqdm import tqdm


class Utterance:
    def __init__(self, dial_id, utterances, segments, utt_lst):
        self.dial_id = dial_id
        self.utterances = utterances
        self.segments = segments
        self.utt_list = utt_lst
        self.len = len(self.utt_list)
        self.handshake = []
        self.embedding = []

    def __len__(self):
        return self.len

    def load_index(self, idx, window=2):
        output = {"previous": [], "current": "", "next": []}

        output["current"] = self.utterances[idx]

        if idx < window:
            output["previous"] = ["<dialogue_start>"] * (window - idx) + self.utterances[:idx]
        else:
            output["previous"] = self.utterances[idx - window:idx]

        remaining = self.len - idx - 1
        if remaining < window:
            output["next"] = self.utterances[idx + 1:] + ["<dialogue_end>"] * (window - remaining)
        else:
            output["next"] = self.utterances[idx + 1:idx + 1 + window]

        return output

    def get_segments(self):
        if not self.segments:
            return [self.utterances]

        segments = []
        start_idx = 0

        for segment_length in self.segments:
            if segment_length > 0:
                end_idx = start_idx + segment_length
                segment = self.utterances[start_idx:end_idx]
                segments.append(segment)
                start_idx = end_idx

        return segments

    def get_segment_info(self):
        if not self.segments:
            return [{"start_idx": 0, "end_idx": len(self.utterances), "utterances": self.utterances}]

        segment_info = []
        start_idx = 0

        for i, segment_length in enumerate(self.segments):
            if segment_length > 0:
                end_idx = start_idx + segment_length
                segment_info.append({
                    "segment_id": i,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "utterances": self.utterances[start_idx:end_idx]
                })
                start_idx = end_idx

        return segment_info


class DialogueDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data: List[Utterance] = []
        self._load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in tqdm(data, desc="Loading dataset"):
            norm = self._normalize_item(item)
            self.data.append(Utterance(**norm))

    def _normalize_item(self, raw: dict) -> dict:
        """Normalize various dataset schemas to Utterance constructor.

        Expected output keys: dial_id(str), utterances(List[str]), utt_lst(List[int]), segments(List[int])
        """
        # 1) extract dial_id
        dial_id = str(
            raw.get('dial_id') or raw.get('dialogue_id') or raw.get('id') or raw.get('conversation_id') or raw.get('conv_id') or ''
        )

        # 2) extract utterances
        utterances = None
        if isinstance(raw.get('utterances'), list):
            utts = raw.get('utterances')
            if utts and isinstance(utts[0], dict):
                # try common text keys
                text_key = next((k for k in ['text', 'utterance', 'content', 'value'] if k in utts[0]), None)
                utterances = [u.get(text_key, '') for u in utts] if text_key else [str(u) for u in utts]
            else:
                utterances = [str(u) for u in utts]
        if utterances is None:
            # other common keys
            for key in ['dialog', 'turns', 'texts', 'sentences']:
                v = raw.get(key)
                if isinstance(v, list):
                    if v and isinstance(v[0], dict):
                        text_key = next((k for k in ['text', 'utterance', 'content', 'value'] if k in v[0]), None)
                        utterances = [d.get(text_key, '') for d in v] if text_key else [str(d) for d in v]
                    else:
                        utterances = [str(x) for x in v]
                    break
        if utterances is None:
            utterances = []

        total_len = len(utterances)

        # 3) extract segments (list of lengths)
        segments = None
        if isinstance(raw.get('segments'), list) and all(isinstance(x, int) for x in raw.get('segments')):
            segments = list(raw.get('segments'))
        elif isinstance(raw.get('boundaries'), list):
            boundaries = [int(bool(x)) for x in raw.get('boundaries')]
            segments = self._boundaries_to_segments(boundaries, total_len)
        elif isinstance(raw.get('boundary'), list):
            boundaries = [int(bool(x)) for x in raw.get('boundary')]
            segments = self._boundaries_to_segments(boundaries, total_len)
        elif isinstance(raw.get('segment_ids'), list):
            seg_ids = raw.get('segment_ids')
            segments = self._segment_ids_to_segments(seg_ids)
        else:
            # fallback: single segment
            segments = [total_len] if total_len > 0 else []

        # 4) build utt_lst from segments
        utt_lst: List[int] = []
        seg_index = 0
        for seg_len in segments:
            for _ in range(seg_len):
                utt_lst.append(seg_index)
            seg_index += 1

        return {
            'dial_id': dial_id,
            'utterances': utterances,
            'utt_lst': utt_lst,
            'segments': segments,
        }

    @staticmethod
    def _boundaries_to_segments(boundaries: List[int], total_length: int) -> List[int]:
        # Boundaries mark the end of a segment at position i (usually between i and i+1)
        # We assume boundaries length equals total_length; last position should be 0.
        segments: List[int] = []
        if total_length <= 0:
            return segments
        count = 1
        for i in range(1, total_length):
            if i - 1 < len(boundaries) and boundaries[i - 1] == 1:
                segments.append(count)
                count = 1
            else:
                count += 1
        segments.append(count)
        return segments

    @staticmethod
    def _segment_ids_to_segments(segment_ids: List[int]) -> List[int]:
        # Convert per-utterance segment id sequence like [0,0,1,1,2] into lengths [2,2,1]
        if not segment_ids:
            return []
        segments: List[int] = []
        current_id = segment_ids[0]
        count = 1
        for sid in segment_ids[1:]:
            if sid == current_id:
                count += 1
            else:
                segments.append(count)
                current_id = sid
                count = 1
        segments.append(count)
        return segments

    def print(self, idx):
        item = self.__getitem__(idx)


if __name__ == "__main__":
    import argparse
    from utils import resolve_dataset_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='vfh', help='vfh | dialseg_711 | doc2dial or path to json')
    args = parser.parse_args()

    ds_path = resolve_dataset_path(args.dataset)
    dataset = DialogueDataset(ds_path)
    if len(dataset) > 0:
        print(len(dataset[0]))
        print(dataset[0].load_index(7, 2))
    print(f"Loaded dataset from: {ds_path}, size={len(dataset)}")
