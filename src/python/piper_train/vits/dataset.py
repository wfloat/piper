import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset

_LOGGER = logging.getLogger("vits.dataset")


@dataclass
class Utterance:
    phoneme_ids: List[int]
    audio_norm_path: Path
    audio_spec_path: Path
    speaker_id: Optional[int] = None
    text: Optional[str] = None


@dataclass
class UtteranceTensors:
    phoneme_ids: LongTensor
    spectrogram: FloatTensor
    audio_norm: FloatTensor
    speaker_id: Optional[LongTensor] = None
    text: Optional[str] = None

    @property
    def spec_length(self) -> int:
        return self.spectrogram.size(1)


@dataclass
class Batch:
    phoneme_ids: LongTensor
    phoneme_lengths: LongTensor
    spectrograms: FloatTensor
    spectrogram_lengths: LongTensor
    audios: FloatTensor
    audio_lengths: LongTensor
    speaker_ids: Optional[LongTensor] = None


def randomize_vad_digits(vad_digits: str) -> str:
    vad = int(vad_digits)
    random_choice = random.choice([-2, -1, 0, 1, 2])
    vad = vad + random_choice
    vad = max(0, vad)
    vad_digits = str(vad).zfill(3)
    return vad_digits


class PiperDataset(Dataset):
    """
    Dataset format:

    * phoneme_ids (required)
    * audio_norm_path (required)
    * audio_spec_path (required)
    * text (optional)
    * phonemes (optional)
    * audio_path (optional)
    """

    def __init__(
        self,
        dataset_paths: List[Union[str, Path]],
        max_phoneme_ids: Optional[int] = None,
    ):
        self.utterances: List[Utterance] = []

        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            _LOGGER.debug("Loading dataset: %s", dataset_path)
            self.utterances.extend(
                PiperDataset.load_dataset(dataset_path, max_phoneme_ids=max_phoneme_ids)
            )

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx) -> UtteranceTensors:
        utt = self.utterances[idx]
        phoneme_ids_to_digit = {
            174: 0,
            175: 1,
            176: 2,
            177: 3,
            178: 4,
            179: 5,
            180: 6,
            181: 7,
            182: 8,
            183: 9,
        }
        digits_to_phoneme_id = {v: k for k, v in phoneme_ids_to_digit.items()}
        phoneme_ids = utt.phoneme_ids.copy()

        valence_digits = "".join(
            [
                str(phoneme_ids_to_digit[utt.phoneme_ids[1]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[2]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[3]]),
            ]
        )
        valence_digits = randomize_vad_digits(valence_digits)
        phoneme_ids[1] = digits_to_phoneme_id[int(valence_digits[0])]
        phoneme_ids[2] = digits_to_phoneme_id[int(valence_digits[1])]
        phoneme_ids[3] = digits_to_phoneme_id[int(valence_digits[2])]

        arousal_digits = "".join(
            [
                str(phoneme_ids_to_digit[utt.phoneme_ids[4]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[5]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[6]]),
            ]
        )
        arousal_digits = randomize_vad_digits(arousal_digits)
        phoneme_ids[4] = digits_to_phoneme_id[int(arousal_digits[0])]
        phoneme_ids[5] = digits_to_phoneme_id[int(arousal_digits[1])]
        phoneme_ids[6] = digits_to_phoneme_id[int(arousal_digits[2])]

        dominance_digits = "".join(
            [
                str(phoneme_ids_to_digit[utt.phoneme_ids[7]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[8]]),
                str(phoneme_ids_to_digit[utt.phoneme_ids[9]]),
            ]
        )
        dominance_digits = randomize_vad_digits(dominance_digits)
        phoneme_ids[7] = digits_to_phoneme_id[int(dominance_digits[0])]
        phoneme_ids[8] = digits_to_phoneme_id[int(dominance_digits[1])]
        phoneme_ids[9] = digits_to_phoneme_id[int(dominance_digits[2])]

        return UtteranceTensors(
            phoneme_ids=LongTensor(phoneme_ids),
            audio_norm=torch.load(utt.audio_norm_path),
            spectrogram=torch.load(utt.audio_spec_path),
            speaker_id=(
                LongTensor([utt.speaker_id]) if utt.speaker_id is not None else None
            ),
            text=utt.text,
        )

    @staticmethod
    def load_dataset(
        dataset_path: Path,
        max_phoneme_ids: Optional[int] = None,
    ) -> Iterable[Utterance]:
        num_skipped = 0

        with open(dataset_path, "r", encoding="utf-8") as dataset_file:
            for line_idx, line in enumerate(dataset_file):
                line = line.strip()
                if not line:
                    continue

                try:
                    utt = PiperDataset.load_utterance(line)
                    if (max_phoneme_ids is None) or (
                        len(utt.phoneme_ids) <= max_phoneme_ids
                    ):
                        yield utt
                    else:
                        num_skipped += 1
                except Exception:
                    _LOGGER.exception(
                        "Error on line %s of %s: %s",
                        line_idx + 1,
                        dataset_path,
                        line,
                    )

        if num_skipped > 0:
            _LOGGER.warning("Skipped %s utterance(s)", num_skipped)

    @staticmethod
    def load_utterance(line: str) -> Utterance:
        utt_dict = json.loads(line)
        return Utterance(
            phoneme_ids=utt_dict["phoneme_ids"],
            audio_norm_path=Path(utt_dict["audio_norm_path"]),
            audio_spec_path=Path(utt_dict["audio_spec_path"]),
            speaker_id=utt_dict.get("speaker_id"),
            text=utt_dict.get("text"),
        )


class UtteranceCollate:
    def __init__(self, is_multispeaker: bool, segment_size: int):
        self.is_multispeaker = is_multispeaker
        self.segment_size = segment_size

    def __call__(self, utterances: Sequence[UtteranceTensors]) -> Batch:
        num_utterances = len(utterances)
        assert num_utterances > 0, "No utterances"

        max_phonemes_length = 0
        max_spec_length = 0
        max_audio_length = 0

        num_mels = 0

        # Determine lengths
        for utt_idx, utt in enumerate(utterances):
            assert utt.spectrogram is not None
            assert utt.audio_norm is not None

            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            max_phonemes_length = max(max_phonemes_length, phoneme_length)
            max_spec_length = max(max_spec_length, spec_length)
            max_audio_length = max(max_audio_length, audio_length)

            num_mels = utt.spectrogram.size(0)
            if self.is_multispeaker:
                assert utt.speaker_id is not None, "Missing speaker id"

        # Audio cannot be smaller than segment size (8192)
        max_audio_length = max(max_audio_length, self.segment_size)

        # Create padded tensors
        phonemes_padded = LongTensor(num_utterances, max_phonemes_length)
        spec_padded = FloatTensor(num_utterances, num_mels, max_spec_length)
        audio_padded = FloatTensor(num_utterances, 1, max_audio_length)

        phonemes_padded.zero_()
        spec_padded.zero_()
        audio_padded.zero_()

        phoneme_lengths = LongTensor(num_utterances)
        spec_lengths = LongTensor(num_utterances)
        audio_lengths = LongTensor(num_utterances)

        speaker_ids: Optional[LongTensor] = None
        if self.is_multispeaker:
            speaker_ids = LongTensor(num_utterances)

        # Sort by decreasing spectrogram length
        sorted_utterances = sorted(
            utterances, key=lambda u: u.spectrogram.size(1), reverse=True
        )
        for utt_idx, utt in enumerate(sorted_utterances):
            phoneme_length = utt.phoneme_ids.size(0)
            spec_length = utt.spectrogram.size(1)
            audio_length = utt.audio_norm.size(1)

            phonemes_padded[utt_idx, :phoneme_length] = utt.phoneme_ids
            phoneme_lengths[utt_idx] = phoneme_length

            spec_padded[utt_idx, :, :spec_length] = utt.spectrogram
            spec_lengths[utt_idx] = spec_length

            audio_padded[utt_idx, :, :audio_length] = utt.audio_norm
            audio_lengths[utt_idx] = audio_length

            if utt.speaker_id is not None:
                assert speaker_ids is not None
                speaker_ids[utt_idx] = utt.speaker_id

        return Batch(
            phoneme_ids=phonemes_padded,
            phoneme_lengths=phoneme_lengths,
            spectrograms=spec_padded,
            spectrogram_lengths=spec_lengths,
            audios=audio_padded,
            audio_lengths=audio_lengths,
            speaker_ids=speaker_ids,
        )
