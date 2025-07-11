#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import itertools
import json
import logging
import os
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import random

from piper_phonemize import (
    phonemize_espeak,
    phonemize_codepoints,
    phoneme_ids_espeak,
    phoneme_ids_codepoints,
    get_codepoints_map,
    get_espeak_map,
    get_max_phonemes,
    tashkeel_run,
)

from .norm_audio import cache_norm_audio, make_silence_detector

_DIR = Path(__file__).parent
_VERSION = (_DIR / "VERSION").read_text(encoding="utf-8").strip()
_LOGGER = logging.getLogger("preprocess")

vad_phonemes = ["⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]

SPEAKER_IDS = {
    "SpongeBob": 0,
    "Mr. Krabs": 1,
    "Squidward": 2,
    "Patrick": 3,
    "Plankton": 4,
}

EMOTIONS = [
    # "Helpless",
    # "Frightened",
    "Overwhelmed",
    "Worried",
    # "Inadequate",
    # "Inferior",
    # "Worthless",
    # "Insignificant",
    # "Excluded",
    # "Persecuted",
    # "Nervous",
    # "Exposed",
    # "Betrayed",
    "Resentful",
    # "Disrespected",
    # "Ridiculed",
    "Indignant",
    # "Violated",
    # "Furious",
    # "Jealous",
    "Provoked",
    # "Hostile",
    # "Infuriated",
    "Annoyed",
    # "Withdrawn",
    # "Numb",
    # "Skeptical",
    "Dismissive",
    # "Judgemental",
    # "Embarrassed",
    # "Appalled",
    # "Revolted",
    # "Nauseated",
    # "Detestable",
    # "Horrified",
    # "Hesitant",
    "Disappointed",
    # "Empty",
    # "Remorseful",
    # "Ashamed",
    # "Powerless",
    # "Grief",
    # "Fragile",
    # "Victimized",
    # "Abandoned",
    # "Isolated",
    "Inspired",
    "Hopeful",
    # "Intimate",
    # "Sensitive",
    # "Thankful",
    # "Loving",
    # "Creative",
    # "Courageous",
    # "Valued",
    # "Respected",
    "Confident",
    # "Successful",
    # "Inquisitive",
    # "Curious",
    "Joyful",
    # "Free",
    # "Cheeky",
    # "Aroused",
    # "Energetic",
    # "Eager",
    # "Awe",
    "Astonished",
    "Perplexed",
    "Disillusioned",
    # "Dismayed",
    "Shocked",
    # "Unfocussed",
    # "Sleepy",
    # "Unmoored",
    # "Rushed",
    # "Pressured",
    # "Apathetic",
    # "Indifferent",
]

emotion_to_emoji = {
    # "Helpless": "🙇",
    # "Frightened": "😱",
    "Overwhelmed": "😵",
    "Worried": "😟",
    # "Inadequate": "😖",
    # "Inferior": "😔",
    # "Worthless": "🗑",
    # "Insignificant": "🐜",
    # "Excluded": "🚪",
    # "Persecuted": "🎯",
    # "Nervous": "😬",
    # "Exposed": "🔦",
    # "Betrayed": "🗡",
    "Resentful": "😤",
    # "Disrespected": "🙅",
    # "Ridiculed": "😝",
    "Indignant": "😠",
    # "Violated": "🔓",
    # "Furious": "🤬",
    # "Jealous": "🟢",
    "Provoked": "📛",
    # "Hostile": "👿",
    # "Infuriated": "😡",
    "Annoyed": "😒",
    # "Withdrawn": "🤐",
    # "Numb": "🧊",
    # "Skeptical": "🤨",
    "Dismissive": "🙄",
    # "Judgemental": "⚖",
    # "Embarrassed": "😳",
    # "Appalled": "😰",
    # "Revolted": "🤢",
    # "Nauseated": "🤮",
    # "Detestable": "💩",
    # "Horrified": "🧟",
    # "Hesitant": "🤔",
    "Disappointed": "😞",
    # "Empty": "🕳",
    # "Remorseful": "😥",
    # "Ashamed": "😣",
    # "Powerless": "😩",
    # "Grief": "😢",
    # "Fragile": "📦",
    # "Victimized": "🤕",
    # "Abandoned": "🏚",
    # "Isolated": "🏝",
    "Inspired": "💡",
    "Hopeful": "🌈",
    # "Intimate": "💞",
    # "Sensitive": "🌸",
    # "Thankful": "🙏",
    # "Loving": "❤",
    # "Creative": "🎨",
    # "Courageous": "🦁",
    # "Valued": "💎",
    # "Respected": "🛡",
    "Confident": "😎",
    # "Successful": "🏆",
    # "Inquisitive": "🧐",
    # "Curious": "🐱",
    "Joyful": "😄",
    # "Free": "🕊",
    # "Cheeky": "😜",
    # "Aroused": "🔥",
    # "Energetic": "⚡",
    # "Eager": "🤩",
    # "Awe": "🤯",
    "Astonished": "😲",
    "Perplexed": "😕",
    "Disillusioned": "🥀",
    # "Dismayed": "😧",
    "Shocked": "😮",
    # "Unfocussed": "🌀",
    # "Sleepy": "😴",
    # "Unmoored": "🌊",
    # "Rushed": "⏱",
    # "Pressured": "⛓",
    # "Apathetic": "😑",
    # "Indifferent": "🤷",
}
emoji_to_emotion = {v: k for k, v in emotion_to_emoji.items()}


def is_single_codepoint(char: str) -> bool:
    return len(char) == 1 and len(char.encode("utf-32-le")) == 4


emoji_counts = Counter(emotion_to_emoji.values())
duplicates = {emoji: count for emoji, count in emoji_counts.items() if count > 1}
mapped_emotions = set(emotion_to_emoji.keys())
expected_emotions = set(EMOTIONS)
missing = expected_emotions - mapped_emotions
extra = mapped_emotions - expected_emotions

assert not duplicates, f"Duplicate emojis found: {duplicates}"
assert not missing, f"Missing emotions: {missing}"
assert not extra, f"Extra emotions: {extra}"
for emotion, emoji in emotion_to_emoji.items():
    assert is_single_codepoint(
        emoji
    ), f"Emoji for emotion '{emotion}' is not a single codepoint: {emoji}"


def load_rejected_samples(*csv_paths: Path):
    rejected = set()
    for path in csv_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                if row.get("is_approved", "").strip().lower() == "n":
                    rejected.add(row["id"])
    return rejected


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    """Phonemes come from espeak-ng"""

    TEXT = "text"
    """Phonemes come from text itself"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Directory with audio dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files for training",
    )
    parser.add_argument("--language", required=True, help="eSpeak-ng voice")
    parser.add_argument(
        "--sample-rate",
        type=int,
        required=True,
        help="Target sample rate for voice (hertz)",
    )
    parser.add_argument(
        "--dataset-format", choices=("ljspeech", "mycroft"), required=True
    )
    parser.add_argument("--cache-dir", help="Directory to cache processed audio files")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--single-speaker", action="store_true", help="Force single speaker dataset"
    )
    parser.add_argument(
        "--speaker-id", type=int, help="Add speaker id to single speaker dataset"
    )
    #
    parser.add_argument(
        "--phoneme-type",
        choices=list(PhonemeType),
        default=PhonemeType.ESPEAK,
        help="Type of phonemes to use (default: espeak)",
    )
    parser.add_argument(
        "--text-casing",
        choices=("ignore", "lower", "upper", "casefold"),
        default="ignore",
        help="Casing applied to utterance text",
    )
    #
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset to put in config (default: name of <ouput_dir>/../)",
    )
    parser.add_argument(
        "--audio-quality",
        help="Audio quality to put in config (default: name of <output_dir>)",
    )
    #
    parser.add_argument(
        "--tashkeel",
        action="store_true",
        help="Diacritize Arabic text with libtashkeel",
    )
    #
    parser.add_argument(
        "--skip-audio", action="store_true", help="Don't preprocess audio"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=0,
        help="Amount of samples to reserve per emotion for each speaker for a test dataset (as test_dataset.json)",
    )
    args = parser.parse_args()

    if args.single_speaker and (args.speaker_id is not None):
        _LOGGER.fatal("--single-speaker and --speaker-id cannot both be provided")
        return

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # Prevent log spam
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Ensure enum
    args.phoneme_type = PhonemeType(args.phoneme_type)

    # Convert to paths and create output directories
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else args.output_dir / "cache" / str(args.sample_rate)
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    global rejected_samples
    rejected_samples = load_rejected_samples(
        Path(os.path.join(args.input_dir, "_samples_dashes.csv")),
        Path(os.path.join(args.input_dir, "_samples_expressive.csv")),
    )

    if args.dataset_format == "mycroft":
        make_dataset = mycroft_dataset
    else:
        make_dataset = ljspeech_dataset

    # Count speakers
    _LOGGER.debug("Counting number of speakers/utterances in the dataset")
    speaker_counts: "Counter[str]" = Counter()
    num_utterances = 0
    for utt in make_dataset(args):
        speaker = utt.speaker or ""
        speaker_counts[speaker] += 1
        num_utterances += 1

    assert num_utterances > 0, "No utterances found"

    is_multispeaker = len(speaker_counts) > 1
    speaker_ids: Dict[str, int] = {}

    if is_multispeaker:
        speaker_ids = SPEAKER_IDS.copy()
        _LOGGER.info("Using fixed speaker ID mapping")
    else:
        _LOGGER.info("Single speaker dataset")

    # Write config
    audio_quality = args.audio_quality or args.output_dir.name
    dataset_name = args.dataset_name or args.output_dir.parent.name

    global codepoints_map
    codepoints_map = get_espeak_map()
    custom_phonemes = set(emotion_to_emoji.values()).union(vad_phonemes)
    codepoint_phonemes = codepoints_map.keys()
    duplicate_codepoints = custom_phonemes & codepoint_phonemes
    assert (
        not duplicate_codepoints
    ), f"Tried to add custom phonemes to the codepoints map that are already reserved by Espeak: {duplicate_codepoints}"

    # Predefined, fixed emoji → ID mapping
    fixed_emoji_ids = {
        "🌈": [158],
        "😄": [159],
        "😞": [160],
        "😵": [161],
        "😒": [162],
        "😎": [163],
        "😠": [164],
        "😮": [165],
        "🙄": [166],
        "😲": [167],
        "💡": [168],
        "😟": [169],
        "🥀": [170],
        "😤": [171],
        "😕": [172],
        "📛": [173],
    }

    # Add VAD phonemes next, starting from the next available ID
    existing_ids = {i for v in codepoints_map.values() for i in v}
    custom_id = max(existing_ids) + 1

    codepoints_map["‽"] = [custom_id]

    # Inject fixed emoji IDs
    for emoji, id_list in fixed_emoji_ids.items():
        codepoints_map[emoji] = id_list

    custom_id += len(fixed_emoji_ids) + 1

    for custom_phoneme in vad_phonemes:
        assert (
            custom_id < get_max_phonemes()
        ), f"Too many phonemes. Attempted to add custom phonemes beyond the current max."
        codepoints_map[custom_phoneme] = [custom_id]
        custom_id += 1

    with open(args.output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "dataset": dataset_name,
                "audio": {
                    "sample_rate": args.sample_rate,
                    "quality": audio_quality,
                },
                "espeak": {
                    "voice": args.language,
                },
                "language": {
                    "code": args.language,
                },
                "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
                "phoneme_type": args.phoneme_type.value,
                "phoneme_map": {},
                "phoneme_id_map": (
                    get_codepoints_map()[args.language]
                    if args.phoneme_type == PhonemeType.TEXT
                    else codepoints_map
                ),
                "num_symbols": get_max_phonemes(),
                "num_speakers": len(speaker_counts),
                "speaker_id_map": speaker_ids,
                "piper_version": _VERSION,
            },
            config_file,
            ensure_ascii=False,
            indent=4,
        )
    _LOGGER.info("Wrote dataset config")

    if (args.max_workers is None) or (args.max_workers < 1):
        args.max_workers = os.cpu_count()

    assert args.max_workers is not None

    batch_size = int(num_utterances / (args.max_workers * 2))
    queue_in: "Queue[Iterable[Utterance]]" = JoinableQueue()
    queue_out: "Queue[Optional[Utterance]]" = Queue()

    # Start workers
    if args.phoneme_type == PhonemeType.TEXT:
        target = phonemize_batch_text
    else:
        target = phonemize_batch_espeak

    processes = [
        Process(target=target, args=(args, queue_in, queue_out))
        for _ in range(args.max_workers)
    ]
    for proc in processes:
        proc.start()

    _LOGGER.info(
        "Processing %s utterance(s) with %s worker(s)", num_utterances, args.max_workers
    )

    utt_dicts = []

    # with open(args.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
    for utt_batch in batched(
        make_dataset(args),
        batch_size,
    ):
        queue_in.put(utt_batch)

    _LOGGER.debug("Waiting for jobs to finish")
    missing_phonemes: "Counter[str]" = Counter()
    for _ in range(num_utterances):
        utt = queue_out.get()
        if utt is not None:
            if utt.speaker is not None:
                utt.speaker_id = speaker_ids[utt.speaker]

            utt_dict = dataclasses.asdict(utt)
            utt_dict.pop("missing_phonemes")

            # # JSONL
            # json.dump(
            #     utt_dict,
            #     dataset_file,
            #     ensure_ascii=False,
            #     cls=PathEncoder,
            # )
            # print("", file=dataset_file)
            utt_dicts.append(utt_dict)

            missing_phonemes.update(utt.missing_phonemes)

    random.shuffle(utt_dicts)

    test_dataset = {
        name: {emoji: [] for emoji in emoji_to_emotion} for name in SPEAKER_IDS
    }

    train_dataset = []

    for utt_dict in utt_dicts:
        speaker = utt_dict["speaker"]
        emotion_phoneme = utt_dict["phonemes"][0]

        if len(test_dataset[speaker][emotion_phoneme]) < args.test_count:
            test_dataset[speaker][emotion_phoneme].append(utt_dict)
        else:
            train_dataset.append(utt_dict)

    with open(args.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
        for utt_dict in train_dataset:
            json.dump(
                utt_dict,
                dataset_file,
                ensure_ascii=False,
                cls=PathEncoder,
            )
            print("", file=dataset_file)

    if args.test_count > 0:
        with open(
            args.output_dir / "test_dataset.jsonl", "w", encoding="utf-8"
        ) as dataset_file:
            for speaker, emotion_samples in test_dataset.items():
                for emotion, samples in emotion_samples.items():
                    for sample in samples:
                        json.dump(
                            sample,
                            dataset_file,
                            ensure_ascii=False,
                            cls=PathEncoder,
                        )
                        print("", file=dataset_file)

    if missing_phonemes:
        for phoneme, count in missing_phonemes.most_common():
            _LOGGER.warning("Missing %s (%s)", phoneme, count)

        _LOGGER.warning("Missing %s phoneme(s)", len(missing_phonemes))

    # Signal workers to stop
    for proc in processes:
        queue_in.put(None)

    # Wait for workers to stop
    for proc in processes:
        proc.join(timeout=1)


# -----------------------------------------------------------------------------


def get_text_casing(casing: str):
    if casing == "lower":
        return str.lower

    if casing == "upper":
        return str.upper

    if casing == "casefold":
        return str.casefold

    return lambda s: s


def vad_phonemize(vad_str: str):
    return [vad_phonemes[int(char)] for char in vad_str]


def vad_phoneme_ids(vad_str: str):
    phonemes = [vad_phonemes[int(char)] for char in vad_str]
    ids = []
    for phoneme in phonemes:
        ids.append(codepoints_map[phoneme][0])
    return ids


def phonemize_batch_espeak(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug(utt)
                    utt_parts = utt.text.split(maxsplit=4)
                    utt_emotion = utt_parts[0]
                    utt_valence = utt_parts[1]
                    utt_arousal = utt_parts[2]
                    utt_dominance = utt_parts[3]
                    utt_text = utt_parts[4]

                    has_interrobang = False
                    if "?!" in utt_text:
                        utt_text = utt_text.replace("?!", "")
                        has_interrobang = True

                    assert utt_emotion in EMOTIONS, f"Invalid emotion: {utt_emotion}"
                    # if not utt_emotion in EMOTIONS:
                    #     continue

                    all_phonemes = phonemize_espeak(casing(utt_text), args.language)

                    # Flatten
                    utt_text_phonemes = [
                        phoneme
                        for sentence_phonemes in all_phonemes
                        for phoneme in sentence_phonemes
                    ]

                    emotion_phoneme = emotion_to_emoji[utt_emotion]

                    utt.phonemes = [
                        emotion_phoneme,
                        # " ",
                        *vad_phonemize(utt_valence),
                        # " ",
                        *vad_phonemize(utt_arousal),
                        # " ",
                        *vad_phonemize(utt_dominance),
                        # " ",
                        *utt_text_phonemes,
                    ]
                    if has_interrobang:
                        utt.phonemes.append("‽")

                    phoneme_ids = phoneme_ids_espeak(
                        utt_text_phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    space_id = codepoints_map[" "][0]
                    emotion_id = codepoints_map[emotion_phoneme]

                    # utt.phoneme_ids = phoneme_ids_espeak(
                    #     utt.phonemes,
                    #     missing_phonemes=utt.missing_phonemes,
                    # )
                    utt.phoneme_ids = [
                        emotion_id[0],
                        # space_id,
                        *vad_phoneme_ids(utt_valence),
                        # space_id,
                        *vad_phoneme_ids(utt_arousal),
                        # space_id,
                        *vad_phoneme_ids(utt_dominance),
                        # space_id,
                        *phoneme_ids,
                    ]
                    if has_interrobang:
                        interrobang_id = codepoints_map["‽"][0]
                        utt.phoneme_ids.insert(-1, interrobang_id)

                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_espeak")


def phonemize_batch_text(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug(utt)
                    all_phonemes = phonemize_codepoints(casing(utt.text))
                    # Flatten
                    utt.phonemes = [
                        phoneme
                        for sentence_phonemes in all_phonemes
                        for phoneme in sentence_phonemes
                    ]
                    utt.phoneme_ids = phoneme_ids_codepoints(
                        args.language,
                        utt.phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_text")


# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    text: str
    audio_path: Path
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    phonemes: Optional[List[str]] = None
    phoneme_ids: Optional[List[int]] = None
    audio_norm_path: Optional[Path] = None
    audio_spec_path: Optional[Path] = None
    missing_phonemes: "Counter[str]" = field(default_factory=Counter)


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def ljspeech_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    speaker_id = args.speaker_id
    skip_audio = args.skip_audio

    # filename|speaker|text
    # speaker is optional
    metadata_path = dataset_dir / "metadata.csv"
    assert metadata_path.exists(), f"Missing {metadata_path}"

    wav_dir = dataset_dir / "wav"
    if not wav_dir.is_dir():
        wav_dir = dataset_dir / "wavs"

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            assert len(row) >= 2, "Not enough columns"

            speaker: Optional[str] = None
            if is_single_speaker or (len(row) == 2):
                filename, text = row[0], row[-1]
            else:
                filename, speaker, text = row[0], row[1], row[-1]

            # Try file name relative to metadata
            wav_path = metadata_path.parent / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = metadata_path.parent / f"{filename}.wav"

            if not wav_path.exists():
                # Try wav/ or wavs/
                wav_path = wav_dir / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = wav_dir / f"{filename}.wav"

            if not skip_audio:
                if not wav_path.exists():
                    _LOGGER.warning("Missing %s", filename)
                    continue

                if wav_path.stat().st_size == 0:
                    _LOGGER.warning("Empty file: %s", wav_path)
                    continue

            if filename in rejected_samples:
                continue

            # Skip the sample that's too long and was causing GPU out of memory errors during training
            if "17bd0229-5a34-4fed-aa91-c2dd4c582bd4" in filename:
                continue

            if speaker is not None and speaker not in SPEAKER_IDS:
                continue

            yield Utterance(
                text=text, audio_path=wav_path, speaker=speaker, speaker_id=speaker_id
            )


def mycroft_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    skip_audio = args.skip_audio

    speaker_id = 0
    for metadata_path in dataset_dir.glob("**/*-metadata.txt"):
        speaker = metadata_path.parent.name if not is_single_speaker else None
        with open(metadata_path, "r", encoding="utf-8") as csv_file:
            # filename|text|length
            reader = csv.reader(csv_file, delimiter="|")
            for row in reader:
                filename, text = row[0], row[1]
                wav_path = metadata_path.parent / filename
                if skip_audio or (wav_path.exists() and (wav_path.stat().st_size > 0)):
                    yield Utterance(
                        text=text,
                        audio_path=wav_path,
                        speaker=speaker,
                        speaker_id=speaker_id if not is_single_speaker else None,
                    )
        speaker_id += 1


# -----------------------------------------------------------------------------


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    batch = list(itertools.islice(it, n))
    while batch:
        yield batch
        batch = list(itertools.islice(it, n))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
