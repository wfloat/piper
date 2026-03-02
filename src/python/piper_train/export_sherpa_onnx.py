# https://k2-fsa.github.io/sherpa/onnx/tts/piper.html

#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import onnx
import torch

from .vits.lightning import VitsModel

def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def load_config(model_config_path: str):
    with open(model_config_path, "r") as file:
        config = json.load(file)
    return config


def _to_int_token_id(value: Any) -> Optional[int]:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _token_id_from_map(id_map: Dict[str, Any], symbol: str) -> Optional[int]:
    if symbol not in id_map:
        return None

    return _to_int_token_id(id_map[symbol])


def generate_tokens(config: Dict[str, Any], out_path: Path):
    id_map = config["phoneme_id_map"]
    with open(out_path, "w", encoding="utf-8") as f:
        for s, i in id_map.items():
            if s == "\n":
                continue

            token_id = _to_int_token_id(i)
            if token_id is None:
                continue

            f.write(f"{s} {token_id}\n")

    print("Generated tokens.txt")

_LOGGER = logging.getLogger("piper_train.export_sherpa_onnx")

OPSET_VERSION = 15


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--output", help="Path to output model (.onnx)")
    parser.add_argument("--config_path", help="Path to model config.json (.onnx)")
    parser.add_argument(
        "--use-eos-bos",
        type=int,
        choices=[0, 1],
        required=True,
        help="Whether Wfloat text frontend should wrap tokens with BOS/PAD/EOS.",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.config_path = Path(args.config_path)

    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None)
    model_g = model.model_g

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    # old_forward = model_g.infer

    def infer_forward(text, text_lengths, scales, sid=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    dummy_input_length = 50
    sequences = torch.randint(
        low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long
    )
    sequence_lengths = torch.LongTensor([sequences.size(1)])

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0])

    # noise, length, noise_w
    scales = torch.FloatTensor([0.667, 1.0, 0.8])
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
    )

    _LOGGER.info("Exported model to %s", args.output)

    config = load_config(args.config_path)
    tokens_path = args.output.parent / f"{args.output.stem}_tokens.txt"
    generate_tokens(config, tokens_path)

    phoneme_id_map = config.get("phoneme_id_map", {})
    use_eos_bos = int(args.use_eos_bos)

    language = config.get("language", {})
    if isinstance(language, dict):
        language = language.get("code", "")

    espeak = config.get("espeak", {})
    if isinstance(espeak, dict):
        voice = espeak.get("voice", "")
    else:
        voice = ""

    audio = config.get("audio", {})
    if isinstance(audio, dict):
        sample_rate = int(audio.get("sample_rate", 22050))
    else:
        sample_rate = 22050

    if sample_rate == 22500:
        sample_rate = 22050

    meta_data = {
        "model_type": "vits",
        "comment": "piper",  # must be piper for models from piper
        "language": language,
        "voice": voice,  # e.g., en-us
        "has_espeak": 1,
        "n_speakers": int(config.get("num_speakers", 0)),
        "sample_rate": sample_rate,
        "add_blank": int(config.get("add_blank", 0)),
        "use_eos_bos": use_eos_bos, 
    }

    pad_id = _token_id_from_map(phoneme_id_map, "_")
    if pad_id is not None:
        meta_data["pad_id"] = pad_id

    bos_id = _token_id_from_map(phoneme_id_map, "^")
    if bos_id is not None:
        meta_data["bos_id"] = bos_id

    eos_id = _token_id_from_map(phoneme_id_map, "$")
    if eos_id is not None:
        meta_data["eos_id"] = eos_id

    add_meta_data(args.output, meta_data)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
