import argparse
import json
import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)
DATASET_REFRESH_FULL_MODE = "full"
DATASET_REFRESH_TRAIN_MODE = "train"


def _run_dataset_refresh(command: str, mode_arg: str, cwd: Path):
    refresh_command = [*shlex.split(command), mode_arg]
    _LOGGER.info("Running dataset refresh command (%s): %s", mode_arg, refresh_command)
    subprocess.run(refresh_command, check=True, cwd=str(cwd))


def _resolve_manifest_path(
    dataset_dir: Path, path_str: Optional[str], default_name: str
) -> Path:
    if path_str:
        path = Path(path_str)
        if not path.is_absolute():
            path = dataset_dir / path
        return path

    return dataset_dir / default_name


def _resolve_split_manifests(args) -> Optional[Tuple[Path, Path, Path]]:
    has_any_split_manifest = any(
        [args.train_manifest, args.val_manifest, args.test_manifest]
    )
    if has_any_split_manifest and not all(
        [args.train_manifest, args.val_manifest, args.test_manifest]
    ):
        raise ValueError(
            "--train-manifest, --val-manifest, and --test-manifest must be set together"
        )

    if not has_any_split_manifest and not args.dataset_refresh_command:
        return None

    return (
        _resolve_manifest_path(args.dataset_dir, args.train_manifest, "train.jsonl"),
        _resolve_manifest_path(args.dataset_dir, args.val_manifest, "val.jsonl"),
        _resolve_manifest_path(args.dataset_dir, args.test_manifest, "test.jsonl"),
    )


def _validate_split_manifests(manifests: Tuple[Path, Path, Path]):
    for split_manifest in manifests:
        if not split_manifest.exists():
            raise FileNotFoundError(f"Missing split manifest: {split_manifest}")


class DatasetRefreshCallback(Callback):
    def __init__(self, command: str, cwd: Path):
        super().__init__()
        self.command = command
        self.cwd = cwd

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.max_epochs is not None) and (trainer.max_epochs > 0) and (
            trainer.current_epoch + 1 >= trainer.max_epochs
        ):
            return

        if hasattr(pl_module, "drop_train_dataset"):
            pl_module.drop_train_dataset()

        refresh_error = None
        if trainer.is_global_zero:
            try:
                _run_dataset_refresh(
                    self.command, DATASET_REFRESH_TRAIN_MODE, cwd=self.cwd
                )
            except Exception as err:
                refresh_error = f"{type(err).__name__}: {err}"

        world_size = getattr(trainer, "world_size", 1)
        if world_size > 1:
            refresh_error = trainer.strategy.broadcast(refresh_error, src=0)

        if refresh_error is not None:
            raise RuntimeError(f"Dataset refresh command failed: {refresh_error}")

        if world_size > 1:
            trainer.strategy.barrier("dataset_refresh_done")

        if not hasattr(pl_module, "reload_train_dataset"):
            raise RuntimeError(
                "Dataset refresh callback requires model.reload_train_dataset()"
            )
        pl_module.reload_train_dataset()

        if world_size > 1:
            trainer.strategy.barrier("dataset_reload_done")


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    parser.add_argument(
        "--dataset-refresh-command",
        help="Command that regenerates dataset manifests; mode argument is appended",
    )
    parser.add_argument(
        "--train-manifest",
        help="Path to train split manifest (JSONL). Relative paths are under --dataset-dir",
    )
    parser.add_argument(
        "--val-manifest",
        help="Path to val split manifest (JSONL). Relative paths are under --dataset-dir",
    )
    parser.add_argument(
        "--test-manifest",
        help="Path to test split manifest (JSONL). Relative paths are under --dataset-dir",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"
    split_manifests = _resolve_split_manifests(args)

    if args.dataset_refresh_command:
        _run_dataset_refresh(
            args.dataset_refresh_command,
            DATASET_REFRESH_FULL_MODE,
            cwd=args.dataset_dir,
        )

        reload_every = getattr(args, "reload_dataloaders_every_n_epochs", 0) or 0
        if reload_every < 1:
            args.reload_dataloaders_every_n_epochs = 1
            _LOGGER.info(
                "Set reload_dataloaders_every_n_epochs=1 for per-epoch dataset refresh"
            )

    if split_manifests is not None:
        _validate_split_manifests(split_manifests)

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    callbacks = []
    if args.checkpoint_epochs is not None:
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.checkpoint_epochs,
                save_top_k=-1,
            )
        )
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )

    if args.dataset_refresh_command:
        callbacks.append(
            DatasetRefreshCallback(
                command=args.dataset_refresh_command,
                cwd=args.dataset_dir,
            )
        )

    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model_kwargs = dict(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        **dict_args,
    )
    if split_manifests is None:
        model_kwargs["dataset"] = [dataset_path]
    else:
        train_manifest, val_manifest, test_manifest = split_manifests
        model_kwargs["train_dataset"] = [train_manifest]
        model_kwargs["val_dataset"] = [val_manifest]
        model_kwargs["test_dataset"] = [test_manifest]

    model = VitsModel(**model_kwargs)

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
            train_dataset=None,
            val_dataset=None,
            test_dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
