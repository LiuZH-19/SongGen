import os
import re
import shutil
from dataclasses import field
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import concatenate_datasets, load_from_disk
import wandb
from wandb import Audio
from datasets import load_from_disk, concatenate_datasets
import numpy as np


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+)-epoch-(\d+)$")
CHECKPOINT_CODEC_PREFIX = "checkpoint"
_RE_CODEC_CHECKPOINT = re.compile(r"^checkpoint-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _RE_CHECKPOINT.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))


def sorted_checkpoints(output_dir=None, checkpoint_prefix="checkpoint") -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None, checkpoint_prefix="checkpoint", logger=None) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir, checkpoint_prefix=checkpoint_prefix)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        shutil.rmtree(checkpoint, ignore_errors=True)


def save_codec_checkpoint(output_dir, dataset, step):
    checkpoint_path = f"{CHECKPOINT_CODEC_PREFIX}-{step}"
    output_path = os.path.join(output_dir, checkpoint_path)
    dataset.save_to_disk(output_path)


def load_codec_checkpoint(checkpoint_path):
    dataset = load_from_disk(checkpoint_path)
    return dataset


def sorted_codec_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_CODEC_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_CODEC_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def load_all_codec_checkpoints(output_dir=None) -> List[str]:
    """Helper function to load and concat all checkpoints."""
    checkpoints_sorted = sorted_codec_checkpoints(output_dir=output_dir)
    datasets = [load_from_disk(checkpoint) for checkpoint in checkpoints_sorted]
    datasets = concatenate_datasets(datasets, axis=0)
    return datasets


def get_last_codec_checkpoint_step(folder) -> int:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CODEC_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return 0
    last_checkpoint = os.path.join(
        folder, max(checkpoints, key=lambda x: int(_RE_CODEC_CHECKPOINT.search(x).groups()[0]))
    )
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+)"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    return cur_step


def log_metric(
    accelerator,
    metrics: Dict,
    train_time: float,
    step: int,
    epoch: int,
    learning_rate: float = None,
    prefix: str = "train",
):
    """Helper function to log all training/evaluation metrics with the correct prefixes and styling."""
    log_metrics = {}
    for k, v in metrics.items():
        log_metrics[f"{prefix}/{k}"] = v
    log_metrics[f"{prefix}/time"] = train_time
    log_metrics[f"{prefix}/epoch"] = epoch
    if learning_rate is not None:
        log_metrics[f"{prefix}/learning_rate"] = learning_rate
    accelerator.log(log_metrics, step=step)



def remove_trailing_padding(audio: np.ndarray): 
    #-1 D squence_len
    non_zero_indices = np.where(audio != 0)[-1]
    if non_zero_indices.size == 0:
        return audio[..., :1]
    # non_zero_index = np.max(np.where(audio != 0)[-1])
    non_zero_index = np.max(non_zero_indices)
    return audio[..., :non_zero_index + 1]

def log_pred(
    accelerator,
    pred_descriptions: List[str],
    pred_prompts: List[str],
    transcriptions: List[str],
    audios: List[torch.Tensor],
    si_sdr_measures: List[float],
    sampling_rate: int,
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        if si_sdr_measures is None:
            # convert str data to a wandb compatible format
            str_data = [
                [pred_descriptions[i], pred_prompts[i], transcriptions[i]] for i in range(len(pred_descriptions))
            ]
            # log as a table with the appropriate headers
            wandb_tracker.log_table(
                table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
                columns=["Target descriptions", "Target prompts", "Predicted transcriptions"],
                data=str_data[:num_lines],
                step=step,
                commit=False,
            )
        else:
            # convert str data to a wandb compatible format
            str_data = [
                [pred_descriptions[i], pred_prompts[i], transcriptions[i], si_sdr_measures[i]]
                for i in range(len(pred_descriptions))
            ]
            # log as a table with the appropriate headers
            wandb_tracker.log_table(
                table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
                columns=["Target descriptions", "Target prompts", "Predicted transcriptions", "Noise estimation"],
                data=str_data[:num_lines],
                step=step,
                commit=False,
            )

        # wandb can only loads 100 audios per step
        wandb_tracker.log(
            {
                "Generated samples": [
                    Audio(
                        remove_trailing_padding(audio),
                        caption=f"{pred_prompts[i]}\nDESCRIPTION: {pred_descriptions[i]}",
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(audios[: min(len(audios), 100)])
                ]
            },
            step=step,
        )

     

def log_gt(
    accelerator,
    pred_descriptions: List[str],
    pred_prompts: List[str],
    sampling_rate: int,
    gt_labels: List[torch.Tensor] =None,
    gt_label_lens: List[int] =None,
    step: int = 1,
    prefix: str = "ground_truth", 
    gt_refaudios: Optional[List[torch.Tensor]] =None,
    gt_refaudio_lens: Optional[List[int]] =None,
    gt_refvoices: Optional[List[torch.Tensor]] =None,
    refvoice_sr: Optional[int]=None,   
):
    """Helper function to log  gt ref_voice and lables (target_audio) to wandb."""
    print(f"accelerator.is_main_process:{accelerator.is_main_process}, gt_labels is not None:{gt_labels is not None}, gt_refaudios is not None:{gt_refaudios is not None}, gt_refvoices is not None:{gt_refvoices is not None}")
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")
        # wandb can only loads 100 audios per step
        if gt_labels is not None:
            wandb_tracker.log(
                {
                    "labels": [
                        Audio(
                            audio[:gt_label_lens[i]],
                            caption=f"Lyrics:{pred_prompts[i]}",
                            sample_rate=sampling_rate,
                        )
                        for (i, audio) in enumerate(gt_labels[: min(len(gt_labels), 100)])
                    ]
                },
                step=step,
            )

        if gt_refaudios is not None:
            # wandb can only loads 100 audios per step
            wandb_tracker.log(
                {
                    "Ref Audios samples": [
                        Audio(
                            audio[:gt_refaudio_lens[i]],
                            caption=f"DESCRIPTION: {pred_descriptions[i]}",
                            sample_rate=sampling_rate,
                        )
                        for (i, audio) in enumerate(gt_refaudios[: min(len(gt_refaudios), 100)])
                    ]
                },
                step=step,
            )

        if gt_refvoices is not None:
            # wandb can only loads 100 audios per step
            wandb_tracker.log(
                {
                    "Ref Voicecs samples": [
                        Audio(
                            audio,
                            caption=f"Lyrics:{pred_prompts[i]}",
                            sample_rate=refvoice_sr,
                        )
                        for (i, audio) in enumerate(gt_refvoices[: min(len(gt_refvoices), 100)])
                    ]
                },
                step=step,
            )
        # wandb.finish() 
        print('log_gt done!')



def log_pred_AV(
    accelerator,
    pred_descriptions: List[str],
    pred_prompts: List[str],
    transcriptions: List[str],
    acc_audios: List[torch.Tensor],
    vocal_audios: List[torch.Tensor],
    audios: List[torch.Tensor],
    si_sdr_measures: List[float],
    sampling_rate: int,
    step: int,
    prefix: str = "eval",
    num_lines: int = 200000,
):
    """Helper function to log target/predicted transcriptions to weights and biases (wandb)."""
    if accelerator.is_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        # pretty name for current step: step 50000 -> step 50k
        cur_step_pretty = f"{int(step // 1000)}k" if step > 1000 else step
        prefix_pretty = prefix.replace("/", "-")

        if si_sdr_measures is None:
            # convert str data to a wandb compatible format
            str_data = [
                [pred_descriptions[i], pred_prompts[i], transcriptions[i]] for i in range(len(pred_descriptions))
            ]
            # log as a table with the appropriate headers
            wandb_tracker.log_table(
                table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
                columns=["Target descriptions", "Target prompts", "Predicted transcriptions"],
                data=str_data[:num_lines],
                step=step,
                commit=False,
            )
        else:
            # convert str data to a wandb compatible format
            str_data = [
                [pred_descriptions[i], pred_prompts[i], transcriptions[i], si_sdr_measures[i]]
                for i in range(len(pred_descriptions))
            ]
            # log as a table with the appropriate headers
            wandb_tracker.log_table(
                table_name=f"predictions/{prefix_pretty}-step-{cur_step_pretty}",
                columns=["Target descriptions", "Target prompts", "Predicted transcriptions", "Noise estimation"],
                data=str_data[:num_lines],
                step=step,
                commit=False,
            )

        # wandb can only loads 100 audios per step
        wandb_tracker.log(
            {
                "Generated MIX samples": [
                    Audio(
                        remove_trailing_padding(audio),
                        caption=f"{pred_prompts[i]}\nDESCRIPTION: {pred_descriptions[i]}",
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(audios[: min(len(audios), 100)])
                ]
            },
            step=step,
        )

        wandb_tracker.log(
            {
                "Generated ACC samples": [
                    Audio(
                        remove_trailing_padding(audio),
                        caption=f"{pred_prompts[i]}\nDESCRIPTION: {pred_descriptions[i]}",
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(acc_audios[: min(len(acc_audios), 100)])
                ]
            },
            step=step,
        )

        wandb_tracker.log(
            {
                "Generated VOCAL samples": [
                    Audio(
                        remove_trailing_padding(audio),
                        caption=f"{pred_prompts[i]}\nDESCRIPTION: {pred_descriptions[i]}",
                        sample_rate=sampling_rate,
                    )
                    for (i, audio) in enumerate(vocal_audios[: min(len(vocal_audios), 100)])
                ]
            },
            step=step,
        )
